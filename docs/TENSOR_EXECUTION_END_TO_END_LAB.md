# Advanced tensor execution laboratory — from logical indexing to CPU SIMD

This chapter follows the actual C++23 source, not a hypothetical tensor runtime. Start with [Tensor.cppm](../Tensor.cppm), then [TensorExecution.cppm](../TensorExecution.cppm), the external [KairoScheduler](https://github.com/swayam8624/KairoScheduler/blob/main/src/KairoScheduler.cppm) and [KairoSIMD](https://github.com/swayam8624/KairoSIMD/blob/main/src/KairoSIMD.cppm) modules, and finally [TensorExecutionSmoke.cpp](../tests/TensorExecutionSmoke.cpp).

**Build distinction:** `TensorExecution.cppm` is added to KairoMath's CMake module file set only when `KAIRO_MATH_USE_SCHEDULER=ON`. That configuration locates and links both KairoScheduler and KairoSIMD. `TensorGPU.cppm` is independent and only included when `KAIRO_MATH_USE_GPU=ON`. Importing the ordinary Tensor module does not secretly turn on either backend.

## 1. Four contracts under one operation

Consider `ParallelAdd(scheduler, lhs, rhs)`:

1. **Shape contract:** the tensors must have the same rank and each matching dimension. Equal element counts alone are insufficient: `[2,3]` and `[3,2]` cannot be elementwise-added as if their axes were interchangeable.
2. **Storage contract:** `Tensor<T>` may own contiguous data or represent a strided view; `Data()` is not proof that advancing the pointer by one reaches the next *logical* element.
3. **Scheduling contract:** `Scheduler::For` distributes a count of logical output elements into disjoint half-open ranges. It does not know tensor axes, alias relationships or SIMD lane width.
4. **Kernel contract:** the Float32 contiguous case creates `std::span` slices and invokes `kairo::simd::Add`; otherwise the code iterates `Tensor::operator[]` over logical positions. The output is a new contiguous tensor.

The source sets the output backend metadata to `TensorBackend::SIMD` after the operation, even when the branch ran a scalar loop or sequential scheduling. **Treat that tag as a selected execution-family marker, not proof that SIMD machine instructions ran or that several CPU workers were active.** Establish actual dispatch from the type, contiguity check, instruction set, workload size and profiler.

## 2. Layout is different from shape

For a logically shaped row-major tensor `[2,3]`, contiguous offsets follow `offset = row * 3 + column`. A view might keep the same logical shape with a different stride vector or an offset into shared storage. `ParallelAdd` and `ParallelReLU` use `IsContiguous()` to choose the direct pointer path; otherwise they use the tensor's logical indexing. That preserves meaning without pretending that a transpose is just a flat pointer.

**Learning exercise:** make a 2D tensor and a transposed or sliced view using the actual public view API from [Tensor.cppm](../Tensor.cppm). Confirm logical elements with the scalar operator first. Then call `ParallelAdd` with compatible shapes and check each logical result; do not construct raw spans from a noncontiguous view.

An owning `Tensor<T>` should be kept alive for all operations borrowing its storage; a non-owning `std::span` does not extend the lifetime of that storage. Run-time shape validation does not prevent invalid external lifetimes.

## 3. Step through a threaded Float32 addition

```cpp
// Illustrative client code: import modules made available by the
// optional KAIRO_MATH_USE_SCHEDULER build configuration.
import Kairo.Foundation.Math.Tensor;
import Kairo.Foundation.Math.TensorExecution;
import Kairo.Scheduler;

using kairo::foundation::math::Tensor;
using kairo::foundation::math::ParallelAdd;

kairo::scheduler::Scheduler scheduler({
    .workerCount = 3,
    .defaultMinChunkSize = 2
});

const Tensor<float> a({2, 3}, {1, 2, 3, 4, 5, 6});
const Tensor<float> b({2, 3}, {6, 5, 4, 3, 2, 1});

// Internally: validate -> allocate -> partition -> dispatch/loop -> join.
const Tensor<float> sum = ParallelAdd(scheduler, a, b);
```

The six output values are all seven. Their mathematical correctness does not depend on which worker executes which range: each writes disjoint elements in the fresh output. A task's input spans point to immutable tensors that remain alive until `Scheduler::For` returns; KairoScheduler joins submitted work before returning. In this example, a tiny chunk may still fall through the scalar tail in KairoSIMD, despite using the Float32 overload.

The complete call chain is `ParallelAdd` -> `Scheduler::For` -> free `ParallelFor` -> `PartitionRange` and `TaskGroup` -> `ThreadPool::Submit` -> worker callback -> `kairo::simd::Add` or scalar logical loop -> `TaskGroup::Wait`. See [the Scheduler concurrency laboratory](https://github.com/swayam8624/KairoScheduler/blob/main/docs/ADVANCED_CONCURRENCY_LAB.md) for lifetime, cancellation and nested-blocking hazards.

## 4. Reductions require a second design decision

`ParallelSum` cannot write disjoint final scalar outputs because only one scalar is returned. It computes a local sum per scheduled range, pushes each `(range.begin, partial)` pair under a mutex, sorts partials by source offset, and combines them in ascending range order.

This makes **combination order stable for a fixed partition**, independent of worker completion order. It does **not** guarantee bitwise identity with a serial sum: floating-point addition is sensitive to grouping, the worker count/chunk configuration changes partitions, and the Float32 contiguous path can use KairoSIMD's different reduction order. Numerical tests should compare against a scalar baseline with a tolerance chosen for the algorithm and input scale.

The initial vector of partials is shared but protected by `partialMutex`; each callback's scalar `sum` is local. The mutex is acquired only to append a small partial, not while iterating an entire range. For larger workloads investigate per-chunk preallocated slots if lock contention materially affects measurements; benchmark before claiming improvement.

## 5. Matrix multiplication and packing

The rank-two implementation validates `lhs[M,K]` and `rhs[K,N]`. It materializes both operands with `Contiguous()`, then creates `packedRight[N,K]`: each column of the original right matrix becomes a contiguous row in the packed matrix.

Each scheduled unit owns **one or more output rows**. For Float32, `result(row,column)` is a SIMD dot product of a left-matrix row and one packed-right row. For other supported floating-point types, it uses a scalar inner loop. Thus the K dimension runs contiguously in both inputs, reducing strided accesses in the innermost computation.

This improves locality for some sizes but introduces conversion/packing cost and extra memory. It does not guarantee speedup for very small matrices, skinny matrices, or cases where cache reuse and packing overhead dominate. Compare total end-to-end latency including packing, not just the dot-product inner loop.

For batched multiplication `[...,M,K] x [...,K,N]`, the source validates matching batch dimensions and partitions `batches * rows` independent output rows. The current API does **not** perform general broadcasted batch dimensions; incompatible batch dimensions raise an error.

## 6. Convolution means an explicit data layout

`ParallelConv2DValidNHWC` requires input `[batch,height,width,channels]` and filters `[outputChannels,kernelHeight,kernelWidth,inputChannels]` (OHWI). It validates rank, stride nonzero, channel agreement, and sufficient spatial input size. Output spatial dimensions use the *valid* formula

`outHeight = floor((inputHeight - kernelHeight) / strideHeight) + 1`

and similarly for width. The output's shape is `[batch,outHeight,outWidth,outputChannels]`. This is not an arbitrary padding/dilation/groups convolution, and it must not be documented as one. Verify this source implementation and the scalar `Conv2DValidNHWC` oracle using small hand-computed kernels.

## 7. How to read backend and performance evidence

The current [smoke test](../tests/TensorExecutionSmoke.cpp) configures a three-worker scheduler with minimum chunk size two, then compares parallel add, sum, MatMul, batched MatMul, convolution, and ReLU with scalar results; its `Equal` helper defaults to a fixed absolute tolerance of `1e-5`. This is a helpful starting test, not proof of all dtypes, large shapes, noncontiguous views, overlapping storage, hostile numerical inputs, or actual speedup.

Further tests should deliberately distinguish:
- **Numerical correctness:** scalar reference, small hand-calculated cases, relative/absolute tolerances, zero and non-finite inputs where valid.
- **Layout correctness:** contiguous versus transposed/sliced strided views; rank and dimension mismatch rejection.
- **Thread safety:** output-element ownership and reductions, including sanitized runs on supported systems.
- **Dispatch honesty:** Float32 contiguous and noncontiguous paths, other supported scalar types, architecture feature guards, tiny counts that only execute scalar tails.
- **Performance:** elapsed time with and without packing, worker count, chunk size, memory bandwidth, cold/warm cache, allocation volume and repetitions.

A test that passes on one CPU is not automatically evidence that NEON, AVX2, AVX-512 and the GPU backend were each executed.

## 8. Exercises connecting the repository family

**Exercise A — trace a value:** start at `Tensor<float>` logical indexing, show how `ParallelAdd` creates contiguous output, then trace its range into KairoScheduler and its Float32 slice into KairoSIMD. Identify exactly which object owns each allocation.

**Exercise B — layout parity:** compare a compatible noncontiguous view against its explicit `Contiguous()` materialization, using scalar and parallel add. Confirm equal logical shape and values, not identical pointers or strides.

**Exercise C — reductions:** sum alternating large and tiny values, repeat with different chunk sizes, and compare serial sum, ordered partial sum and a higher-precision reference. Explain the numerical differences rather than labeling all deviations bugs.

**Exercise D — MatMul crossover:** compare scalar MatMul and ParallelMatMul across matrix dimensions and worker counts. Include the right-side packing time. Plot end-to-end throughput alongside numerical error.

**Exercise E — execution evidence:** run CMake with `KAIRO_MATH_USE_SCHEDULER=ON` and explicit sibling repository paths if necessary, then execute `KairoMath.TensorExecutionSmoke` through CTest. Record the exact commit, compiler version, CPU features and result. Do not claim a run without the logs.

**Evidence status:** written from the repository source and available tests. This documentation pass did not compile, execute sanitizers, benchmark the host, or modify runtime behavior.
