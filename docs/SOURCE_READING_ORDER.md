# Teaching source map — KairoMath

A C++23 mathematical foundation: read scalar/vector invariants before algorithms that use them.

Follow source links sequentially. For each API/function, identify its input contract, owning data structure, called modules, failure modes, outputs, test and downstream consumers.

**Worked tensor execution exercise:** [Trace shapes, views, CPU scheduling, SIMD and numerical parity end-to-end](TENSOR_EXECUTION_END_TO_END_LAB.md). Read this after Stage C source modules.\n\n**Worked mathematical exercise:** [Pivoting, LUP factor reuse, orientation quaternions and SLerp with calculations](LINEAR_SOLVERS_AND_ROTATIONS_WORKED_LAB.md). Compare source behavior with a hand-derived answer before attempting performance improvements.

## Read in this order

The stages below are a *learning route*, not a claim about compiler build order.
Follow the links and compare each module's actual imports against the CMake
file set before changing module dependencies.

### Stage A — Public API and geometry foundations

1. [`CMakeLists.txt`](../CMakeLists.txt): what the build compiles and which backends are optional.
2. [`KairoMath.cppm`](../KairoMath.cppm): understand `export module` versus `export import`; note exclusions.
3. [`Vector.cppm`](../Vector.cppm): scalar concepts, norms, cross product, guards for near-zero input.
4. [`Matrix.cppm`](../Matrix.cppm): row-major CPU storage versus column-vector multiplication; transforms/projections.
5. [`Quaternion.cppm`](../Quaternion.cppm): rotation representation, normalization, hemisphere-aware interpolation.
6. [`Transform.cppm`](../Transform.cppm): translate/rotate/scale composition; why shear cannot always fit in TRS.
7. [`tests/MathTests.cpp`](../tests/MathTests.cpp): confirm vector/matrix/transform conventions.

**Checkpoint:** demonstrate why `M * v` and `A * B * v` have their stated
meaning, and identify a scale/rotation combination that cannot be inverted
as another pure TRS object.

### Stage B — Dynamic and numerical linear algebra

8. [`DynamicMatrix.cppm`](../DynamicMatrix.cppm): buffer ownership, dimensions and bounds.
9. [`LinearSolve.cppm`](../LinearSolve.cppm): direct solving, elimination and pivot assumptions.
10. [`Decomposition.cppm`](../Decomposition.cppm): LU/LUP, QR, Cholesky and LDLT domains.
11. [`Eigen.cppm`](../Eigen.cppm): eigenvectors and convergence checks.
12. [`SVD.cppm`](../SVD.cppm): singular values, rank, pseudoinverse and conditioning.
13. [`MatrixFunctions.cppm`](../MatrixFunctions.cppm): the matrix exponential versus scalar elementwise exponential.
14. [`Optimization.cppm`](../Optimization.cppm): gradient methods and nonlinear solvers.
15. [`Statistics.cppm`](../Statistics.cppm): sample axes, covariance/PCA and regression.
16. [`Probability.cppm`](../Probability.cppm): random generators, distribution and seed assumptions.
17. [`tests/FoundationCertification.cpp`](../tests/FoundationCertification.cpp): deterministic numerical campaigns.

**Checkpoint:** solve a well-conditioned and an ill-conditioned system; distinguish
a small residual from an accurate solution.

### Stage C — Tensor storage, learning and hardware integration

18. [`Tensor.cppm`](../Tensor.cppm): shapes, strides, views, contiguity and dtypes.
19. [`TensorAutograd.cppm`](../TensorAutograd.cppm): reverse-mode graph, leaf gradients and backward rules.
20. [`TensorTraining.cppm`](../TensorTraining.cppm): optimizer moments, learning-rate schedules and checkpoints.
21. [`TensorData.cppm`](../TensorData.cppm): indexed datasets, batches, prefetch and worker lifetime.
22. [`TensorExecution.cppm`](../TensorExecution.cppm): optional Scheduler/SIMD path, data races and reductions.
23. [`TensorGPU.cppm`](../TensorGPU.cppm): optional KairoGPU upload, dispatch and readback path.
24. [`tests/TensorAutogradSmoke.cpp`](../tests/TensorAutogradSmoke.cpp) and [`tests/TensorTrainingSmoke.cpp`](../tests/TensorTrainingSmoke.cpp): gradient and resume contracts.
25. [`tests/TensorDataSmoke.cpp`](../tests/TensorDataSmoke.cpp), [`tests/TensorExecutionSmoke.cpp`](../tests/TensorExecutionSmoke.cpp), and [`tests/TensorGPUSmoke.cpp`](../tests/TensorGPUSmoke.cpp): validate only the available host paths.

**Checkpoint:** distinguish logical shape from physical strides; verify a
finite-difference gradient and document where CPU and GPU buffers live.

### Stage D — Visual demonstration and study companion

26. [`visual/VisualTests.cpp`](../visual/VisualTests.cpp): C++ test/visualizer entry point.
27. [`visual/server.py`](../visual/server.py): Python gateway and subprocess boundary; keep it local.
28. [`MATH_FOUNDATIONS_WALKTHROUGH.md`](MATH_FOUNDATIONS_WALKTHROUGH.md): conceptual explanations to reread while following the modules.
29. Continue into the complete inventory below for every first-party test and script.

## How this repository interacts with others

Downstream Geometry, Spatial, PhysicsMath and Renderer import these numerical contracts. Verify matrix convention, transform composition, numerical conditioning and gradient tests before changing APIs.

## Verification and study rules

- Read the public C++23 module/entrypoint first, then its implementation and tests.
- Explain invariants, ownership, error states, concurrency or numerical assumptions in comments, not trivial syntax.
- A passing CI job covers only tests actually run on that commit/host; source presence is not runtime certification.
- Check [STATUS.md](../STATUS.md) and [STATUS.yaml](../STATUS.yaml) for current explicit verification limitations.
- Original work may be modified and extended under the repository license; third-party content retains upstream terms.


## Complete source-file inventory

After the curated teaching path, follow this deterministic inventory of first-party source, scripts, shaders, and tests. Grouping is a suggested learning order, not an indication that every function is already annotated. Vendored dependencies, ROMs, binaries, data, media, and generated evidence are excluded.

### Entry points and build (1)

1. [`CMakeLists.txt`](../CMakeLists.txt)

### Interfaces and modules (20)

1. [`Decomposition.cppm`](../Decomposition.cppm)
2. [`DynamicMatrix.cppm`](../DynamicMatrix.cppm)
3. [`Eigen.cppm`](../Eigen.cppm)
4. [`KairoMath.cppm`](../KairoMath.cppm)
5. [`LinearSolve.cppm`](../LinearSolve.cppm)
6. [`Matrix.cppm`](../Matrix.cppm)
7. [`MatrixFunctions.cppm`](../MatrixFunctions.cppm)
8. [`Optimization.cppm`](../Optimization.cppm)
9. [`Probability.cppm`](../Probability.cppm)
10. [`Quaternion.cppm`](../Quaternion.cppm)
11. [`Statistics.cppm`](../Statistics.cppm)
12. [`SVD.cppm`](../SVD.cppm)
13. [`Tensor.cppm`](../Tensor.cppm)
14. [`TensorAutograd.cppm`](../TensorAutograd.cppm)
15. [`TensorData.cppm`](../TensorData.cppm)
16. [`TensorExecution.cppm`](../TensorExecution.cppm)
17. [`TensorGPU.cppm`](../TensorGPU.cppm)
18. [`TensorTraining.cppm`](../TensorTraining.cppm)
19. [`Transform.cppm`](../Transform.cppm)
20. [`Vector.cppm`](../Vector.cppm)

### Implementation and other source (2)

1. [`visual/server.py`](../visual/server.py)
2. [`visual/VisualTests.cpp`](../visual/VisualTests.cpp)

### Verification and tests (9)

1. [`scratch/update_visual_tests.py`](../scratch/update_visual_tests.py)
2. [`tests/FoundationCertification.cpp`](../tests/FoundationCertification.cpp)
3. [`tests/MathTests.cpp`](../tests/MathTests.cpp)
4. [`tests/TensorAutogradSmoke.cpp`](../tests/TensorAutogradSmoke.cpp)
5. [`tests/TensorDataSmoke.cpp`](../tests/TensorDataSmoke.cpp)
6. [`tests/TensorDTypeSmoke.cpp`](../tests/TensorDTypeSmoke.cpp)
7. [`tests/TensorExecutionSmoke.cpp`](../tests/TensorExecutionSmoke.cpp)
8. [`tests/TensorGPUSmoke.cpp`](../tests/TensorGPUSmoke.cpp)
9. [`tests/TensorTrainingSmoke.cpp`](../tests/TensorTrainingSmoke.cpp)

Snapshot totals: 32 first-party source/build/test files, 46 tracked files overall. Re-audit when files are renamed or added.
