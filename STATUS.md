# KairoMath Status

Wave: A — foundation certification  
Frozen v1 target: 95/100  
Source gate: complete  
Execution gate: `cmake --build <build> && ctest --test-dir <build> --output-on-failure`

## Frozen v1 scope

KairoMath v1 owns allocation-free engine vectors/matrices/quaternions/transforms, dynamic linear algebra, decomposition/eigen/SVD/statistics/matrix functions, numerical optimization/probability, Tensor storage, autograd, training/checkpointing, mixed precision, and dataset primitives. New unrelated mathematics is not required for the 95 target.

## 95 exit evidence

- Public module surface is complete for the frozen scope.
- Existing correctness suite covers algebra, decomposition, optimization, tensors, autograd, training, dtypes, persistence, and invalid inputs.
- `KairoMath.Certification` adds deterministic conditioned-matrix inversion checks and non-uniform TRS world/local round-trip stress.
- Downstream Geometry, PhysicsMath, Spatial, Renderer, RayTracer, and ML packages consume this contract.
- Future work after certification is performance/backend specialization or bug repair, not feature-count expansion.

## Verification policy

The 95 score is a scope/completeness score. A release is called *verified at a SHA* only after its tests run on that exact SHA. Certification tests are deterministic and contain no network or device dependency.
