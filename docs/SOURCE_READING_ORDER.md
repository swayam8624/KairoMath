# Teaching source map — KairoMath

A C++23 mathematical foundation: read scalar/vector invariants before algorithms that use them.

Follow source links sequentially. For each API/function, identify its input contract, owning data structure, called modules, failure modes, outputs, test and downstream consumers.

## Read in this order

1. [`CMakeLists.txt`](CMakeLists.txt)
2. [`KairoMath.cppm`](KairoMath.cppm)
3. [`Vector.cppm`](Vector.cppm)
4. [`Matrix.cppm`](Matrix.cppm)
5. [`Quaternion.cppm`](Quaternion.cppm)
6. [`Transform.cppm`](Transform.cppm)
7. [`DynamicMatrix.cppm`](DynamicMatrix.cppm)
8. [`LinearSolve.cppm`](LinearSolve.cppm)
9. [`Decomposition.cppm`](Decomposition.cppm)
10. [`Eigen.cppm`](Eigen.cppm)
11. [`SVD.cppm`](SVD.cppm)
12. [`Optimization.cppm`](Optimization.cppm)
13. [`Probability.cppm`](Probability.cppm)
14. [`Statistics.cppm`](Statistics.cppm)
15. [`Tensor.cppm`](Tensor.cppm)
16. [`TensorAutograd.cppm`](TensorAutograd.cppm)
17. [`TensorTraining.cppm`](TensorTraining.cppm)
18. [`tests`](tests)

## How this repository interacts with others

Downstream Geometry, Spatial, PhysicsMath and Renderer import these numerical contracts. Verify matrix convention, transform composition, numerical conditioning and gradient tests before changing APIs.

## Verification and study rules

- Read the public C++23 module/entrypoint first, then its implementation and tests.
- Explain invariants, ownership, error states, concurrency or numerical assumptions in comments, not trivial syntax.
- A passing CI job covers only tests actually run on that commit/host; source presence is not runtime certification.
- Check [STATUS.md](STATUS.md) and [STATUS.yaml](STATUS.yaml) for current explicit verification limitations.
- Original work may be modified and extended under the repository license; third-party content retains upstream terms.
