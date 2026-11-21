# Teaching source map — KairoMath

A C++23 mathematical foundation: read scalar/vector invariants before algorithms that use them.

Follow source links sequentially. For each API/function, identify its input contract, owning data structure, called modules, failure modes, outputs, test and downstream consumers.

## Read in this order

1. [`CMakeLists.txt`](../CMakeLists.txt)
2. [`KairoMath.cppm`](../KairoMath.cppm)
3. [`Vector.cppm`](../Vector.cppm)
4. [`Matrix.cppm`](../Matrix.cppm)
5. [`Quaternion.cppm`](../Quaternion.cppm)
6. [`Transform.cppm`](../Transform.cppm)
7. [`DynamicMatrix.cppm`](../DynamicMatrix.cppm)
8. [`LinearSolve.cppm`](../LinearSolve.cppm)
9. [`Decomposition.cppm`](../Decomposition.cppm)
10. [`Eigen.cppm`](../Eigen.cppm)
11. [`SVD.cppm`](../SVD.cppm)
12. [`Optimization.cppm`](../Optimization.cppm)
13. [`Probability.cppm`](../Probability.cppm)
14. [`Statistics.cppm`](../Statistics.cppm)
15. [`Tensor.cppm`](../Tensor.cppm)
16. [`TensorAutograd.cppm`](../TensorAutograd.cppm)
17. [`TensorTraining.cppm`](../TensorTraining.cppm)
18. [`tests`](../tests)

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
