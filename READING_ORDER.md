# KairoMath — source reading order and API notebook

Begin with scalar/vector semantics and coordinate conventions, then matrix/quaternion/transform behavior, dynamic linear algebra and decomposition, finally numerical solvers and optimization. KairoGeometry, KairoRenderer, PhysicsMath and AI consumers depend on these mathematical contracts. Explain numerical conditioning and unit/handedness assumptions while reading.

**Before reading:** [README.md](README.md) for setup, [CMakeLists.txt](CMakeLists.txt) if present for module boundaries, [LICENSE](LICENSE) for original-code reuse. Read each exported/public interface, implementation, its imported sibling APIs and matching tests. The sequence below links **all tracked implementation and test source files** in a suggested conceptual order; it is not an execution trace or a claim of complete verification.

## Core API and interaction questions

API contracts: Vec/Mat/Quat/Transform representation, row/column conventions, projection depth range, numerical tolerances, error handling, solver conditioning, and tests for degeneracy.

## Ordered source files (31)

1. [`KairoMath.cppm`](KairoMath.cppm) — identify this file's public imports/includes, state ownership, exported functions, edge cases, and call sites in the next files or tests.
2. [`Vector.cppm`](Vector.cppm) — identify this file's public imports/includes, state ownership, exported functions, edge cases, and call sites in the next files or tests.
3. [`DynamicMatrix.cppm`](DynamicMatrix.cppm) — identify this file's public imports/includes, state ownership, exported functions, edge cases, and call sites in the next files or tests.
4. [`Matrix.cppm`](Matrix.cppm) — identify this file's public imports/includes, state ownership, exported functions, edge cases, and call sites in the next files or tests.
5. [`Quaternion.cppm`](Quaternion.cppm) — identify this file's public imports/includes, state ownership, exported functions, edge cases, and call sites in the next files or tests.
6. [`Transform.cppm`](Transform.cppm) — identify this file's public imports/includes, state ownership, exported functions, edge cases, and call sites in the next files or tests.
7. [`LinearSolve.cppm`](LinearSolve.cppm) — identify this file's public imports/includes, state ownership, exported functions, edge cases, and call sites in the next files or tests.
8. [`Decomposition.cppm`](Decomposition.cppm) — identify this file's public imports/includes, state ownership, exported functions, edge cases, and call sites in the next files or tests.
9. [`Eigen.cppm`](Eigen.cppm) — identify this file's public imports/includes, state ownership, exported functions, edge cases, and call sites in the next files or tests.
10. [`SVD.cppm`](SVD.cppm) — identify this file's public imports/includes, state ownership, exported functions, edge cases, and call sites in the next files or tests.
11. [`Optimization.cppm`](Optimization.cppm) — identify this file's public imports/includes, state ownership, exported functions, edge cases, and call sites in the next files or tests.
12. [`Probability.cppm`](Probability.cppm) — identify this file's public imports/includes, state ownership, exported functions, edge cases, and call sites in the next files or tests.
13. [`MatrixFunctions.cppm`](MatrixFunctions.cppm) — identify this file's public imports/includes, state ownership, exported functions, edge cases, and call sites in the next files or tests.
14. [`scratch/update_visual_tests.py`](scratch/update_visual_tests.py) — identify this file's public imports/includes, state ownership, exported functions, edge cases, and call sites in the next files or tests.
15. [`Statistics.cppm`](Statistics.cppm) — identify this file's public imports/includes, state ownership, exported functions, edge cases, and call sites in the next files or tests.
16. [`Tensor.cppm`](Tensor.cppm) — identify this file's public imports/includes, state ownership, exported functions, edge cases, and call sites in the next files or tests.
17. [`TensorAutograd.cppm`](TensorAutograd.cppm) — identify this file's public imports/includes, state ownership, exported functions, edge cases, and call sites in the next files or tests.
18. [`TensorData.cppm`](TensorData.cppm) — identify this file's public imports/includes, state ownership, exported functions, edge cases, and call sites in the next files or tests.
19. [`TensorExecution.cppm`](TensorExecution.cppm) — identify this file's public imports/includes, state ownership, exported functions, edge cases, and call sites in the next files or tests.
20. [`TensorGPU.cppm`](TensorGPU.cppm) — identify this file's public imports/includes, state ownership, exported functions, edge cases, and call sites in the next files or tests.
21. [`TensorTraining.cppm`](TensorTraining.cppm) — identify this file's public imports/includes, state ownership, exported functions, edge cases, and call sites in the next files or tests.
22. [`tests/FoundationCertification.cpp`](tests/FoundationCertification.cpp) — identify this file's public imports/includes, state ownership, exported functions, edge cases, and call sites in the next files or tests.
23. [`tests/MathTests.cpp`](tests/MathTests.cpp) — identify this file's public imports/includes, state ownership, exported functions, edge cases, and call sites in the next files or tests.
24. [`tests/TensorAutogradSmoke.cpp`](tests/TensorAutogradSmoke.cpp) — identify this file's public imports/includes, state ownership, exported functions, edge cases, and call sites in the next files or tests.
25. [`tests/TensorDataSmoke.cpp`](tests/TensorDataSmoke.cpp) — identify this file's public imports/includes, state ownership, exported functions, edge cases, and call sites in the next files or tests.
26. [`tests/TensorDTypeSmoke.cpp`](tests/TensorDTypeSmoke.cpp) — identify this file's public imports/includes, state ownership, exported functions, edge cases, and call sites in the next files or tests.
27. [`tests/TensorExecutionSmoke.cpp`](tests/TensorExecutionSmoke.cpp) — identify this file's public imports/includes, state ownership, exported functions, edge cases, and call sites in the next files or tests.
28. [`tests/TensorGPUSmoke.cpp`](tests/TensorGPUSmoke.cpp) — identify this file's public imports/includes, state ownership, exported functions, edge cases, and call sites in the next files or tests.
29. [`tests/TensorTrainingSmoke.cpp`](tests/TensorTrainingSmoke.cpp) — identify this file's public imports/includes, state ownership, exported functions, edge cases, and call sites in the next files or tests.
30. [`visual/server.py`](visual/server.py) — identify this file's public imports/includes, state ownership, exported functions, edge cases, and call sites in the next files or tests.
31. [`visual/VisualTests.cpp`](visual/VisualTests.cpp) — identify this file's public imports/includes, state ownership, exported functions, edge cases, and call sites in the next files or tests.

## Use it as a learning exercise

Choose one API, write its preconditions/postconditions and units, trace the caller in the next layer, and create a minimal deterministic test for the happy case, an invalid-input case and a precision/boundary case. Comment each function's **reason, invariant, output and interaction with other files**, not merely its syntax. Original code is [MIT-licensed](LICENSE) for studying and further development; external dependencies retain their own licenses. Do not infer a finished product or a passed native build from this inventory.
