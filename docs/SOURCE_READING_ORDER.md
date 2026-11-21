# KairoMath: teaching source reading order

This file is an executable-path-oriented study sequence, **not** a build-order claim. Read [README](../README.md), [CMakeLists](../CMakeLists.txt), and [tests](../tests/) before assuming all optional features compile on every platform.

This package covers mathematical types, matrix conventions, decompositions, tensor storage, differentiation and GPU-oriented adapters. Read the modules in the following progression, then return to their tests and dependent Kairo packages.

1. [Vector.cppm](../Vector.cppm) — read public types, inputs/outputs, invariants and callers before proceeding
2. [Matrix.cppm](../Matrix.cppm) — read public types, inputs/outputs, invariants and callers before proceeding
3. [Quaternion.cppm](../Quaternion.cppm) — read public types, inputs/outputs, invariants and callers before proceeding
4. [Transform.cppm](../Transform.cppm) — read public types, inputs/outputs, invariants and callers before proceeding
5. [DynamicMatrix.cppm](../DynamicMatrix.cppm) — read public types, inputs/outputs, invariants and callers before proceeding
6. [LinearSolve.cppm](../LinearSolve.cppm) — read public types, inputs/outputs, invariants and callers before proceeding
7. [Decomposition.cppm](../Decomposition.cppm) — read public types, inputs/outputs, invariants and callers before proceeding
8. [SVD.cppm](../SVD.cppm) — read public types, inputs/outputs, invariants and callers before proceeding
9. [Tensor.cppm](../Tensor.cppm) — read public types, inputs/outputs, invariants and callers before proceeding
10. [TensorAutograd.cppm](../TensorAutograd.cppm) — read public types, inputs/outputs, invariants and callers before proceeding
11. [TensorTraining.cppm](../TensorTraining.cppm) — read public types, inputs/outputs, invariants and callers before proceeding
12. [TensorExecution.cppm](../TensorExecution.cppm) — read public types, inputs/outputs, invariants and callers before proceeding
13. [TensorGPU.cppm](../TensorGPU.cppm) — read public types, inputs/outputs, invariants and callers before proceeding
14. [KairoMath.cppm](../KairoMath.cppm) — umbrella C++23 module exporting the focused units

## Cross-repository data flow

KairoEngineCore TransformComponent → KairoMath Transform / Vector / Quaternion → KairoGeometry and render/physics consumers; Tensor → TensorAutograd → TensorTraining, with TensorExecution/TensorGPU as optional execution layers.

## How to read each function

For each class/function, identify its **input and output types**, valid ranges, ownership/lifetimes, numerical convention, exceptions/failure states, asymptotic cost where applicable, and the next consumer. Follow `import` declarations to the exact modules they depend on; C++23 `export import` re-exports an interface, and does not turn a dependency into an implementation of the current module.

## Exercises

1. Choose a public API and trace at least two actual caller/callee boundaries through the associated modules.
2. Find a degenerate case (empty data, singular geometry, invalid timestep, zero axis, or stale handle, as applicable) and locate the validated behaviour.
3. Run the tests for the API and write down the invariant each test proves; avoid inferring platform-wide correctness from one passing smoke test.
4. Compare the API's data representation against its downstream physics/rendering consumer, especially coordinate conventions and ownership.
5. Inspect compiler flags and module build requirements in CMake before benchmarking.

## Licensing and provenance

Original repository-authored source and this guide are intended for study, reuse, modification, and further development under [LICENSE](../LICENSE). This does not relicense third-party code, tools, datasets or bundled assets, if present; retain external notices and verify their provenance.
