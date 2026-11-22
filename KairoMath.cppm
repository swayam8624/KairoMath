// TEACHING MODULE: This is the *public umbrella*, not a second implementation
// of the math library. A consumer that writes `import Kairo.Foundation.Math;`
// receives the exported declarations of the modules below.
//
// C++23 modules: `export module` names this interface unit; `export import`
// makes another module's exported API visible to this module's consumers.
// A normal `import` would let this file use declarations internally without
// re-exporting them. CMake must discover and build the imported interfaces in
// dependency order; the textual order here is a teaching map, not a substitute
// for build-system dependency scanning.
//
// Dependency journey:
//   Vector -> Matrix -> Quaternion -> Transform -> Geometry/Renderer
//   DynamicMatrix -> Solve/Decomposition -> Eigen/SVD -> Statistics
//   Tensor -> Autograd -> Training, with TensorData feeding training batches.
//
// IMPORTANT: Do not assume every optional backend is included here.
// TensorExecution uses Scheduler/SIMD; TensorGPU uses KairoGPU and requires
// native resource/upload/dispatch/readback management. Import those explicitly
// where supported and configure their CMake options separately.
//
// See docs/SOURCE_READING_ORDER.md and docs/MATH_FOUNDATIONS_WALKTHROUGH.md.
// Behavioral evidence belongs in tests/; this file only publishes interfaces.
export module Kairo.Foundation.Math;

// Fixed-size values. Vector defines scalar concepts, elementary operators,
// and direction conventions. Matrix defines column-vector transforms with
// row-major CPU storage. Quaternion and Transform build on those contracts.
// Changes to these APIs propagate into Geometry, Spatial, Physics and Renderer.
export import Kairo.Foundation.Math.Vector;
export import Kairo.Foundation.Math.Matrix;
export import Kairo.Foundation.Math.Quaternion;
export import Kairo.Foundation.Math.Transform;

// Heap-backed algebra and tensor storage have shape, allocation and lifetime
// contracts that differ from stack-owned fixed-size vector/matrix values.
export import Kairo.Foundation.Math.DynamicMatrix;
export import Kairo.Foundation.Math.Tensor;

// Reverse-mode gradients consume tensor operators. Training adds optimizers,
// scheduler/checkpoint state and loss scaling. TensorData defines dataset
// indexing, batching and prefetch; do not confuse it with tensor execution.
export import Kairo.Foundation.Math.TensorAutograd;
export import Kairo.Foundation.Math.TensorTraining;
export import Kairo.Foundation.Math.TensorData;

// Numerical workflows: solve A*x=b without explicitly computing A^-1 when
// practical; decompositions support conditioning, eigen and SVD workflows.
// Each algorithm imposes different shape/rank/definiteness constraints.
export import Kairo.Foundation.Math.LinearAlgebra.LinearSolve;
export import Kairo.Foundation.Math.LinearAlgebra.Decomposition;
export import Kairo.Foundation.Math.LinearAlgebra.Eigen;
export import Kairo.Foundation.Math.LinearAlgebra.SVD;
export import Kairo.Foundation.Math.LinearAlgebra.Statistics;
export import Kairo.Foundation.Math.LinearAlgebra.MatrixFunctions;

// Optimization consumes objective/gradient and often dynamic-matrix APIs;
// probability owns random sampling and statistical primitives. Deterministic
// tests should fix seeds and compare numerical results with tolerances.
export import Kairo.Foundation.Math.Optimization;
export import Kairo.Foundation.Math.Probability;
