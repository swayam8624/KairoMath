// TEACHING MAP — KairoMath.cppm
// C++23 umbrella exports Vector, Matrix, Quaternion, Transform and numerical/tensor tools.
// Geometry, Renderer and PhysicsMath import these contracts; module imports do not
// implicitly choose units or coordinate spaces.
// Read READING_ORDER.md for math conventions, caller modules and related tests.

export module Kairo.Foundation.Math;

export import Kairo.Foundation.Math.Vector;
export import Kairo.Foundation.Math.Matrix;
export import Kairo.Foundation.Math.Quaternion;
export import Kairo.Foundation.Math.Transform;
export import Kairo.Foundation.Math.DynamicMatrix;
export import Kairo.Foundation.Math.Tensor;
export import Kairo.Foundation.Math.TensorAutograd;
export import Kairo.Foundation.Math.TensorTraining;
export import Kairo.Foundation.Math.TensorData;
export import Kairo.Foundation.Math.LinearAlgebra.LinearSolve;
export import Kairo.Foundation.Math.LinearAlgebra.Decomposition;
export import Kairo.Foundation.Math.LinearAlgebra.Eigen;
export import Kairo.Foundation.Math.LinearAlgebra.SVD;
export import Kairo.Foundation.Math.LinearAlgebra.Statistics;
export import Kairo.Foundation.Math.LinearAlgebra.MatrixFunctions;
export import Kairo.Foundation.Math.Optimization;
export import Kairo.Foundation.Math.Probability;
