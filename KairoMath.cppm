// TEACHING NOTE:
// Umbrella module exporting narrow vector, matrix, quaternion, transform, tensor, linear algebra,
// optimization and probability interfaces. Consumers can choose the umbrella or a smaller import.
// See docs/SOURCE_READING_ORDER.md for the consumer chain and exercises.

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
