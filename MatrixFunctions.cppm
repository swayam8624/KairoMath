module;

#include <cmath>
#include <concepts>
#include <cstddef>
#include <stdexcept>

export module Kairo.Foundation.Math.LinearAlgebra.MatrixFunctions;

import Kairo.Foundation.Math.Vector;
import Kairo.Foundation.Math.DynamicMatrix;
import Kairo.Foundation.Math.LinearAlgebra.LinearSolve;

export namespace kairo::foundation::math
{
    /// Compute the matrix 1-norm: maximum absolute column sum.
    ///
    /// Input: any dynamic matrix.
    /// Output: max_j sum_i abs(A(i, j)).
    /// Task: provide the norm used by scaling-and-squaring matrix functions
    /// without exposing callers to the implementation details of MatrixExp.
    template<FloatingPoint T>
    [[nodiscard]]
    T MatrixOneNorm(
        const DynamicMatrix<T>& matrix) noexcept
    {
        T maxColumnSum =
            T(0);

        for (std::size_t column = 0; column < matrix.Columns(); ++column)
        {
            T columnSum =
                T(0);

            for (std::size_t row = 0; row < matrix.Rows(); ++row)
            {
                columnSum +=
                    std::abs(matrix(row, column));
            }

            maxColumnSum =
                Max(
                    maxColumnSum,
                    columnSum);
        }

        return maxColumnSum;
    }

    /// Compute exp(A), the matrix exponential, using scaling and squaring with
    /// the [13/13] Pade approximant.
    ///
    /// Input: square dynamic matrix A.
    /// Output: dense matrix exponential e^A.
    /// Task: complete the Phase B matrix-function surface with a production
    /// baseline algorithm. Scaling controls the matrix norm before evaluating
    /// the rational Pade approximation; repeated squaring restores the original
    /// scale. This handles diagonal, nilpotent, skew-symmetric, and general
    /// dense square matrices substantially better than a fixed Taylor series.
    template<FloatingPoint T>
    [[nodiscard]]
    DynamicMatrix<T> MatrixExponential(
        const DynamicMatrix<T>& matrix)
    {
        if (matrix.Rows() != matrix.Columns())
        {
            throw std::invalid_argument(
                "MatrixExponential failed: matrix must be square.");
        }

        const std::size_t size =
            matrix.Rows();

        DynamicMatrix<T> identity =
            DynamicMatrix<T>::Identity(size);

        if (size == 0)
        {
            return identity;
        }

        const T norm =
            MatrixOneNorm(matrix);

        if (norm == T(0))
        {
            return identity;
        }

        constexpr T theta13 =
            T(5.371920351148152);

        const int scalePower =
            norm > theta13
                ? static_cast<int>(
                    std::ceil(
                        std::log2(norm / theta13)))
                : 0;

        const T scale =
            std::ldexp(T(1), scalePower);

        const DynamicMatrix<T> a =
            matrix / scale;

        const DynamicMatrix<T> a2 =
            a * a;

        const DynamicMatrix<T> a4 =
            a2 * a2;

        const DynamicMatrix<T> a6 =
            a4 * a2;

        constexpr T b0 = T(64764752532480000.0);
        constexpr T b1 = T(32382376266240000.0);
        constexpr T b2 = T(7771770303897600.0);
        constexpr T b3 = T(1187353796428800.0);
        constexpr T b4 = T(129060195264000.0);
        constexpr T b5 = T(10559470521600.0);
        constexpr T b6 = T(670442572800.0);
        constexpr T b7 = T(33522128640.0);
        constexpr T b8 = T(1323241920.0);
        constexpr T b9 = T(40840800.0);
        constexpr T b10 = T(960960.0);
        constexpr T b11 = T(16380.0);
        constexpr T b12 = T(182.0);
        constexpr T b13 = T(1.0);

        const DynamicMatrix<T> u =
            a *
            (
                a6 * (a6 * b13 + a4 * b11 + a2 * b9) +
                a6 * b7 +
                a4 * b5 +
                a2 * b3 +
                identity * b1
            );

        const DynamicMatrix<T> v =
            a6 * (a6 * b12 + a4 * b10 + a2 * b8) +
            a6 * b6 +
            a4 * b4 +
            a2 * b2 +
            identity * b0;

        DynamicMatrix<T> result =
            Inverse(v - u) * (v + u);

        for (int i = 0; i < scalePower; ++i)
        {
            result =
                result * result;
        }

        return result;
    }

    /// Alias for MatrixExponential, matching common mathematical notation.
    ///
    /// Input: square dynamic matrix A.
    /// Output: e^A.
    /// Task: provide a shorter name for call sites that already operate in a
    /// matrix-function context while keeping MatrixExponential as the explicit
    /// API entry point.
    template<FloatingPoint T>
    [[nodiscard]]
    DynamicMatrix<T> Exp(
        const DynamicMatrix<T>& matrix)
    {
        return MatrixExponential(matrix);
    }

} // namespace kairo::foundation::math
