module;

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

export module Kairo.Foundation.Math.LinearAlgebra.Decomposition;

import Kairo.Foundation.Math.Vector;
import Kairo.Foundation.Math.DynamicMatrix;

export namespace kairo::foundation::math
{
    /// Result structure for LU decomposition.
    template<FloatingPoint T>
    struct LUResult
    {
        DynamicMatrix<T> L;
        DynamicMatrix<T> U;
    };

    /// Result structure for LUP decomposition.
    template<FloatingPoint T>
    struct LUPResult
    {
        DynamicMatrix<T> L;
        DynamicMatrix<T> U;
        std::vector<std::size_t> P; // Permutation vector
    };

    /// Result structure for QR decomposition.
    template<FloatingPoint T>
    struct QRResult
    {
        DynamicMatrix<T> Q;
        DynamicMatrix<T> R;
    };

    /// Result structure for LDLT decomposition.
    template<FloatingPoint T>
    struct LDLTResult
    {
        DynamicMatrix<T> L;
        std::vector<T> D;
    };

    /// Input: a vector of floating-point values.
    /// Output: Euclidean 2-norm computed through std::hypot.
    /// Task: avoid `sum += x*x` overflow for huge values and underflow for tiny
    /// values in Householder-style algorithms.
    template<FloatingPoint T>
    [[nodiscard]]
    T RobustNorm2(const std::vector<T>& values) noexcept
    {
        T norm = T(0);
        for (T value : values)
        {
            norm = std::hypot(norm, value);
        }

        return norm;
    }

    /// Input: a matrix.
    /// Output: largest absolute entry, or zero for an empty matrix.
    /// Task: give decomposition checks a single scale estimate instead of
    /// duplicating max-abs scans in every algorithm.
    template<FloatingPoint T>
    [[nodiscard]]
    T MaxAbsEntry(const DynamicMatrix<T>& A) noexcept
    {
        T maxAbs = T(0);
        for (std::size_t r = 0; r < A.Rows(); ++r)
        {
            for (std::size_t c = 0; c < A.Columns(); ++c)
            {
                maxAbs = std::max(maxAbs, std::abs(A(r, c)));
            }
        }

        return maxAbs;
    }

    /// Input: matrix scale.
    /// Output: absolute tolerance suitable for symmetry/pivot checks.
    /// Task: make approximate comparisons scale with the matrix while still
    /// allowing zero/small matrices to use a nonzero machine-epsilon floor.
    template<FloatingPoint T>
    [[nodiscard]]
    T DecompositionTolerance(T maxAbs) noexcept
    {
        return std::max(maxAbs, T(1)) *
            std::numeric_limits<T>::epsilon() *
            T(100);
    }

    /// Input: square matrix and tolerance.
    /// Output: true when A(i,j) approximately equals A(j,i).
    /// Task: validate the full matrix contract required by Cholesky and LDLT.
    template<FloatingPoint T>
    [[nodiscard]]
    bool IsApproximatelySymmetric(
        const DynamicMatrix<T>& A,
        T tolerance = T(-1))
    {
        if (A.Rows() != A.Columns())
        {
            return false;
        }

        if (tolerance < T(0))
        {
            tolerance = DecompositionTolerance(MaxAbsEntry(A));
        }

        for (std::size_t r = 0; r < A.Rows(); ++r)
        {
            for (std::size_t c = r + 1; c < A.Columns(); ++c)
            {
                if (std::abs(A(r, c) - A(c, r)) > tolerance)
                {
                    return false;
                }
            }
        }

        return true;
    }

    /// Input: candidate symmetric matrix and decomposition name.
    /// Output: throws when the full matrix is not approximately symmetric.
    /// Task: prevent lower-triangle-only Cholesky/LDLT behavior from silently
    /// accepting invalid full matrices.
    template<FloatingPoint T>
    void ValidateSymmetricMatrix(
        const DynamicMatrix<T>& A,
        const char* decompositionName)
    {
        if (!IsApproximatelySymmetric(A))
        {
            throw std::invalid_argument(
                std::string(decompositionName) +
                " decomposition failed: Matrix must be symmetric.");
        }
    }

    /// LU Decomposition (A = L * U).
    /// Assumes no pivoting is needed. Throws if a zero pivot is encountered.
    template<FloatingPoint T>
    [[nodiscard]]
    LUResult<T> LU(const DynamicMatrix<T>& A)
    {
        if (A.Rows() != A.Columns())
        {
            throw std::invalid_argument("LU decomposition failed: Matrix must be square.");
        }
        std::size_t n = A.Rows();

        T maxAbs = T(0);
        for (std::size_t r = 0; r < n; ++r)
        {
            for (std::size_t c = 0; c < n; ++c)
            {
                maxAbs = std::max(maxAbs, std::abs(A(r, c)));
            }
        }
        T threshold = maxAbs * std::numeric_limits<T>::epsilon() * T(10);

        DynamicMatrix<T> L = DynamicMatrix<T>::Identity(n);
        DynamicMatrix<T> U = A;

        for (std::size_t i = 0; i < n; ++i)
        {
            if (std::abs(U(i, i)) <= threshold)
            {
                throw std::runtime_error("LU decomposition failed: Zero pivot encountered. LUP should be used.");
            }

            for (std::size_t j = i + 1; j < n; ++j)
            {
                T factor = U(j, i) / U(i, i);
                L(j, i) = factor;
                for (std::size_t k = i; k < n; ++k)
                {
                    U(j, k) -= factor * U(i, k);
                }
            }
        }

        return { L, U };
    }

    /// LUP Decomposition (P * A = L * U).
    /// Performs partial pivoting.
    template<FloatingPoint T>
    [[nodiscard]]
    LUPResult<T> LUP(const DynamicMatrix<T>& A)
    {
        if (A.Rows() != A.Columns())
        {
            throw std::invalid_argument("LUP decomposition failed: Matrix must be square.");
        }
        std::size_t n = A.Rows();

        T maxAbs = T(0);
        for (std::size_t r = 0; r < n; ++r)
        {
            for (std::size_t c = 0; c < n; ++c)
            {
                maxAbs = std::max(maxAbs, std::abs(A(r, c)));
            }
        }
        T threshold = maxAbs * std::numeric_limits<T>::epsilon() * T(10);

        DynamicMatrix<T> M = A;
        std::vector<std::size_t> P(n);
        std::iota(P.begin(), P.end(), std::size_t(0));

        for (std::size_t i = 0; i < n; ++i)
        {
            // Find pivot row
            std::size_t pivotRow = i;
            T maxVal = std::abs(M(i, i));
            for (std::size_t row = i + 1; row < n; ++row)
            {
                T val = std::abs(M(row, i));
                if (val > maxVal)
                {
                    maxVal = val;
                    pivotRow = row;
                }
            }

            if (maxVal <= threshold)
            {
                throw std::runtime_error("LUP decomposition failed: Singular matrix encountered.");
            }

            if (pivotRow != i)
            {
                M.SwapRows(i, pivotRow);
                std::swap(P[i], P[pivotRow]);
            }

            for (std::size_t j = i + 1; j < n; ++j)
            {
                T factor = M(j, i) / M(i, i);
                M(j, i) = factor; // Store L in strictly lower triangular part
                for (std::size_t k = i + 1; k < n; ++k)
                {
                    M(j, k) -= factor * M(i, k);
                }
            }
        }

        // Unpack L and U from M
        DynamicMatrix<T> L = DynamicMatrix<T>::Identity(n);
        DynamicMatrix<T> U = DynamicMatrix<T>::Zero(n, n);

        for (std::size_t r = 0; r < n; ++r)
        {
            for (std::size_t c = 0; c < n; ++c)
            {
                if (r > c)
                {
                    L(r, c) = M(r, c);
                }
                else
                {
                    U(r, c) = M(r, c);
                }
            }
        }

        return { L, U, P };
    }

    /// QR Decomposition (A = Q * R) using Householder reflections.
    /// Works for general m x n matrices (m >= n).
    template<FloatingPoint T>
    [[nodiscard]]
    QRResult<T> QR(const DynamicMatrix<T>& A)
    {
        std::size_t m = A.Rows();
        std::size_t n = A.Columns();

        if (m < n)
        {
            throw std::invalid_argument("QR decomposition failed: row count must be greater than or equal to column count.");
        }

        DynamicMatrix<T> Q = DynamicMatrix<T>::Identity(m);
        DynamicMatrix<T> R = A;

        for (std::size_t g = 0; g < n; ++g)
        {
            // Vector x = R[g..m-1, g]
            std::size_t len = m - g;
            std::vector<T> x(len);
            for (std::size_t i = 0; i < len; ++i)
            {
                x[i] = R(g + i, g);
            }
            T normX = RobustNorm2(x);
            if (normX <= std::numeric_limits<T>::epsilon() * T(10))
            {
                continue;
            }

            // Alpha and Householder vector v
            T alpha = (x[0] >= T(0)) ? -normX : normX;
            std::vector<T> v(len);
            v[0] = x[0] - alpha;
            for (std::size_t i = 1; i < len; ++i)
            {
                v[i] = x[i];
            }

            // Normalize v
            T vNorm = RobustNorm2(v);
            if (vNorm > std::numeric_limits<T>::epsilon() * T(10))
            {
                for (std::size_t i = 0; i < len; ++i)
                {
                    v[i] /= vNorm;
                }
            }
            else
            {
                continue;
            }

            // Apply H = I - 2*v*v^T to R: R[g..m-1, g..n-1] -= 2 * v * (v^T * R[g..m-1, g..n-1])
            for (std::size_t col = g; col < n; ++col)
            {
                T dotProduct = T(0);
                for (std::size_t row = 0; row < len; ++row)
                {
                    dotProduct += v[row] * R(g + row, col);
                }
                for (std::size_t row = 0; row < len; ++row)
                {
                    R(g + row, col) -= T(2) * v[row] * dotProduct;
                }
            }

            // Apply H to Q: Q[0..m-1, g..m-1] -= 2 * (Q[0..m-1, g..m-1] * v) * v^T
            for (std::size_t row = 0; row < m; ++row)
            {
                T dotProduct = T(0);
                for (std::size_t col = 0; col < len; ++col)
                {
                    dotProduct += Q(row, g + col) * v[col];
                }
                for (std::size_t col = 0; col < len; ++col)
                {
                    Q(row, g + col) -= T(2) * dotProduct * v[col];
                }
            }
        }

        return { Q, R };
    }

    /// Cholesky Decomposition (A = L * L^T).
    /// Matrix A must be symmetric positive-definite.
    template<FloatingPoint T>
    [[nodiscard]]
    DynamicMatrix<T> Cholesky(const DynamicMatrix<T>& A)
    {
        if (A.Rows() != A.Columns())
        {
            throw std::invalid_argument("Cholesky decomposition failed: Matrix must be square.");
        }
        ValidateSymmetricMatrix(A, "Cholesky");
        std::size_t n = A.Rows();

        T maxAbs = MaxAbsEntry(A);
        T threshold = maxAbs * std::numeric_limits<T>::epsilon() * T(10);

        DynamicMatrix<T> L = DynamicMatrix<T>::Zero(n, n);

        for (std::size_t i = 0; i < n; ++i)
        {
            for (std::size_t j = 0; j <= i; ++j)
            {
                T sum = T(0);
                for (std::size_t k = 0; k < j; ++k)
                {
                    sum += L(i, k) * L(j, k);
                }

                if (i == j)
                {
                    T val = A(i, i) - sum;
                    if (val <= T(0))
                    {
                        throw std::runtime_error("Cholesky failed: Matrix is not positive-definite.");
                    }
                    L(i, j) = std::sqrt(val);
                }
                else
                {
                    T diag = L(j, j);
                    if (std::abs(diag) <= threshold)
                    {
                        throw std::runtime_error("Cholesky failed: Singular matrix / division by zero.");
                    }
                    L(i, j) = (A(i, j) - sum) / diag;
                }
            }
        }

        return L;
    }

    /// LDLT Decomposition (A = L * D * L^T).
    /// Matrix A must be symmetric positive-semidefinite (or indefinite, but requires positive pivots).
    template<FloatingPoint T>
    [[nodiscard]]
    LDLTResult<T> LDLT(const DynamicMatrix<T>& A)
    {
        if (A.Rows() != A.Columns())
        {
            throw std::invalid_argument("LDLT decomposition failed: Matrix must be square.");
        }
        ValidateSymmetricMatrix(A, "LDLT");
        std::size_t n = A.Rows();

        T maxAbs = MaxAbsEntry(A);
        T threshold = maxAbs * std::numeric_limits<T>::epsilon() * T(10);

        DynamicMatrix<T> L = DynamicMatrix<T>::Identity(n);
        std::vector<T> D(n, T(0));

        for (std::size_t j = 0; j < n; ++j)
        {
            T sumD = T(0);
            for (std::size_t k = 0; k < j; ++k)
            {
                sumD += L(j, k) * L(j, k) * D[k];
            }
            D[j] = A(j, j) - sumD;

            if (std::abs(D[j]) <= threshold)
            {
                // Pivot is zero, matrix is singular or semi-definite, continue but warn if needed
                D[j] = T(0);
            }

            for (std::size_t i = j + 1; i < n; ++i)
            {
                T sumL = T(0);
                for (std::size_t k = 0; k < j; ++k)
                {
                    sumL += L(i, k) * L(j, k) * D[k];
                }
                if (std::abs(D[j]) > threshold)
                {
                    L(i, j) = (A(i, j) - sumL) / D[j];
                }
                else
                {
                    L(i, j) = T(0);
                }
            }
        }
        return { L, D };
    }

    /// Helper to compute the sign of a permutation vector P.
    /// Returns +1 for even permutations, -1 for odd permutations.
    template<typename T>
    [[nodiscard]]
    T PermutationSign(const std::vector<std::size_t>& P) noexcept
    {
        std::size_t n = P.size();
        std::vector<bool> visited(n, false);
        std::size_t cycles = 0;
        for (std::size_t i = 0; i < n; ++i)
        {
            if (!visited[i])
            {
                ++cycles;
                std::size_t curr = i;
                while (!visited[curr])
                {
                    visited[curr] = true;
                    curr = P[curr];
                }
            }
        }
        return ((n - cycles) % 2 == 0) ? T(1) : T(-1);
    }

    /// Compute determinant of a square matrix using LUP decomposition.
    template<FloatingPoint T>
    [[nodiscard]]
    T Determinant(const DynamicMatrix<T>& A)
    {
        if (A.Rows() != A.Columns())
        {
            throw std::invalid_argument("Determinant failed: Matrix must be square.");
        }
        std::size_t n = A.Rows();
        if (n == 0) return T(1);

        try
        {
            LUPResult<T> lup = LUP(A);
            T det = PermutationSign<T>(lup.P);
            for (std::size_t i = 0; i < n; ++i)
            {
                det *= lup.U(i, i);
            }
            return det;
        }
        catch (const std::runtime_error&)
        {
            // LUP throws if matrix is singular (pivot <= threshold)
            return T(0);
        }
    }

} // namespace kairo::foundation::math
