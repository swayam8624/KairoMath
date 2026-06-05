module;

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

export module Kairo.Foundation.Math.LinearAlgebra.LinearSolve;

import Kairo.Foundation.Math.Vector;
import Kairo.Foundation.Math.DynamicMatrix;
import Kairo.Foundation.Math.LinearAlgebra.Decomposition;

export namespace kairo::foundation::math
{
    /// Forward substitution to solve L * x = b where L is lower triangular.
    template<FloatingPoint T>
    [[nodiscard]]
    std::vector<T> ForwardSubstitution(const DynamicMatrix<T>& L, const std::vector<T>& b)
    {
        assert(L.Rows() == L.Columns());
        assert(L.Rows() == b.size());
        std::size_t n = b.size();
        std::vector<T> x(n);

        T maxAbs = T(0);
        for (std::size_t r = 0; r < n; ++r)
        {
            for (std::size_t c = 0; c <= r; ++c)
            {
                maxAbs = std::max(maxAbs, std::abs(L(r, c)));
            }
        }
        T threshold = maxAbs * std::numeric_limits<T>::epsilon() * T(10);

        for (std::size_t i = 0; i < n; ++i)
        {
            T sum = T(0);
            for (std::size_t j = 0; j < i; ++j)
            {
                sum += L(i, j) * x[j];
            }
            T diag = L(i, i);
            if (std::abs(diag) <= threshold)
            {
                throw std::runtime_error("Singular matrix in ForwardSubstitution: zero on diagonal.");
            }
            x[i] = (b[i] - sum) / diag;
        }

        return x;
    }

    /// Backward substitution to solve U * x = b where U is upper triangular.
    template<FloatingPoint T>
    [[nodiscard]]
    std::vector<T> BackwardSubstitution(const DynamicMatrix<T>& U, const std::vector<T>& b)
    {
        assert(U.Rows() == U.Columns());
        assert(U.Rows() == b.size());
        std::size_t n = b.size();
        std::vector<T> x(n);

        T maxAbs = T(0);
        for (std::size_t r = 0; r < n; ++r)
        {
            for (std::size_t c = r; c < n; ++c)
            {
                maxAbs = std::max(maxAbs, std::abs(U(r, c)));
            }
        }
        T threshold = maxAbs * std::numeric_limits<T>::epsilon() * T(10);

        for (int i = static_cast<int>(n) - 1; i >= 0; --i)
        {
            T sum = T(0);
            for (std::size_t j = static_cast<std::size_t>(i) + 1; j < n; ++j)
            {
                sum += U(i, j) * x[j];
            }
            T diag = U(i, i);
            if (std::abs(diag) <= threshold)
            {
                throw std::runtime_error("Singular matrix in BackwardSubstitution: zero on diagonal.");
            }
            x[i] = (b[i] - sum) / diag;
        }

        return x;
    }

    /// Convert a matrix to Row Echelon Form (REF) using Gaussian elimination.
    template<FloatingPoint T>
    [[nodiscard]]
    DynamicMatrix<T> RowEchelonForm(const DynamicMatrix<T>& A)
    {
        DynamicMatrix<T> M = A;
        std::size_t numRows = M.Rows();
        std::size_t numCols = M.Columns();

        T maxAbs = T(0);
        for (std::size_t r = 0; r < numRows; ++r)
        {
            for (std::size_t c = 0; c < numCols; ++c)
            {
                maxAbs = std::max(maxAbs, std::abs(A(r, c)));
            }
        }
        T threshold = maxAbs * std::numeric_limits<T>::epsilon() * T(10);

        std::size_t r = 0;
        for (std::size_t c = 0; c < numCols && r < numRows; ++c)
        {
            // Find the first row starting from row r with a non-zero entry in column c
            std::size_t pivotRow = numRows;
            for (std::size_t row = r; row < numRows; ++row)
            {
                if (std::abs(M(row, c)) > threshold)
                {
                    pivotRow = row;
                    break;
                }
            }

            if (pivotRow == numRows)
            {
                continue;
            }

            if (pivotRow != r)
            {
                M.SwapRows(pivotRow, r);
            }

            for (std::size_t row = r + 1; row < numRows; ++row)
            {
                T factor = M(row, c) / M(r, c);
                M(row, c) = T(0); // Zero out column entry to avoid drift
                for (std::size_t col = c + 1; col < numCols; ++col)
                {
                    M(row, col) -= factor * M(r, col);
                }
            }
            ++r;
        }

        return M;
    }

    /// Convert a matrix to Reduced Row Echelon Form (RREF) using Gauss-Jordan elimination.
    template<FloatingPoint T>
    [[nodiscard]]
    DynamicMatrix<T> ReducedRowEchelonForm(const DynamicMatrix<T>& A)
    {
        DynamicMatrix<T> M = A;
        std::size_t numRows = M.Rows();
        std::size_t numCols = M.Columns();

        T maxAbs = T(0);
        for (std::size_t r = 0; r < numRows; ++r)
        {
            for (std::size_t c = 0; c < numCols; ++c)
            {
                maxAbs = std::max(maxAbs, std::abs(A(r, c)));
            }
        }
        T threshold = maxAbs * std::numeric_limits<T>::epsilon() * T(10);

        std::size_t r = 0;
        for (std::size_t c = 0; c < numCols && r < numRows; ++c)
        {
            // Find the first row starting from row r with a non-zero entry in column c
            std::size_t pivotRow = numRows;
            for (std::size_t row = r; row < numRows; ++row)
            {
                if (std::abs(M(row, c)) > threshold)
                {
                    pivotRow = row;
                    break;
                }
            }

            if (pivotRow == numRows)
            {
                continue;
            }

            if (pivotRow != r)
            {
                M.SwapRows(pivotRow, r);
            }

            T pivotVal = M(r, c);
            for (std::size_t col = c; col < numCols; ++col)
            {
                M(r, col) /= pivotVal;
            }

            for (std::size_t row = 0; row < numRows; ++row)
            {
                if (row != r)
                {
                    T factor = M(row, c);
                    M(row, c) = T(0); // Zero out column entry to avoid drift
                    for (std::size_t col = c + 1; col < numCols; ++col)
                    {
                        M(row, col) -= factor * M(r, col);
                    }
                }
            }
            ++r;
        }

        return M;
    }

    /// Solve A * x = b using Gaussian elimination with partial pivoting.
    template<FloatingPoint T>
    [[nodiscard]]
    std::vector<T> GaussianElimination(const DynamicMatrix<T>& A, const std::vector<T>& b)
    {
        assert(A.Rows() == A.Columns());
        assert(A.Rows() == b.size());
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

        // Build augmented matrix [A | b]
        DynamicMatrix<T> M(n, n + 1);
        for (std::size_t r = 0; r < n; ++r)
        {
            for (std::size_t c = 0; c < n; ++c)
            {
                M(r, c) = A(r, c);
            }
            M(r, n) = b[r];
        }

        // Gaussian elimination with partial pivoting
        for (std::size_t i = 0; i < n; ++i)
        {
            // Find row with largest pivot
            std::size_t maxRow = i;
            T maxVal = std::abs(M(i, i));
            for (std::size_t row = i + 1; row < n; ++row)
            {
                T val = std::abs(M(row, i));
                if (val > maxVal)
                {
                    maxVal = val;
                    maxRow = row;
                }
            }

            if (maxVal <= threshold)
            {
                throw std::runtime_error("GaussianElimination error: Singular matrix detected.");
            }

            if (maxRow != i)
            {
                M.SwapRows(i, maxRow);
            }

            // Elimination
            for (std::size_t row = i + 1; row < n; ++row)
            {
                T factor = M(row, i) / M(i, i);
                for (std::size_t col = i; col <= n; ++col)
                {
                    M(row, col) -= factor * M(i, col);
                }
            }
        }

        // Separate back A and b from M
        DynamicMatrix<T> U(n, n);
        std::vector<T> y(n);
        for (std::size_t r = 0; r < n; ++r)
        {
            for (std::size_t c = 0; c < n; ++c)
            {
                U(r, c) = M(r, c);
            }
            y[r] = M(r, n);
        }

        return BackwardSubstitution(U, y);
    }

    /// Solve A * x = b using Gauss-Jordan elimination.
    template<FloatingPoint T>
    [[nodiscard]]
    std::vector<T> GaussJordanElimination(const DynamicMatrix<T>& A, const std::vector<T>& b)
    {
        assert(A.Rows() == A.Columns());
        assert(A.Rows() == b.size());
        std::size_t n = A.Rows();

        // Build augmented matrix [A | b]
        DynamicMatrix<T> M(n, n + 1);
        for (std::size_t r = 0; r < n; ++r)
        {
            for (std::size_t c = 0; c < n; ++c)
            {
                M(r, c) = A(r, c);
            }
            M(r, n) = b[r];
        }

        // Run RREF
        DynamicMatrix<T> R = ReducedRowEchelonForm(M);

        // Verify that R's left n x n part is identity (i.e. matrix was invertible)
        for (std::size_t i = 0; i < n; ++i)
        {
            if (std::abs(R(i, i) - T(1)) > std::numeric_limits<T>::epsilon() * T(100))
            {
                throw std::runtime_error("GaussJordanElimination error: Singular matrix detected.");
            }
        }

        // Extract solution from the last column
        std::vector<T> x(n);
        for (std::size_t r = 0; r < n; ++r)
        {
            x[r] = R(r, n);
        }

        return x;
    }

    /// Generic linear solver: solves A * x = b using Gaussian elimination.
    template<FloatingPoint T>
    [[nodiscard]]
    std::vector<T> LinearSolve(const DynamicMatrix<T>& A, const std::vector<T>& b)
    {
        return GaussianElimination(A, b);
    }

    /// Compute matrix inverse using LUP decomposition.
    template<FloatingPoint T>
    [[nodiscard]]
    DynamicMatrix<T> Inverse(const DynamicMatrix<T>& A)
    {
        if (A.Rows() != A.Columns())
        {
            throw std::invalid_argument("Matrix inversion failed: Matrix must be square.");
        }
        std::size_t n = A.Rows();
        
        LUPResult<T> lup = LUP(A); // Will throw if singular
        
        DynamicMatrix<T> inv(n, n);
        for (std::size_t j = 0; j < n; ++j)
        {
            std::vector<T> ej(n, T(0));
            ej[j] = T(1);
            
            std::vector<T> pb(n);
            for (std::size_t i = 0; i < n; ++i)
            {
                pb[i] = ej[lup.P[i]];
            }
            
            std::vector<T> y = ForwardSubstitution(lup.L, pb);
            std::vector<T> xj = BackwardSubstitution(lup.U, y);
            
            for (std::size_t r = 0; r < n; ++r)
            {
                inv(r, j) = xj[r];
            }
        }
        return inv;
    }

} // namespace kairo::foundation::math
