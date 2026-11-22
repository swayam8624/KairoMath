// TEACHING MODULE: Linear system solvers: forward/back substitution, numerical pivoting
// and decomposition-backed solve paths. Solving A*x=b is different from
// forming inv(A); conditioning, near-zero pivots and residual checks matter.
// Optimization and inverse/least-squares tools consume these methods.
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
    /// TEACHING API: ForwardSubstitution — Solve Lx=b row by row; small diagonal entries make the triangular system singular/ill-conditioned.
    // Solve L*x=b using already-computed x[0..i). The diagonal is the
    // only unknown of row i: x[i]=(b[i]-sum_{j<i} Lij*xj)/Lii.
    // Use a scale-dependent pivot threshold to reject a numerically
    // unusable near-zero diagonal, not only a literal zero.
    // KairoMath::LUP reuses this for many right-hand sides.
    std::vector<T> ForwardSubstitution(const DynamicMatrix<T>& L, const std::vector<T>& b)
    {
        if (L.Rows() != L.Columns())
        {
            throw std::invalid_argument("ForwardSubstitution failed: matrix must be square.");
        }
        if (L.Rows() != b.size())
        {
            throw std::invalid_argument("ForwardSubstitution failed: matrix rows must match b size.");
        }

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
    /// TEACHING API: BackwardSubstitution — Solve Ux=b from final row upward, respecting matrix size and near-zero pivot checks.
    // Solve U*x=b from the final row toward row zero. Every x[j]
    // with j>i is known when x[i] is computed. This is the second
    // half of an LUP-backed solve and also completes GaussianElimination.
    // The threshold scales with the triangular coefficients.
    std::vector<T> BackwardSubstitution(const DynamicMatrix<T>& U, const std::vector<T>& b)
    {
        if (U.Rows() != U.Columns())
        {
            throw std::invalid_argument("BackwardSubstitution failed: matrix must be square.");
        }
        if (U.Rows() != b.size())
        {
            throw std::invalid_argument("BackwardSubstitution failed: matrix rows must match b size.");
        }

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
    /// TEACHING API: RowEchelonForm — Gaussian elimination identifies pivot structure but is not an orthogonal decomposition.
    // Find pivots and eliminate *below* them to produce row-echelon
    // form. This is useful for rank/pivot inspection; it does not
    // itself solve Ax=b or compute an orthogonal QR basis.
    // Zero-column/pivot tolerance is relative to matrix scale.
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
    /// TEACHING API: ReducedRowEchelonForm — Normalize pivots and eliminate above/below to reveal rank/linear constraints.
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
    /// TEACHING API: GaussianElimination — Solve Ax=b via pivoted elimination and back-substitution; use residual norms to judge accuracy.
    // Solve Ax=b by forming [A|b]. At each column choose the
    // available row with largest |coefficient| (partial pivoting),
    // swap it into place, and eliminate below. Extract upper U and
    // transformed RHS, then use BackwardSubstitution. Relative pivot
    // cutoff rejects singular/near-singular input; a small residual
    // alone does not establish good conditioning.
    std::vector<T> GaussianElimination(const DynamicMatrix<T>& A, const std::vector<T>& b)
    {
        if (A.Rows() != A.Columns())
        {
            throw std::invalid_argument("GaussianElimination failed: matrix must be square.");
        }
        if (A.Rows() != b.size())
        {
            throw std::invalid_argument("GaussianElimination failed: matrix rows must match b size.");
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

        // This is partial (row) pivoting: choose the largest *absolute*
        // entry in the current column among remaining rows. Swapping
        // the entire augmented row also permutes the RHS correctly.
        // It reduces growth relative to no pivoting but is not a
        // guarantee of low error on every badly conditioned matrix.
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
    /// TEACHING API: GaussJordanElimination — Eliminate both above and below pivot to solve directly, at additional arithmetic cost.
    // Unlike forward elimination + back substitution, reduce the
    // full augmented matrix to RREF, then read x from its final
    // column if the left side is identity. This performs extra
    // elimination work and should be checked against the pivoted
    // Gaussian solve on ill-conditioned systems.
    std::vector<T> GaussJordanElimination(const DynamicMatrix<T>& A, const std::vector<T>& b)
    {
        if (A.Rows() != A.Columns())
        {
            throw std::invalid_argument("GaussJordanElimination failed: matrix must be square.");
        }
        if (A.Rows() != b.size())
        {
            throw std::invalid_argument("GaussJordanElimination failed: matrix rows must match b size.");
        }

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
    /// TEACHING API: LinearSolve — Preferred high-level solve API for A*x=b; avoid explicitly forming inverse for routine systems.
    // High-level one-RHS entry point calls GaussianElimination.
    // For an already factorized matrix or many RHS columns, prefer
    // reusing the LUP decomposition instead of redoing elimination.
    std::vector<T> LinearSolve(const DynamicMatrix<T>& A, const std::vector<T>& b)
    {
        return GaussianElimination(A, b);
    }

    /// Solve A * X = B for multiple right-hand sides.
    template<FloatingPoint T>
    [[nodiscard]]
    // For A*X=B, compute LUP(A) only once; for each column of B,
    // apply permutation P to that RHS, forward-solve Ly=Pb, then
    // backward-solve Ux=y. Factorization is shared across columns,
    // reducing repeated work relative to independent full solves.
    DynamicMatrix<T> LinearSolve(const DynamicMatrix<T>& A, const DynamicMatrix<T>& B)
    {
        if (A.Rows() != A.Columns())
        {
            throw std::invalid_argument("LinearSolve failed: coefficient matrix must be square.");
        }
        if (A.Rows() != B.Rows())
        {
            throw std::invalid_argument("LinearSolve failed: coefficient rows must match right-hand-side rows.");
        }

        const std::size_t n = A.Rows();
        const std::size_t rhsCount = B.Columns();

        // The one-time decomposition encodes permutation, lower and
        // upper factors. Each right-hand side reuses them instead of
        // repeating an O(n^3) factorization.
        LUPResult<T> lup = LUP(A);
        DynamicMatrix<T> result(n, rhsCount);

        for (std::size_t column = 0; column < rhsCount; ++column)
        {
            std::vector<T> rhs(n);
            for (std::size_t row = 0; row < n; ++row)
            {
                rhs[row] = B(lup.P[row], column);
            }

            std::vector<T> y = ForwardSubstitution(lup.L, rhs);
            std::vector<T> x = BackwardSubstitution(lup.U, y);

            for (std::size_t row = 0; row < n; ++row)
            {
                result(row, column) = x[row];
            }
        }

        return result;
    }

    /// Compute matrix inverse using LUP decomposition.
    template<FloatingPoint T>
    [[nodiscard]]
    /// TEACHING API: Inverse — Compute dynamic matrix inverse through linear solves; numerical stability depends on conditioning.
    // Form inverse columns by solving A*x=e_j for each identity
    // column, reusing one LUP factorization. For a routine Ax=b
    // request, avoid computing all columns of inverse: it costs more
    // and may amplify floating-point error without improving x.
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
