module;

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

export module Kairo.Foundation.Math.LinearAlgebra.SVD;

import Kairo.Foundation.Math.Vector;
import Kairo.Foundation.Math.DynamicMatrix;
import Kairo.Foundation.Math.LinearAlgebra.Eigen;

export namespace kairo::foundation::math
{
    /// Result structure for Singular Value Decomposition (thin SVD).
    /// A = U * diag(Sigma) * V^T
    template<FloatingPoint T>
    struct SVDResult
    {
        DynamicMatrix<T> U;     // m x k
        std::vector<T> Sigma;   // k
        DynamicMatrix<T> V;     // n x k
    };

    template<FloatingPoint T>
    struct BidiagonalResult
    {
        DynamicMatrix<T> B; // Upper bidiagonal matrix
        DynamicMatrix<T> U; // m x m orthogonal matrix
        DynamicMatrix<T> V; // n x n orthogonal matrix
    };

    /// Reduce general m x n matrix A (with m >= n) to upper bidiagonal matrix B
    /// using alternating left and right Householder reflections.
    template<FloatingPoint T>
    [[nodiscard]]
    BidiagonalResult<T> GolubKahanBidiagonalization(const DynamicMatrix<T>& A)
    {
        std::size_t m = A.Rows();
        std::size_t n = A.Columns();

        if (m < n)
        {
            throw std::invalid_argument("GolubKahanBidiagonalization failed: row count must be greater than or equal to column count.");
        }

        DynamicMatrix<T> B = A;
        DynamicMatrix<T> U = DynamicMatrix<T>::Identity(m);
        DynamicMatrix<T> V = DynamicMatrix<T>::Identity(n);

        for (std::size_t k = 0; k < n; ++k)
        {
            // --- Left Householder reflection (on columns) ---
            std::size_t lenL = m - k;
            std::vector<T> xL(lenL);
            T normSqL = T(0);
            for (std::size_t i = 0; i < lenL; ++i)
            {
                xL[i] = B(k + i, k);
                normSqL += xL[i] * xL[i];
            }
            T normXL = std::sqrt(normSqL);

            if (normXL > std::numeric_limits<T>::epsilon() * T(10))
            {
                T alphaL = (xL[0] >= T(0)) ? -normXL : normXL;
                std::vector<T> u(lenL);
                u[0] = xL[0] - alphaL;
                for (std::size_t i = 1; i < lenL; ++i)
                {
                    u[i] = xL[i];
                }

                T uNormSq = T(0);
                for (std::size_t i = 0; i < lenL; ++i) uNormSq += u[i] * u[i];
                T uNorm = std::sqrt(uNormSq);
                if (uNorm > std::numeric_limits<T>::epsilon() * T(10))
                {
                    for (std::size_t i = 0; i < lenL; ++i) u[i] /= uNorm;

                    // Apply (I - 2*u*u^T) to B[k..m-1, k..n-1] from the left
                    for (std::size_t col = k; col < n; ++col)
                    {
                        T dot = T(0);
                        for (std::size_t row = 0; row < lenL; ++row)
                        {
                            dot += u[row] * B(k + row, col);
                        }
                        for (std::size_t row = 0; row < lenL; ++row)
                        {
                            B(k + row, col) -= T(2) * u[row] * dot;
                        }
                    }

                    // Accumulate into U
                    for (std::size_t row = 0; row < m; ++row)
                    {
                        T dot = T(0);
                        for (std::size_t col = 0; col < lenL; ++col)
                        {
                            dot += U(row, k + col) * u[col];
                        }
                        for (std::size_t col = 0; col < lenL; ++col)
                        {
                            U(row, k + col) -= T(2) * dot * u[col];
                        }
                    }
                }
            }

            // --- Right Householder reflection (on rows) ---
            if (k < n - 2)
            {
                std::size_t lenR = n - 1 - k;
                std::vector<T> xR(lenR);
                T normSqR = T(0);
                for (std::size_t i = 0; i < lenR; ++i)
                {
                    xR[i] = B(k, k + 1 + i);
                    normSqR += xR[i] * xR[i];
                }
                T normXR = std::sqrt(normSqR);

                if (normXR > std::numeric_limits<T>::epsilon() * T(10))
                {
                    T alphaR = (xR[0] >= T(0)) ? -normXR : normXR;
                    std::vector<T> v(lenR);
                    v[0] = xR[0] - alphaR;
                    for (std::size_t i = 1; i < lenR; ++i)
                    {
                        v[i] = xR[i];
                    }

                    T vNormSq = T(0);
                    for (std::size_t i = 0; i < lenR; ++i) vNormSq += v[i] * v[i];
                    T vNorm = std::sqrt(vNormSq);
                    if (vNorm > std::numeric_limits<T>::epsilon() * T(10))
                    {
                        for (std::size_t i = 0; i < lenR; ++i) v[i] /= vNorm;

                        // Apply (I - 2*v*v^T) to B[k..m-1, k+1..n-1] from the right
                        for (std::size_t row = k; row < m; ++row)
                        {
                            T dot = T(0);
                            for (std::size_t col = 0; col < lenR; ++col)
                            {
                                dot += B(row, k + 1 + col) * v[col];
                            }
                            for (std::size_t col = 0; col < lenR; ++col)
                            {
                                B(row, k + 1 + col) -= T(2) * dot * v[col];
                            }
                        }

                        // Accumulate into V
                        for (std::size_t row = 0; row < n; ++row)
                        {
                            T dot = T(0);
                            for (std::size_t col = 0; col < lenR; ++col)
                            {
                                dot += V(row, k + 1 + col) * v[col];
                            }
                            for (std::size_t col = 0; col < lenR; ++col)
                            {
                                V(row, k + 1 + col) -= T(2) * dot * v[col];
                            }
                        }
                    }
                }
            }
        }

        // Clean up bidiagonal elements (zero out everything else)
        DynamicMatrix<T> cleanB = DynamicMatrix<T>::Zero(m, n);
        for (std::size_t i = 0; i < n; ++i)
        {
            cleanB(i, i) = B(i, i);
            if (i < n - 1)
            {
                cleanB(i, i + 1) = B(i, i + 1);
            }
        }

        return { cleanB, U, V };
    }

    /// Compute Singular Value Decomposition (thin SVD) using Golub-Kahan bidiagonalization
    /// and symmetric tridiagonal QR eigenvalues.
    template<FloatingPoint T>
    [[nodiscard]]
    SVDResult<T> SingularValueDecomposition(const DynamicMatrix<T>& A)
    {
        std::size_t m = A.Rows();
        std::size_t n = A.Columns();

        if (m == 0 || n == 0)
        {
            throw std::invalid_argument("SingularValueDecomposition failed: matrix must be non-empty.");
        }

        std::size_t k = std::min(m, n);

        if (m >= n)
        {
            // 1. Golub-Kahan Bidiagonalization
            BidiagonalResult<T> bid = GolubKahanBidiagonalization(A);

            // 2. Compute T = B^T * B
            DynamicMatrix<T> BT = bid.B.Transpose();
            DynamicMatrix<T> T_mat = BT * bid.B;

            // 3. Solve eigenvalues and eigenvectors of T using tridiagonal QR
            QREigenResult<T> eigenResult = QREigenVectors(T_mat);

            // 4. Sort eigenvalues and eigenvectors in descending order
            std::vector<std::size_t> indices(n);
            std::iota(indices.begin(), indices.end(), std::size_t(0));
            std::sort(indices.begin(), indices.end(), [&](std::size_t a, std::size_t b) {
                return eigenResult.eigenvalues[a] > eigenResult.eigenvalues[b];
            });

            std::vector<T> sortedSigma(k, T(0));
            DynamicMatrix<T> V_tilde(n, k);
            for (std::size_t j = 0; j < k; ++j)
            {
                std::size_t origIndex = indices[j];
                T eigVal = eigenResult.eigenvalues[origIndex];
                sortedSigma[j] = std::sqrt(std::max(T(0), eigVal));

                for (std::size_t i = 0; i < n; ++i)
                {
                    V_tilde(i, j) = eigenResult.eigenvectors(i, origIndex);
                }
            }

            // 5. Accumulate right singular vectors V = V_bid * V_tilde
            DynamicMatrix<T> V = bid.V * V_tilde;

            // 6. Compute U = A * V * inv(Sigma) using Gram-Schmidt basis completion for small singular values
            DynamicMatrix<T> U(m, k);
            for (std::size_t j = 0; j < k; ++j)
            {
                T s = sortedSigma[j];
                if (s > std::numeric_limits<T>::epsilon() * T(1e2))
                {
                    for (std::size_t i = 0; i < m; ++i)
                    {
                        T val = T(0);
                        for (std::size_t c = 0; c < n; ++c)
                        {
                            val += A(i, c) * V(c, j);
                        }
                        U(i, j) = val / s;
                    }
                }
                else
                {
                    // Singular value is near-zero; generate a basis vector orthogonal to all prior U columns
                    std::vector<T> u_col(m, T(0));
                    bool found = false;
                    for (std::size_t basis_idx = 0; basis_idx < m; ++basis_idx)
                    {
                        std::fill(u_col.begin(), u_col.end(), T(0));
                        u_col[basis_idx] = T(1);

                        // Gram-Schmidt projection
                        for (std::size_t prev = 0; prev < j; ++prev)
                        {
                            T dot = T(0);
                            for (std::size_t i = 0; i < m; ++i)
                            {
                                dot += u_col[i] * U(i, prev);
                            }
                            for (std::size_t i = 0; i < m; ++i)
                            {
                                u_col[i] -= dot * U(i, prev);
                            }
                        }

                        // Normalize and verify linear independence
                        T normSq = T(0);
                        for (T val : u_col) normSq += val * val;
                        T norm = std::sqrt(normSq);
                        if (norm > std::numeric_limits<T>::epsilon() * T(10))
                        {
                            for (std::size_t i = 0; i < m; ++i)
                            {
                                U(i, j) = u_col[i] / norm;
                            }
                            found = true;
                            break;
                        }
                    }

                    if (!found)
                    {
                        for (std::size_t i = 0; i < m; ++i)
                        {
                            U(i, j) = T(0);
                        }
                    }
                }
            }

            return { U, sortedSigma, V };
        }
        else
        {
            // m < n: Transpose and solve SVD for A^T
            DynamicMatrix<T> AT(n, m);
            for (std::size_t r = 0; r < n; ++r)
            {
                for (std::size_t c = 0; c < m; ++c)
                {
                    AT(r, c) = A(c, r);
                }
            }

            SVDResult<T> subResult = SingularValueDecomposition(AT);

            return { subResult.V, subResult.Sigma, subResult.U };
        }
    }

    /// Compute Moore-Penrose Pseudo-Inverse A^+ using SVD.
    template<FloatingPoint T>
    [[nodiscard]]
    DynamicMatrix<T> PseudoInverse(const DynamicMatrix<T>& A)
    {
        std::size_t m = A.Rows();
        std::size_t n = A.Columns();

        SVDResult<T> svd = SingularValueDecomposition(A);

        // Find max singular value for thresholding
        T maxSigma = T(0);
        for (T val : svd.Sigma)
        {
            maxSigma = std::max(maxSigma, val);
        }

        T threshold = std::max(m, n) * maxSigma * std::numeric_limits<T>::epsilon() * T(1e3);

        std::size_t k = svd.Sigma.size();
        std::vector<T> invSigma(k, T(0));
        for (std::size_t i = 0; i < k; ++i)
        {
            if (svd.Sigma[i] > threshold)
            {
                invSigma[i] = T(1) / svd.Sigma[i];
            }
        }

        // A^+ = V * Sigma^+ * U^T
        // Let's compute V * Sigma^+ of size n x k
        DynamicMatrix<T> VS(n, k);
        for (std::size_t r = 0; r < n; ++r)
        {
            for (std::size_t c = 0; c < k; ++c)
            {
                VS(r, c) = svd.V(r, c) * invSigma[c];
            }
        }

        // Compute A^+ = VS * U^T of size n x m
        DynamicMatrix<T> result(n, m, T(0));
        for (std::size_t r = 0; r < n; ++r)
        {
            for (std::size_t c = 0; c < m; ++c)
            {
                T sum = T(0);
                for (std::size_t i = 0; i < k; ++i)
                {
                    sum += VS(r, i) * svd.U(c, i); // Transposed U index
                }
                result(r, c) = sum;
            }
        }

        return result;
    }

    /// Compute Low Rank Approximation of matrix A with rank k.
    template<FloatingPoint T>
    [[nodiscard]]
    DynamicMatrix<T> LowRankApproximation(const DynamicMatrix<T>& A, std::size_t rank)
    {
        std::size_t m = A.Rows();
        std::size_t n = A.Columns();

        SVDResult<T> svd = SingularValueDecomposition(A);

        std::size_t k = std::min(rank, svd.Sigma.size());

        // A_k = U_k * Sigma_k * V_k^T
        DynamicMatrix<T> result(m, n, T(0));
        for (std::size_t r = 0; r < m; ++r)
        {
            for (std::size_t c = 0; c < n; ++c)
            {
                T sum = T(0);
                for (std::size_t i = 0; i < k; ++i)
                {
                    sum += svd.U(r, i) * svd.Sigma[i] * svd.V(c, i);
                }
                result(r, c) = sum;
            }
        }

        return result;
    }

    /// Compute Rank of a general matrix using SVD.
    template<FloatingPoint T>
    [[nodiscard]]
    std::size_t Rank(const DynamicMatrix<T>& A)
    {
        if (A.Empty()) return 0;

        SVDResult<T> svd = SingularValueDecomposition(A);
        T maxSigma = T(0);
        for (T val : svd.Sigma)
        {
            maxSigma = std::max(maxSigma, val);
        }

        T threshold = std::max(A.Rows(), A.Columns()) * maxSigma * std::sqrt(std::numeric_limits<T>::epsilon()) * T(10);
        if (threshold == T(0))
        {
            return 0;
        }

        std::size_t r = 0;
        for (T val : svd.Sigma)
        {
            if (val > threshold)
            {
                ++r;
            }
        }
        return r;
    }

    /// Compute 2-norm Condition Number of a matrix using SVD.
    template<FloatingPoint T>
    [[nodiscard]]
    T ConditionNumber(const DynamicMatrix<T>& A)
    {
        if (A.Empty()) return T(1);

        SVDResult<T> svd = SingularValueDecomposition(A);
        T maxSigma = T(0);
        T minSigma = std::numeric_limits<T>::max();
        for (T val : svd.Sigma)
        {
            maxSigma = std::max(maxSigma, val);
            minSigma = std::min(minSigma, val);
        }

        T threshold = std::max(A.Rows(), A.Columns()) * maxSigma * std::sqrt(std::numeric_limits<T>::epsilon()) * T(10);
        if (minSigma <= threshold)
        {
            return std::numeric_limits<T>::infinity();
        }
        return maxSigma / minSigma;
    }

} // namespace kairo::foundation::math
