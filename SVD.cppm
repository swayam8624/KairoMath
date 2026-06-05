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

    /// Compute Singular Value Decomposition (thin SVD) for general rectangular matrices.
    template<FloatingPoint T>
    [[nodiscard]]
    SVDResult<T> SingularValueDecomposition(const DynamicMatrix<T>& A)
    {
        std::size_t m = A.Rows();
        std::size_t n = A.Columns();
        assert(m > 0 && n > 0);

        std::size_t k = std::min(m, n);

        if (m >= n)
        {
            // Compute A^T * A of size n x n
            DynamicMatrix<T> ATA(n, n, T(0));
            for (std::size_t r = 0; r < n; ++r)
            {
                for (std::size_t c = 0; c < n; ++c)
                {
                    T sum = T(0);
                    for (std::size_t i = 0; i < m; ++i)
                    {
                        sum += A(i, r) * A(i, c);
                    }
                    ATA(r, c) = sum;
                }
            }

            // Eigenvalues and eigenvectors of A^T * A
            QREigenResult<T> eigenResult = QREigenVectors(ATA);

            // Sort eigenvalues and corresponding eigenvectors in descending order
            std::vector<std::size_t> indices(n);
            std::iota(indices.begin(), indices.end(), std::size_t(0));
            std::sort(indices.begin(), indices.end(), [&](std::size_t a, std::size_t b) {
                return eigenResult.eigenvalues[a] > eigenResult.eigenvalues[b];
            });

            std::vector<T> sortedSigma(k, T(0));
            DynamicMatrix<T> V(n, k);
            for (std::size_t j = 0; j < k; ++j)
            {
                std::size_t origIndex = indices[j];
                T eigVal = eigenResult.eigenvalues[origIndex];
                sortedSigma[j] = std::sqrt(std::max(T(0), eigVal));

                for (std::size_t i = 0; i < n; ++i)
                {
                    V(i, j) = eigenResult.eigenvectors(i, origIndex);
                }
            }

            // Compute U = A * V * inv(Sigma)
            // To ensure U columns are orthonormal even for zero/small singular values,
            // we use Gram-Schmidt orthogonalization.
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

            // A^T = U_sub * Sigma_sub * V_sub^T => A = V_sub * Sigma_sub * U_sub^T
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

} // namespace kairo::foundation::math
