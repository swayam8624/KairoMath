module;

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

export module Kairo.Foundation.Math.LinearAlgebra.Statistics;

import Kairo.Foundation.Math.Vector;
import Kairo.Foundation.Math.DynamicMatrix;
import Kairo.Foundation.Math.LinearAlgebra.Eigen;
import Kairo.Foundation.Math.LinearAlgebra.SVD;

export namespace kairo::foundation::math
{
    /// Result structure for Principal Component Analysis.
    template<FloatingPoint T>
    struct PCAResult
    {
        std::vector<T> explainedVariance; // Eigenvalues in descending order
        DynamicMatrix<T> components;     // Eigenvectors (columns) corresponding to eigenvalues
    };

    /// Compute the Covariance Matrix of a dataset X (N samples x M features).
    /// Assumes samples are rows, features are columns.
    template<FloatingPoint T>
    [[nodiscard]]
    DynamicMatrix<T> CovarianceMatrix(const DynamicMatrix<T>& X)
    {
        std::size_t n = X.Rows();
        std::size_t m = X.Columns();
        if (n <= 1)
        {
            throw std::runtime_error("CovarianceMatrix requires at least 2 samples.");
        }

        // Compute mean of each feature
        std::vector<T> means(m, T(0));
        for (std::size_t j = 0; j < m; ++j)
        {
            T sum = T(0);
            for (std::size_t i = 0; i < n; ++i)
            {
                sum += X(i, j);
            }
            means[j] = sum / static_cast<T>(n);
        }

        // Compute covariance matrix
        DynamicMatrix<T> cov(m, m, T(0));
        T divisor = static_cast<T>(n - 1);
        for (std::size_t j = 0; j < m; ++j)
        {
            for (std::size_t k = j; k < m; ++k)
            {
                T sum = T(0);
                for (std::size_t i = 0; i < n; ++i)
                {
                    sum += (X(i, j) - means[j]) * (X(i, k) - means[k]);
                }
                T val = sum / divisor;
                cov(j, k) = val;
                cov(k, j) = val; // Symmetric
            }
        }

        return cov;
    }

    /// Compute the Correlation Matrix of a dataset X (N samples x M features).
    template<FloatingPoint T>
    [[nodiscard]]
    DynamicMatrix<T> CorrelationMatrix(const DynamicMatrix<T>& X)
    {
        DynamicMatrix<T> cov = CovarianceMatrix(X);
        std::size_t m = cov.Rows();

        DynamicMatrix<T> corr(m, m, T(0));
        for (std::size_t j = 0; j < m; ++j)
        {
            for (std::size_t k = j; k < m; ++k)
            {
                T varJ = cov(j, j);
                T varK = cov(k, k);
                if (varJ <= T(0) || varK <= T(0))
                {
                    corr(j, k) = T(0);
                    corr(k, j) = T(0);
                }
                else
                {
                    T val = cov(j, k) / std::sqrt(varJ * varK);
                    corr(j, k) = val;
                    corr(k, j) = val;
                }
            }
        }

        return corr;
    }

    /// Run Principal Component Analysis (PCA) on a dataset X (N samples x M features).
    template<FloatingPoint T>
    [[nodiscard]]
    PCAResult<T> PCA(const DynamicMatrix<T>& X)
    {
        // 1. Covariance matrix
        DynamicMatrix<T> cov = CovarianceMatrix(X);

        // 2. Eigenvalues and eigenvectors
        QREigenResult<T> eigenResult = QREigenVectors(cov);

        // 3. Sort descending
        std::size_t m = cov.Rows();
        std::vector<std::size_t> indices(m);
        for (std::size_t i = 0; i < m; ++i) indices[i] = i;
        std::sort(indices.begin(), indices.end(), [&](std::size_t a, std::size_t b) {
            return eigenResult.eigenvalues[a] > eigenResult.eigenvalues[b];
        });

        std::vector<T> sortedVariance(m);
        DynamicMatrix<T> sortedComponents(m, m);

        for (std::size_t j = 0; j < m; ++j)
        {
            std::size_t origIndex = indices[j];
            sortedVariance[j] = std::max(T(0), eigenResult.eigenvalues[origIndex]);
            for (std::size_t i = 0; i < m; ++i)
            {
                sortedComponents(i, j) = eigenResult.eigenvectors(i, origIndex);
            }
        }

        return { sortedVariance, sortedComponents };
    }

    /// Perform Linear Regression using Moore-Penrose Pseudo-Inverse.
    /// Fits model X * beta = y, returning beta vector.
    template<FloatingPoint T>
    [[nodiscard]]
    std::vector<T> LinearRegression(const DynamicMatrix<T>& X, const std::vector<T>& y)
    {
        std::size_t n = X.Rows();
        std::size_t m = X.Columns();
        assert(n == y.size());

        // A^+ = PseudoInverse(X) of size m x n
        DynamicMatrix<T> invX = PseudoInverse(X);

        // beta = invX * y
        std::vector<T> beta(m, T(0));
        for (std::size_t r = 0; r < m; ++r)
        {
            T sum = T(0);
            for (std::size_t c = 0; c < n; ++c)
            {
                sum += invX(r, c) * y[c];
            }
            beta[r] = sum;
        }

        return beta;
    }

} // namespace kairo::foundation::math
