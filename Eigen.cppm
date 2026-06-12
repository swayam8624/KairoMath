module;

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

export module Kairo.Foundation.Math.LinearAlgebra.Eigen;

import Kairo.Foundation.Math.Vector;
import Kairo.Foundation.Math.DynamicMatrix;
import Kairo.Foundation.Math.LinearAlgebra.LinearSolve;
import Kairo.Foundation.Math.LinearAlgebra.Decomposition;

export namespace kairo::foundation::math
{
    /// Result structure for power iteration.
    template<FloatingPoint T>
    struct PowerIterationResult
    {
        T eigenvalue;
        std::vector<T> eigenvector;
    };

    /// Result structure for QR eigenvalues and eigenvectors.
    template<FloatingPoint T>
    struct QREigenResult
    {
        std::vector<T> eigenvalues;
        DynamicMatrix<T> eigenvectors;
    };

    /// Power Iteration to find the dominant eigenvalue and its corresponding eigenvector.
    template<FloatingPoint T>
    [[nodiscard]]
    PowerIterationResult<T> PowerIteration(
        const DynamicMatrix<T>& A,
        std::size_t maxIterations = 1000,
        T tolerance = std::numeric_limits<T>::epsilon() * T(1e4))
    {
        if (A.Rows() != A.Columns())
        {
            throw std::invalid_argument("PowerIteration failed: matrix must be square.");
        }
        std::size_t n = A.Rows();

        // Initial guess: non-uniform values to avoid symmetry issues
        std::vector<T> b(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            b[i] = T(i + 1);
        }
        T bNorm = T(0);
        for (T val : b) bNorm += val * val;
        bNorm = std::sqrt(bNorm);
        for (T& val : b) val /= bNorm;

        T eigenvalue = T(0);
        for (std::size_t iter = 0; iter < maxIterations; ++iter)
        {
            // Compute w = A * b
            std::vector<T> w(n, T(0));
            for (std::size_t r = 0; r < n; ++r)
            {
                for (std::size_t c = 0; c < n; ++c)
                {
                    w[r] += A(r, c) * b[c];
                }
            }

            // Norm of w
            T wNormSq = T(0);
            for (T val : w) wNormSq += val * val;
            T wNorm = std::sqrt(wNormSq);
            if (wNorm <= std::numeric_limits<T>::epsilon() * T(10))
            {
                break;
            }

            // Next b
            std::vector<T> bNext(n);
            for (std::size_t i = 0; i < n; ++i)
            {
                bNext[i] = w[i] / wNorm;
            }

            // Rayleigh quotient eigenvalue = b^T * A * b
            T nextEigenvalue = T(0);
            for (std::size_t i = 0; i < n; ++i)
            {
                nextEigenvalue += b[i] * w[i]; // w is A * b, and b is normalized
            }

            // Check convergence of eigenvalue and eigenvector
            T diff = T(0);
            for (std::size_t i = 0; i < n; ++i)
            {
                T d = std::abs(bNext[i]) - std::abs(b[i]);
                diff += d * d;
            }
            diff = std::sqrt(diff);

            T previousEigenvalue = eigenvalue;
            b = std::move(bNext);
            eigenvalue = nextEigenvalue;

            if (diff < tolerance && std::abs(eigenvalue - previousEigenvalue) < tolerance)
            {
                break;
            }
        }

        return { eigenvalue, b };
    }

    /// Inverse Power Iteration to find the eigenvalue closest to zero.
    template<FloatingPoint T>
    [[nodiscard]]
    PowerIterationResult<T> InversePowerIteration(
        const DynamicMatrix<T>& A,
        std::size_t maxIterations = 1000,
        T tolerance = std::numeric_limits<T>::epsilon() * T(1e4))
    {
        if (A.Rows() != A.Columns())
        {
            throw std::invalid_argument("InversePowerIteration failed: matrix must be square.");
        }
        std::size_t n = A.Rows();

        // Initial guess: non-uniform values to avoid symmetry issues
        std::vector<T> b(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            b[i] = T(i + 1);
        }
        T bNorm = T(0);
        for (T val : b) bNorm += val * val;
        bNorm = std::sqrt(bNorm);
        for (T& val : b) val /= bNorm;

        T eigenvalue = T(0);
        for (std::size_t iter = 0; iter < maxIterations; ++iter)
        {
            // Solve A * w = b using LinearSolve
            std::vector<T> w;
            try
            {
                w = LinearSolve(A, b);
            }
            catch (const std::exception&)
            {
                // Matrix is singular, so 0 is the exact eigenvalue
                return { T(0), b };
            }

            T wNormSq = T(0);
            for (T val : w) wNormSq += val * val;
            T wNorm = std::sqrt(wNormSq);
            if (wNorm <= std::numeric_limits<T>::epsilon() * T(10))
            {
                break;
            }

            std::vector<T> bNext(n);
            for (std::size_t i = 0; i < n; ++i)
            {
                bNext[i] = w[i] / wNorm;
            }

            // Compute actual eigenvalue corresponding to A * v = lambda * v
            // Compute A * bNext
            std::vector<T> AbNext(n, T(0));
            for (std::size_t r = 0; r < n; ++r)
            {
                for (std::size_t c = 0; c < n; ++c)
                {
                    AbNext[r] += A(r, c) * bNext[c];
                }
            }
            T num = T(0);
            for (std::size_t i = 0; i < n; ++i)
            {
                num += bNext[i] * AbNext[i];
            }
            T nextEigenvalue = num;

            T diff = T(0);
            for (std::size_t i = 0; i < n; ++i)
            {
                T d = std::abs(bNext[i]) - std::abs(b[i]);
                diff += d * d;
            }
            diff = std::sqrt(diff);

            b = std::move(bNext);
            eigenvalue = nextEigenvalue;

            if (diff < tolerance)
            {
                break;
            }
        }

        return { eigenvalue, b };
    }

    /// Shifted Inverse Power Iteration to find the eigenvalue closest to a target shift value.
    template<FloatingPoint T>
    [[nodiscard]]
    PowerIterationResult<T> ShiftedInversePowerIteration(
        const DynamicMatrix<T>& A,
        T shift,
        std::size_t maxIterations = 1000,
        T tolerance = std::numeric_limits<T>::epsilon() * T(1e4))
    {
        if (A.Rows() != A.Columns())
        {
            throw std::invalid_argument("ShiftedInversePowerIteration failed: matrix must be square.");
        }
        std::size_t n = A.Rows();

        // Shifted matrix: A - shift * I
        DynamicMatrix<T> shiftedA = A;
        for (std::size_t i = 0; i < n; ++i)
        {
            shiftedA(i, i) -= shift;
        }

        // Run Inverse Power Iteration on A - shift * I
        PowerIterationResult<T> subResult = InversePowerIteration(shiftedA, maxIterations, tolerance);

        // Eigenvalue of A is eigenvalue of shiftedA + shift
        return { subResult.eigenvalue + shift, subResult.eigenvector };
    }

    template<FloatingPoint T>
    struct TridiagonalResult
    {
        DynamicMatrix<T> T_mat;
        DynamicMatrix<T> Q;
    };

    /// Reduce a symmetric matrix to symmetric tridiagonal form using Householder reflections.
    template<FloatingPoint T>
    [[nodiscard]]
    TridiagonalResult<T> HouseholderTridiagonalization(const DynamicMatrix<T>& A)
    {
        if (A.Rows() != A.Columns())
        {
            throw std::invalid_argument("Tridiagonalization failed: Matrix must be square.");
        }
        std::size_t n = A.Rows();
        DynamicMatrix<T> Q = DynamicMatrix<T>::Identity(n);
        DynamicMatrix<T> T_mat = A;

        for (std::size_t k = 0; k < n - 2; ++k)
        {
            // Vector x is T_mat[k+1..n-1, k]
            std::size_t len = n - 1 - k;
            std::vector<T> x(len);
            T normSq = T(0);
            for (std::size_t i = 0; i < len; ++i)
            {
                x[i] = T_mat(k + 1 + i, k);
                normSq += x[i] * x[i];
            }
            T normX = std::sqrt(normSq);
            if (normX <= std::numeric_limits<T>::epsilon() * T(10))
            {
                continue;
            }

            T alpha = (x[0] >= T(0)) ? -normX : normX;
            std::vector<T> v(len);
            v[0] = x[0] - alpha;
            for (std::size_t i = 1; i < len; ++i)
            {
                v[i] = x[i];
            }

            T vNormSq = T(0);
            for (std::size_t i = 0; i < len; ++i) vNormSq += v[i] * v[i];
            T vNorm = std::sqrt(vNormSq);
            if (vNorm > std::numeric_limits<T>::epsilon() * T(10))
            {
                for (std::size_t i = 0; i < len; ++i) v[i] /= vNorm;
            }
            else
            {
                continue;
            }

            // Apply Householder reflection from both sides to the submatrix T_mat[k+1..n-1, k+1..n-1]
            // p = 2 * T_mat[k+1..n-1, k+1..n-1] * v
            std::vector<T> p(len, T(0));
            for (std::size_t i = 0; i < len; ++i)
            {
                T sum = T(0);
                for (std::size_t j = 0; j < len; ++j)
                {
                    sum += T_mat(k + 1 + i, k + 1 + j) * v[j];
                }
                p[i] = T(2) * sum;
            }

            // w = p - (p^T * v) * v
            T pv = T(0);
            for (std::size_t i = 0; i < len; ++i) pv += p[i] * v[i];
            std::vector<T> w(len);
            for (std::size_t i = 0; i < len; ++i)
            {
                w[i] = p[i] - pv * v[i];
            }

            // Update submatrix T_mat = T_mat - v * w^T - w * v^T
            for (std::size_t i = 0; i < len; ++i)
            {
                for (std::size_t j = 0; j < len; ++j)
                {
                    T_mat(k + 1 + i, k + 1 + j) -= v[i] * w[j] + w[i] * v[j];
                }
            }

            T_mat(k + 1, k) = alpha;
            T_mat(k, k + 1) = alpha;
            for (std::size_t i = k + 2; i < n; ++i)
            {
                T_mat(i, k) = T(0);
                T_mat(k, i) = T(0);
            }

            // Update Q: Q = Q * H_k => Q[:, k+1..n-1] = Q[:, k+1..n-1] * (I - 2*v*v^T)
            for (std::size_t row = 0; row < n; ++row)
            {
                T dot = T(0);
                for (std::size_t col = 0; col < len; ++col)
                {
                    dot += Q(row, k + 1 + col) * v[col];
                }
                for (std::size_t col = 0; col < len; ++col)
                {
                    Q(row, k + 1 + col) -= T(2) * dot * v[col];
                }
            }
        }

        return { T_mat, Q };
    }

    /// Run one Implicit QR step on symmetric tridiagonal matrix T_mat and eigenvectors V.
    template<typename T>
    void ImplicitQRStep(DynamicMatrix<T>& T_mat, DynamicMatrix<T>& V, std::size_t l, std::size_t m)
    {
        T d1 = T_mat(m - 1, m - 1);
        T d2 = T_mat(m, m);
        T e1 = T_mat(m - 1, m);
        T delta = (d1 - d2) / T(2);
        T shift = d2;
        if (delta != T(0) || e1 != T(0))
        {
            T denom = std::abs(delta) + std::sqrt(delta * delta + e1 * e1);
            T sign = (delta >= T(0)) ? T(1) : T(-1);
            shift = d2 - (sign * e1 * e1) / denom;
        }

        T x = T_mat(l, l) - shift;
        T z = T_mat(l + 1, l);

        for (std::size_t k = l; k < m; ++k)
        {
            T c = T(1), s = T(0);
            if (std::abs(z) > T(0))
            {
                T r = std::hypot(x, z);
                c = x / r;
                s = -z / r;
            }

            // Apply Givens rotation to T_mat (left and right)
            for (std::size_t j = 0; j < T_mat.Columns(); ++j)
            {
                T r1 = T_mat(k, j);
                T r2 = T_mat(k + 1, j);
                T_mat(k, j) = c * r1 - s * r2;
                T_mat(k + 1, j) = s * r1 + c * r2;
            }
            for (std::size_t i = 0; i < T_mat.Rows(); ++i)
            {
                T c1 = T_mat(i, k);
                T c2 = T_mat(i, k + 1);
                T_mat(i, k) = c * c1 - s * c2;
                T_mat(i, k + 1) = s * c1 + c * c2;
            }

            // Accumulate into V
            for (std::size_t i = 0; i < V.Rows(); ++i)
            {
                T c1 = V(i, k);
                T c2 = V(i, k + 1);
                V(i, k) = c * c1 - s * c2;
                V(i, k + 1) = s * c1 + c * c2;
            }

            if (k < m - 1)
            {
                x = T_mat(k + 1, k);
                z = T_mat(k + 2, k);
            }
        }
    }

    /// Find all eigenvalues and eigenvectors of a symmetric matrix using Tridiagonalization and Implicit QR.
    template<FloatingPoint T>
    [[nodiscard]]
    QREigenResult<T> QREigenVectors(
        const DynamicMatrix<T>& A,
        std::size_t maxIterations = 1000,
        T tolerance = std::numeric_limits<T>::epsilon() * T(1e4))
    {
        if (A.Rows() != A.Columns())
        {
            throw std::invalid_argument("QREigenVectors failed: matrix must be square.");
        }
        std::size_t n = A.Rows();
        if (n == 0) return { {}, {} };
        if (n == 1) return { {A(0, 0)}, DynamicMatrix<T>::Identity(1) };

        // 1. Tridiagonalization
        TridiagonalResult<T> tri = HouseholderTridiagonalization(A);
        DynamicMatrix<T> T_mat = tri.T_mat;
        DynamicMatrix<T> V = tri.Q;

        // 2. Implicit symmetric QR on tridiagonal matrix
        std::size_t activeN = n;
        while (activeN > 1)
        {
            std::size_t iter = 0;
            while (iter < maxIterations)
            {
                T offDiag = std::abs(T_mat(activeN - 1, activeN - 2));
                T scale = std::abs(T_mat(activeN - 1, activeN - 1)) + std::abs(T_mat(activeN - 2, activeN - 2));
                if (scale == T(0)) scale = T(1);

                if (offDiag <= scale * tolerance)
                {
                    T_mat(activeN - 1, activeN - 2) = T(0);
                    T_mat(activeN - 2, activeN - 1) = T(0);
                    break;
                }

                // Find active submatrix [l..activeN-1]
                std::size_t l = activeN - 1;
                for (std::size_t i = activeN - 1; i > 0; --i)
                {
                    T sub = std::abs(T_mat(i, i - 1));
                    T diagScale = std::abs(T_mat(i, i)) + std::abs(T_mat(i - 1, i - 1));
                    if (diagScale == T(0)) diagScale = T(1);

                    if (sub <= diagScale * tolerance)
                    {
                        T_mat(i, i - 1) = T(0);
                        T_mat(i - 1, i) = T(0);
                        l = i;
                        break;
                    }
                    if (i == 1)
                    {
                        l = 0;
                    }
                }

                if (l >= activeN - 1)
                {
                    break;
                }

                ImplicitQRStep(T_mat, V, l, activeN - 1);
                ++iter;
            }

            --activeN;
        }

        std::vector<T> eigenvalues(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            eigenvalues[i] = T_mat(i, i);
        }

        return { eigenvalues, V };
    }

    /// Find eigenvalues of a symmetric matrix.
    template<FloatingPoint T>
    [[nodiscard]]
    std::vector<T> QREigenValues(
        const DynamicMatrix<T>& A,
        std::size_t maxIterations = 1000,
        T tolerance = std::numeric_limits<T>::epsilon() * T(1e4))
    {
        return QREigenVectors(A, maxIterations, tolerance).eigenvalues;
    }

} // namespace kairo::foundation::math
