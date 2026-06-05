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
        assert(A.Rows() == A.Columns());
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
        assert(A.Rows() == A.Columns());
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
        assert(A.Rows() == A.Columns());
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

    /// QR Algorithm to find all eigenvalues of a symmetric matrix using Wilkinson shifts and deflation.
    template<FloatingPoint T>
    [[nodiscard]]
    std::vector<T> QREigenValues(
        const DynamicMatrix<T>& A,
        std::size_t maxIterations = 1000,
        T tolerance = std::numeric_limits<T>::epsilon() * T(1e4))
    {
        assert(A.Rows() == A.Columns());
        std::size_t n = A.Rows();

        DynamicMatrix<T> Ak = A;
        std::vector<T> eigenvalues(n, T(0));

        std::size_t activeN = n;
        while (activeN > 1)
        {
            std::size_t iter = 0;
            while (iter < maxIterations)
            {
                T offDiag = std::abs(Ak(activeN - 1, activeN - 2));
                T scale = std::abs(Ak(activeN - 1, activeN - 1)) + std::abs(Ak(activeN - 2, activeN - 2));
                if (scale == T(0)) scale = T(1);

                if (offDiag <= scale * tolerance)
                {
                    Ak(activeN - 1, activeN - 2) = T(0);
                    Ak(activeN - 2, activeN - 1) = T(0);
                    break;
                }

                // Wilkinson shift
                T a = Ak(activeN - 2, activeN - 2);
                T b = Ak(activeN - 1, activeN - 2);
                T c = Ak(activeN - 1, activeN - 1);
                T delta = (a - c) / T(2);
                T shift = c;
                if (delta != T(0) || b != T(0))
                {
                    T denom = std::abs(delta) + std::sqrt(delta * delta + b * b);
                    T sign = (delta >= T(0)) ? T(1) : T(-1);
                    shift = c - (sign * b * b) / denom;
                }

                // Active submatrix
                DynamicMatrix<T> activeAk(activeN, activeN);
                for (std::size_t r = 0; r < activeN; ++r)
                {
                    for (std::size_t col = 0; col < activeN; ++col)
                    {
                        activeAk(r, col) = Ak(r, col);
                        if (r == col) activeAk(r, col) -= shift;
                    }
                }

                QRResult<T> qr = QR(activeAk);

                // Update Ak active part
                DynamicMatrix<T> nextActive = qr.R * qr.Q;
                for (std::size_t r = 0; r < activeN; ++r)
                {
                    for (std::size_t col = 0; col < activeN; ++col)
                    {
                        Ak(r, col) = nextActive(r, col);
                        if (r == col) Ak(r, col) += shift;
                    }
                }

                ++iter;
            }

            eigenvalues[activeN - 1] = Ak(activeN - 1, activeN - 1);
            --activeN;
        }

        eigenvalues[0] = Ak(0, 0);
        return eigenvalues;
    }

    /// QR Algorithm to find all eigenvalues and eigenvectors (for symmetric matrices) using Wilkinson shifts and deflation.
    template<FloatingPoint T>
    [[nodiscard]]
    QREigenResult<T> QREigenVectors(
        const DynamicMatrix<T>& A,
        std::size_t maxIterations = 1000,
        T tolerance = std::numeric_limits<T>::epsilon() * T(1e4))
    {
        assert(A.Rows() == A.Columns());
        std::size_t n = A.Rows();

        DynamicMatrix<T> Ak = A;
        DynamicMatrix<T> V = DynamicMatrix<T>::Identity(n);
        std::vector<T> eigenvalues(n, T(0));

        std::size_t activeN = n;
        while (activeN > 1)
        {
            std::size_t iter = 0;
            while (iter < maxIterations)
            {
                T offDiag = std::abs(Ak(activeN - 1, activeN - 2));
                T scale = std::abs(Ak(activeN - 1, activeN - 1)) + std::abs(Ak(activeN - 2, activeN - 2));
                if (scale == T(0)) scale = T(1);

                if (offDiag <= scale * tolerance)
                {
                    Ak(activeN - 1, activeN - 2) = T(0);
                    Ak(activeN - 2, activeN - 1) = T(0);
                    break;
                }

                // Wilkinson shift
                T a = Ak(activeN - 2, activeN - 2);
                T b = Ak(activeN - 1, activeN - 2);
                T c = Ak(activeN - 1, activeN - 1);
                T delta = (a - c) / T(2);
                T shift = c;
                if (delta != T(0) || b != T(0))
                {
                    T denom = std::abs(delta) + std::sqrt(delta * delta + b * b);
                    T sign = (delta >= T(0)) ? T(1) : T(-1);
                    shift = c - (sign * b * b) / denom;
                }

                // Active submatrix
                DynamicMatrix<T> activeAk(activeN, activeN);
                for (std::size_t r = 0; r < activeN; ++r)
                {
                    for (std::size_t col = 0; col < activeN; ++col)
                    {
                        activeAk(r, col) = Ak(r, col);
                        if (r == col) activeAk(r, col) -= shift;
                    }
                }

                QRResult<T> qr = QR(activeAk);

                // Update Ak active part
                DynamicMatrix<T> nextActive = qr.R * qr.Q;
                for (std::size_t r = 0; r < activeN; ++r)
                {
                    for (std::size_t col = 0; col < activeN; ++col)
                    {
                        Ak(r, col) = nextActive(r, col);
                        if (r == col) Ak(r, col) += shift;
                    }
                }

                // Update V: V[:, 0..activeN-1] = V[:, 0..activeN-1] * qr.Q
                DynamicMatrix<T> V_active(n, activeN, T(0));
                for (std::size_t r = 0; r < n; ++r)
                {
                    for (std::size_t col = 0; col < activeN; ++col)
                    {
                        T sum = T(0);
                        for (std::size_t i = 0; i < activeN; ++i)
                        {
                            sum += V(r, i) * qr.Q(i, col);
                        }
                        V_active(r, col) = sum;
                    }
                }
                for (std::size_t r = 0; r < n; ++r)
                {
                    for (std::size_t col = 0; col < activeN; ++col)
                    {
                        V(r, col) = V_active(r, col);
                    }
                }

                ++iter;
            }

            eigenvalues[activeN - 1] = Ak(activeN - 1, activeN - 1);
            --activeN;
        }

        eigenvalues[0] = Ak(0, 0);
        return { eigenvalues, V };
    }

} // namespace kairo::foundation::math
