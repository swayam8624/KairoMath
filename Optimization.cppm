module;

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

export module Kairo.Foundation.Math.Optimization;

import Kairo.Foundation.Math.Vector;
import Kairo.Foundation.Math.DynamicMatrix;
import Kairo.Foundation.Math.LinearAlgebra.LinearSolve;

export namespace kairo::foundation::math
{
    //=========================================================
    // Optimization Settings / Result
    //=========================================================

    template<FloatingPoint T>
    struct OptimizationSettings final
    {
        std::size_t MaxIterations = 1000;

        T LearningRate = T(1e-2);

        T GradientTolerance = T(1e-6);

        T StepTolerance = T(1e-9);

        T ValueTolerance = T(1e-12);

        T ArmijoC1 = T(1e-4);

        T BacktrackingShrink = T(0.5);

        T MinimumStepScale = T(1e-12);

        bool UseLineSearch = true;
    };

    template<FloatingPoint T>
    struct OptimizationResult final
    {
        DynamicMatrix<T> X;

        T Value = T(0);

        T GradientNorm = T(0);

        T StepNorm = T(0);

        std::size_t Iterations = 0;

        bool Converged = false;

        std::string Message;
    };

    template<FloatingPoint T>
    struct ConjugateGradientResult final
    {
        DynamicMatrix<T> X;

        T ResidualNorm = T(0);

        std::size_t Iterations = 0;

        bool Converged = false;

        std::string Message;
    };

    //=========================================================
    // Internal Helpers
    //=========================================================

    namespace optimization_detail
    {
        template<FloatingPoint T>
        void RequireColumnVector(
            const DynamicMatrix<T>& vector,
            const char* name)
        {
            if (vector.Columns() != 1)
            {
                throw std::invalid_argument(
                    std::string(name) + " must be a column vector.");
            }
        }

        template<FloatingPoint T>
        void RequireSameShape(
            const DynamicMatrix<T>& lhs,
            const DynamicMatrix<T>& rhs,
            const char* operation)
        {
            if (lhs.Rows() != rhs.Rows() ||
                lhs.Columns() != rhs.Columns())
            {
                throw std::invalid_argument(
                    std::string(operation) + " failed: matrix shape mismatch.");
            }
        }

        template<FloatingPoint T>
        [[nodiscard]]
        DynamicMatrix<T> ZeroLike(
            const DynamicMatrix<T>& matrix)
        {
            return DynamicMatrix<T>(
                matrix.Rows(),
                matrix.Columns(),
                T(0));
        }

        template<FloatingPoint T>
        [[nodiscard]]
        T DotColumn(
            const DynamicMatrix<T>& lhs,
            const DynamicMatrix<T>& rhs)
        {
            RequireColumnVector(lhs, "lhs");
            RequireColumnVector(rhs, "rhs");
            RequireSameShape(lhs, rhs, "DotColumn");

            T sum = T(0);

            for (std::size_t i = 0; i < lhs.Rows(); ++i)
            {
                sum += lhs(i, 0) * rhs(i, 0);
            }

            return sum;
        }

        template<FloatingPoint T>
        [[nodiscard]]
        T NormSquared(
            const DynamicMatrix<T>& vector)
        {
            return DotColumn(
                vector,
                vector);
        }

        template<FloatingPoint T>
        [[nodiscard]]
        T Norm(
            const DynamicMatrix<T>& vector)
        {
            return std::sqrt(
                NormSquared(vector));
        }

        template<FloatingPoint T>
        [[nodiscard]]
        DynamicMatrix<T> Negative(
            DynamicMatrix<T> vector)
        {
            for (std::size_t i = 0; i < vector.Size(); ++i)
            {
                vector[i] = -vector[i];
            }

            return vector;
        }

        template<FloatingPoint T>
        [[nodiscard]]
        DynamicMatrix<T> AddScaled(
            DynamicMatrix<T> lhs,
            const DynamicMatrix<T>& direction,
            T scale)
        {
            RequireSameShape(
                lhs,
                direction,
                "AddScaled");

            for (std::size_t i = 0; i < lhs.Size(); ++i)
            {
                lhs[i] += direction[i] * scale;
            }

            return lhs;
        }

        template<FloatingPoint T>
        [[nodiscard]]
        bool IsFinite(
            const DynamicMatrix<T>& matrix)
        {
            for (std::size_t i = 0; i < matrix.Size(); ++i)
            {
                if (!std::isfinite(matrix[i]))
                {
                    return false;
                }
            }

            return true;
        }

        template<FloatingPoint T>
        [[nodiscard]]
        DynamicMatrix<T> HadamardSquare(
            const DynamicMatrix<T>& matrix)
        {
            DynamicMatrix<T> result(
                matrix.Rows(),
                matrix.Columns());

            for (std::size_t i = 0; i < matrix.Size(); ++i)
            {
                result[i] = matrix[i] * matrix[i];
            }

            return result;
        }

        template<FloatingPoint T>
        [[nodiscard]]
        DynamicMatrix<T> ElementwiseSqrt(
            const DynamicMatrix<T>& matrix)
        {
            DynamicMatrix<T> result(
                matrix.Rows(),
                matrix.Columns());

            for (std::size_t i = 0; i < matrix.Size(); ++i)
            {
                result[i] = std::sqrt(matrix[i]);
            }

            return result;
        }

        template<FloatingPoint T>
        [[nodiscard]]
        DynamicMatrix<T> ElementwiseDivide(
            const DynamicMatrix<T>& numerator,
            const DynamicMatrix<T>& denominator,
            T epsilon)
        {
            RequireSameShape(
                numerator,
                denominator,
                "ElementwiseDivide");

            DynamicMatrix<T> result(
                numerator.Rows(),
                numerator.Columns());

            for (std::size_t i = 0; i < numerator.Size(); ++i)
            {
                result[i] =
                    numerator[i] /
                    (denominator[i] + epsilon);
            }

            return result;
        }

        template<FloatingPoint T>
        [[nodiscard]]
        DynamicMatrix<T> ColumnFromStdVector(
            const std::vector<T>& values)
        {
            DynamicMatrix<T> result(
                values.size(),
                1);

            for (std::size_t i = 0; i < values.size(); ++i)
            {
                result(i, 0) = values[i];
            }

            return result;
        }

        template<FloatingPoint T>
        [[nodiscard]]
        std::vector<T> ToStdVector(
            const DynamicMatrix<T>& column)
        {
            RequireColumnVector(
                column,
                "column");

            std::vector<T> result(
                column.Rows());

            for (std::size_t i = 0; i < column.Rows(); ++i)
            {
                result[i] = column(i, 0);
            }

            return result;
        }

        template<FloatingPoint T, typename Objective>
        [[nodiscard]]
        T BacktrackingLineSearch(
            Objective&& objective,
            const DynamicMatrix<T>& x,
            const DynamicMatrix<T>& direction,
            const DynamicMatrix<T>& gradient,
            T currentValue,
            const OptimizationSettings<T>& settings)
        {
            T stepScale =
                T(1);

            const T directionalDerivative =
                DotColumn(
                    gradient,
                    direction);

            while (stepScale >= settings.MinimumStepScale)
            {
                const DynamicMatrix<T> candidate =
                    AddScaled(
                        x,
                        direction,
                        stepScale);

                const T candidateValue =
                    objective(candidate);

                if (std::isfinite(candidateValue) &&
                    candidateValue <=
                        currentValue +
                        settings.ArmijoC1 *
                        stepScale *
                        directionalDerivative)
                {
                    return stepScale;
                }

                stepScale *=
                    settings.BacktrackingShrink;
            }

            return T(0);
        }

        template<FloatingPoint T>
        [[nodiscard]]
        T LeastSquaresValue(
            const DynamicMatrix<T>& residual)
        {
            RequireColumnVector(
                residual,
                "residual");

            return
                T(0.5) *
                NormSquared(residual);
        }

        template<FloatingPoint T>
        [[nodiscard]]
        DynamicMatrix<T> NormalEquationsGradient(
            const DynamicMatrix<T>& jacobian,
            const DynamicMatrix<T>& residual)
        {
            return
                jacobian.Transpose() *
                residual;
        }
    }

    //=========================================================
    // Gradient Descent
    //=========================================================

    template<FloatingPoint T, typename Objective, typename Gradient>
    [[nodiscard]]
    OptimizationResult<T> GradientDescent(
        Objective&& objective,
        Gradient&& gradient,
        DynamicMatrix<T> initialX,
        OptimizationSettings<T> settings = {})
    {
        optimization_detail::RequireColumnVector(
            initialX,
            "initialX");

        DynamicMatrix<T> x =
            std::move(initialX);

        T value =
            objective(x);

        OptimizationResult<T> result;
        result.X = x;
        result.Value = value;

        for (std::size_t iteration = 0;
             iteration < settings.MaxIterations;
             ++iteration)
        {
            const DynamicMatrix<T> g =
                gradient(x);

            optimization_detail::RequireSameShape(
                x,
                g,
                "GradientDescent");

            const T gradientNorm =
                optimization_detail::Norm(g);

            if (gradientNorm <= settings.GradientTolerance)
            {
                result =
                {
                    x,
                    value,
                    gradientNorm,
                    T(0),
                    iteration,
                    true,
                    "GradientDescent converged: gradient norm below tolerance."
                };

                return result;
            }

            const DynamicMatrix<T> direction =
                optimization_detail::Negative(g);

            T stepScale =
                settings.LearningRate;

            if (settings.UseLineSearch)
            {
                stepScale *=
                    optimization_detail::BacktrackingLineSearch(
                        objective,
                        x,
                        direction,
                        g,
                        value,
                        settings);
            }

            const DynamicMatrix<T> nextX =
                optimization_detail::AddScaled(
                    x,
                    direction,
                    stepScale);

            if (!optimization_detail::IsFinite(nextX))
            {
                result =
                {
                    x,
                    value,
                    gradientNorm,
                    T(0),
                    iteration,
                    false,
                    "GradientDescent stopped: non-finite iterate."
                };

                return result;
            }

            const T nextValue =
                objective(nextX);

            const T stepNorm =
                optimization_detail::Norm(
                    nextX - x);

            const T valueChange =
                std::abs(nextValue - value);

            x =
                nextX;

            value =
                nextValue;

            if (stepNorm <= settings.StepTolerance ||
                valueChange <= settings.ValueTolerance)
            {
                result =
                {
                    x,
                    value,
                    gradientNorm,
                    stepNorm,
                    iteration + 1,
                    true,
                    "GradientDescent converged: step/value tolerance reached."
                };

                return result;
            }
        }

        result =
        {
            x,
            value,
            optimization_detail::Norm(gradient(x)),
            T(0),
            settings.MaxIterations,
            false,
            "GradientDescent stopped: maximum iterations reached."
        };

        return result;
    }

    //=========================================================
    // Momentum Gradient Descent
    //=========================================================

    template<FloatingPoint T, typename Objective, typename Gradient>
    [[nodiscard]]
    OptimizationResult<T> Momentum(
        Objective&& objective,
        Gradient&& gradient,
        DynamicMatrix<T> initialX,
        T beta = T(0.9),
        OptimizationSettings<T> settings = {})
    {
        optimization_detail::RequireColumnVector(
            initialX,
            "initialX");

        DynamicMatrix<T> x =
            std::move(initialX);

        DynamicMatrix<T> velocity =
            optimization_detail::ZeroLike(x);

        T value =
            objective(x);

        for (std::size_t iteration = 0;
             iteration < settings.MaxIterations;
             ++iteration)
        {
            const DynamicMatrix<T> g =
                gradient(x);

            optimization_detail::RequireSameShape(
                x,
                g,
                "Momentum");

            const T gradientNorm =
                optimization_detail::Norm(g);

            if (gradientNorm <= settings.GradientTolerance)
            {
                return
                {
                    x,
                    value,
                    gradientNorm,
                    T(0),
                    iteration,
                    true,
                    "Momentum converged: gradient norm below tolerance."
                };
            }

            velocity =
                velocity * beta -
                g * settings.LearningRate;

            const DynamicMatrix<T> nextX =
                x + velocity;

            const T nextValue =
                objective(nextX);

            const T stepNorm =
                optimization_detail::Norm(
                    nextX - x);

            const T valueChange =
                std::abs(nextValue - value);

            x =
                nextX;

            value =
                nextValue;

            if (stepNorm <= settings.StepTolerance ||
                valueChange <= settings.ValueTolerance)
            {
                return
                {
                    x,
                    value,
                    gradientNorm,
                    stepNorm,
                    iteration + 1,
                    true,
                    "Momentum converged: step/value tolerance reached."
                };
            }
        }

        return
        {
            x,
            value,
            optimization_detail::Norm(gradient(x)),
            T(0),
            settings.MaxIterations,
            false,
            "Momentum stopped: maximum iterations reached."
        };
    }

    //=========================================================
    // Nesterov Accelerated Gradient
    //=========================================================

    template<FloatingPoint T, typename Objective, typename Gradient>
    [[nodiscard]]
    OptimizationResult<T> Nesterov(
        Objective&& objective,
        Gradient&& gradient,
        DynamicMatrix<T> initialX,
        T beta = T(0.9),
        OptimizationSettings<T> settings = {})
    {
        optimization_detail::RequireColumnVector(
            initialX,
            "initialX");

        DynamicMatrix<T> x =
            std::move(initialX);

        DynamicMatrix<T> velocity =
            optimization_detail::ZeroLike(x);

        T value =
            objective(x);

        for (std::size_t iteration = 0;
             iteration < settings.MaxIterations;
             ++iteration)
        {
            const DynamicMatrix<T> lookahead =
                x + velocity * beta;

            const DynamicMatrix<T> g =
                gradient(lookahead);

            optimization_detail::RequireSameShape(
                x,
                g,
                "Nesterov");

            const T gradientNorm =
                optimization_detail::Norm(g);

            if (gradientNorm <= settings.GradientTolerance)
            {
                return
                {
                    x,
                    value,
                    gradientNorm,
                    T(0),
                    iteration,
                    true,
                    "Nesterov converged: gradient norm below tolerance."
                };
            }

            velocity =
                velocity * beta -
                g * settings.LearningRate;

            const DynamicMatrix<T> nextX =
                x + velocity;

            const T nextValue =
                objective(nextX);

            const T stepNorm =
                optimization_detail::Norm(
                    nextX - x);

            const T valueChange =
                std::abs(nextValue - value);

            x =
                nextX;

            value =
                nextValue;

            if (stepNorm <= settings.StepTolerance ||
                valueChange <= settings.ValueTolerance)
            {
                return
                {
                    x,
                    value,
                    gradientNorm,
                    stepNorm,
                    iteration + 1,
                    true,
                    "Nesterov converged: step/value tolerance reached."
                };
            }
        }

        return
        {
            x,
            value,
            optimization_detail::Norm(gradient(x)),
            T(0),
            settings.MaxIterations,
            false,
            "Nesterov stopped: maximum iterations reached."
        };
    }

    //=========================================================
    // Adam
    //=========================================================

    template<FloatingPoint T, typename Objective, typename Gradient>
    [[nodiscard]]
    OptimizationResult<T> Adam(
        Objective&& objective,
        Gradient&& gradient,
        DynamicMatrix<T> initialX,
        T beta1 = T(0.9),
        T beta2 = T(0.999),
        T epsilon = T(1e-8),
        OptimizationSettings<T> settings = {})
    {
        optimization_detail::RequireColumnVector(
            initialX,
            "initialX");

        DynamicMatrix<T> x =
            std::move(initialX);

        DynamicMatrix<T> m =
            optimization_detail::ZeroLike(x);

        DynamicMatrix<T> v =
            optimization_detail::ZeroLike(x);

        T value =
            objective(x);

        T beta1Power =
            T(1);

        T beta2Power =
            T(1);

        for (std::size_t iteration = 0;
             iteration < settings.MaxIterations;
             ++iteration)
        {
            const DynamicMatrix<T> g =
                gradient(x);

            optimization_detail::RequireSameShape(
                x,
                g,
                "Adam");

            const T gradientNorm =
                optimization_detail::Norm(g);

            if (gradientNorm <= settings.GradientTolerance)
            {
                return
                {
                    x,
                    value,
                    gradientNorm,
                    T(0),
                    iteration,
                    true,
                    "Adam converged: gradient norm below tolerance."
                };
            }

            beta1Power *=
                beta1;

            beta2Power *=
                beta2;

            m =
                m * beta1 +
                g * (T(1) - beta1);

            v =
                v * beta2 +
                optimization_detail::HadamardSquare(g) *
                (T(1) - beta2);

            DynamicMatrix<T> mHat =
                m / (T(1) - beta1Power);

            DynamicMatrix<T> vHat =
                v / (T(1) - beta2Power);

            const DynamicMatrix<T> step =
                optimization_detail::ElementwiseDivide(
                    mHat,
                    optimization_detail::ElementwiseSqrt(vHat),
                    epsilon)
                * settings.LearningRate;

            const DynamicMatrix<T> nextX =
                x - step;

            const T nextValue =
                objective(nextX);

            const T stepNorm =
                optimization_detail::Norm(step);

            const T valueChange =
                std::abs(nextValue - value);

            x =
                nextX;

            value =
                nextValue;

            if (stepNorm <= settings.StepTolerance ||
                valueChange <= settings.ValueTolerance)
            {
                return
                {
                    x,
                    value,
                    gradientNorm,
                    stepNorm,
                    iteration + 1,
                    true,
                    "Adam converged: step/value tolerance reached."
                };
            }
        }

        return
        {
            x,
            value,
            optimization_detail::Norm(gradient(x)),
            T(0),
            settings.MaxIterations,
            false,
            "Adam stopped: maximum iterations reached."
        };
    }

    //=========================================================
    // Conjugate Gradient for SPD Linear Systems
    //
    // Solves:
    //     A * x = b
    //
    // Assumes:
    //     A is symmetric positive definite.
    //=========================================================

    template<FloatingPoint T>
    [[nodiscard]]
    ConjugateGradientResult<T> ConjugateGradientSolve(
        const DynamicMatrix<T>& A,
        const DynamicMatrix<T>& b,
        DynamicMatrix<T> initialX,
        OptimizationSettings<T> settings = {})
    {
        if (A.Rows() != A.Columns())
        {
            throw std::invalid_argument(
                "ConjugateGradientSolve failed: A must be square.");
        }

        optimization_detail::RequireColumnVector(
            b,
            "b");

        optimization_detail::RequireColumnVector(
            initialX,
            "initialX");

        if (A.Rows() != b.Rows() ||
            initialX.Rows() != b.Rows())
        {
            throw std::invalid_argument(
                "ConjugateGradientSolve failed: dimension mismatch.");
        }

        DynamicMatrix<T> x =
            std::move(initialX);

        DynamicMatrix<T> r =
            b - A * x;

        DynamicMatrix<T> p =
            r;

        T rsOld =
            optimization_detail::DotColumn(
                r,
                r);

        const T initialResidual =
            std::sqrt(rsOld);

        if (initialResidual <= settings.GradientTolerance)
        {
            return
            {
                x,
                initialResidual,
                0,
                true,
                "ConjugateGradientSolve converged: initial residual below tolerance."
            };
        }

        for (std::size_t iteration = 0;
             iteration < settings.MaxIterations;
             ++iteration)
        {
            const DynamicMatrix<T> Ap =
                A * p;

            const T denominator =
                optimization_detail::DotColumn(
                    p,
                    Ap);

            if (std::abs(denominator) <=
                std::numeric_limits<T>::epsilon())
            {
                return
                {
                    x,
                    std::sqrt(rsOld),
                    iteration,
                    false,
                    "ConjugateGradientSolve stopped: near-zero denominator."
                };
            }

            const T alpha =
                rsOld / denominator;

            x =
                x + p * alpha;

            r =
                r - Ap * alpha;

            const T rsNew =
                optimization_detail::DotColumn(
                    r,
                    r);

            const T residualNorm =
                std::sqrt(rsNew);

            if (residualNorm <= settings.GradientTolerance)
            {
                return
                {
                    x,
                    residualNorm,
                    iteration + 1,
                    true,
                    "ConjugateGradientSolve converged: residual below tolerance."
                };
            }

            const T beta =
                rsNew / rsOld;

            p =
                r + p * beta;

            rsOld =
                rsNew;
        }

        return
        {
            x,
            std::sqrt(rsOld),
            settings.MaxIterations,
            false,
            "ConjugateGradientSolve stopped: maximum iterations reached."
        };
    }

    template<FloatingPoint T>
    [[nodiscard]]
    ConjugateGradientResult<T> ConjugateGradientSolve(
        const DynamicMatrix<T>& A,
        const DynamicMatrix<T>& b,
        OptimizationSettings<T> settings = {})
    {
        return ConjugateGradientSolve(
            A,
            b,
            DynamicMatrix<T>(
                b.Rows(),
                1,
                T(0)),
            settings);
    }

    //=========================================================
    // Newton Method
    //=========================================================

    template<FloatingPoint T, typename Objective, typename Gradient, typename Hessian>
    [[nodiscard]]
    OptimizationResult<T> Newton(
        Objective&& objective,
        Gradient&& gradient,
        Hessian&& hessian,
        DynamicMatrix<T> initialX,
        OptimizationSettings<T> settings = {})
    {
        optimization_detail::RequireColumnVector(
            initialX,
            "initialX");

        DynamicMatrix<T> x =
            std::move(initialX);

        T value =
            objective(x);

        for (std::size_t iteration = 0;
             iteration < settings.MaxIterations;
             ++iteration)
        {
            const DynamicMatrix<T> g =
                gradient(x);

            optimization_detail::RequireSameShape(
                x,
                g,
                "Newton");

            const T gradientNorm =
                optimization_detail::Norm(g);

            if (gradientNorm <= settings.GradientTolerance)
            {
                return
                {
                    x,
                    value,
                    gradientNorm,
                    T(0),
                    iteration,
                    true,
                    "Newton converged: gradient norm below tolerance."
                };
            }

            const DynamicMatrix<T> H =
                hessian(x);

            if (H.Rows() != H.Columns() ||
                H.Rows() != x.Rows())
            {
                throw std::invalid_argument(
                    "Newton failed: Hessian must be square and match x dimension.");
            }

            const DynamicMatrix<T> rhs =
                optimization_detail::Negative(g);

            DynamicMatrix<T> direction =
                LinearSolve(
                    H,
                    rhs);

            T stepScale =
                T(1);

            if (settings.UseLineSearch)
            {
                stepScale =
                    optimization_detail::BacktrackingLineSearch(
                        objective,
                        x,
                        direction,
                        g,
                        value,
                        settings);
            }

            const DynamicMatrix<T> nextX =
                optimization_detail::AddScaled(
                    x,
                    direction,
                    stepScale);

            const T nextValue =
                objective(nextX);

            const T stepNorm =
                optimization_detail::Norm(
                    nextX - x);

            const T valueChange =
                std::abs(nextValue - value);

            x =
                nextX;

            value =
                nextValue;

            if (stepNorm <= settings.StepTolerance ||
                valueChange <= settings.ValueTolerance)
            {
                return
                {
                    x,
                    value,
                    gradientNorm,
                    stepNorm,
                    iteration + 1,
                    true,
                    "Newton converged: step/value tolerance reached."
                };
            }
        }

        return
        {
            x,
            value,
            optimization_detail::Norm(gradient(x)),
            T(0),
            settings.MaxIterations,
            false,
            "Newton stopped: maximum iterations reached."
        };
    }

    //=========================================================
    // Gauss-Newton
    //
    // Minimizes:
    //     1/2 * ||r(x)||^2
    //
    // User supplies:
    //     residual(x) -> column vector r
    //     jacobian(x) -> matrix J
    //=========================================================

    template<FloatingPoint T, typename Residual, typename Jacobian>
    [[nodiscard]]
    OptimizationResult<T> GaussNewton(
        Residual&& residual,
        Jacobian&& jacobian,
        DynamicMatrix<T> initialX,
        OptimizationSettings<T> settings = {})
    {
        optimization_detail::RequireColumnVector(
            initialX,
            "initialX");

        DynamicMatrix<T> x =
            std::move(initialX);

        for (std::size_t iteration = 0;
             iteration < settings.MaxIterations;
             ++iteration)
        {
            const DynamicMatrix<T> r =
                residual(x);

            optimization_detail::RequireColumnVector(
                r,
                "residual");

            const DynamicMatrix<T> J =
                jacobian(x);

            if (J.Rows() != r.Rows() ||
                J.Columns() != x.Rows())
            {
                throw std::invalid_argument(
                    "GaussNewton failed: Jacobian shape must be residual_size x parameter_count.");
            }

            const DynamicMatrix<T> JT =
                J.Transpose();

            const DynamicMatrix<T> gradient =
                JT * r;

            const T gradientNorm =
                optimization_detail::Norm(
                    gradient);

            const T value =
                optimization_detail::LeastSquaresValue(
                    r);

            if (gradientNorm <= settings.GradientTolerance)
            {
                return
                {
                    x,
                    value,
                    gradientNorm,
                    T(0),
                    iteration,
                    true,
                    "GaussNewton converged: gradient norm below tolerance."
                };
            }

            const DynamicMatrix<T> normalMatrix =
                JT * J;

            const DynamicMatrix<T> rhs =
                optimization_detail::Negative(
                    gradient);

            const DynamicMatrix<T> direction =
                LinearSolve(
                    normalMatrix,
                    rhs);

            auto objective =
                [&](const DynamicMatrix<T>& candidate) -> T
                {
                    return optimization_detail::LeastSquaresValue(
                        residual(candidate));
                };

            T stepScale =
                T(1);

            if (settings.UseLineSearch)
            {
                stepScale =
                    optimization_detail::BacktrackingLineSearch(
                        objective,
                        x,
                        direction,
                        gradient,
                        value,
                        settings);
            }

            const DynamicMatrix<T> nextX =
                optimization_detail::AddScaled(
                    x,
                    direction,
                    stepScale);

            const T nextValue =
                objective(nextX);

            const T stepNorm =
                optimization_detail::Norm(
                    nextX - x);

            const T valueChange =
                std::abs(nextValue - value);

            x =
                nextX;

            if (stepNorm <= settings.StepTolerance ||
                valueChange <= settings.ValueTolerance)
            {
                return
                {
                    x,
                    nextValue,
                    gradientNorm,
                    stepNorm,
                    iteration + 1,
                    true,
                    "GaussNewton converged: step/value tolerance reached."
                };
            }
        }

        const DynamicMatrix<T> finalResidual =
            residual(x);

        const DynamicMatrix<T> finalJ =
            jacobian(x);

        const DynamicMatrix<T> finalGradient =
            finalJ.Transpose() *
            finalResidual;

        return
        {
            x,
            optimization_detail::LeastSquaresValue(finalResidual),
            optimization_detail::Norm(finalGradient),
            T(0),
            settings.MaxIterations,
            false,
            "GaussNewton stopped: maximum iterations reached."
        };
    }

    //=========================================================
    // Levenberg-Marquardt
    //
    // Solves damped normal equations:
    //
    //     (J^T J + lambda I) delta = -J^T r
    //
    // Lambda decreases on successful steps and increases on rejected steps.
    //=========================================================

    template<FloatingPoint T, typename Residual, typename Jacobian>
    [[nodiscard]]
    OptimizationResult<T> LevenbergMarquardt(
        Residual&& residual,
        Jacobian&& jacobian,
        DynamicMatrix<T> initialX,
        T initialLambda = T(1e-3),
        T lambdaUp = T(10),
        T lambdaDown = T(0.1),
        OptimizationSettings<T> settings = {})
    {
        optimization_detail::RequireColumnVector(
            initialX,
            "initialX");

        DynamicMatrix<T> x =
            std::move(initialX);

        T lambda =
            initialLambda;

        DynamicMatrix<T> r =
            residual(x);

        optimization_detail::RequireColumnVector(
            r,
            "residual");

        T value =
            optimization_detail::LeastSquaresValue(
                r);

        for (std::size_t iteration = 0;
             iteration < settings.MaxIterations;
             ++iteration)
        {
            const DynamicMatrix<T> J =
                jacobian(x);

            if (J.Rows() != r.Rows() ||
                J.Columns() != x.Rows())
            {
                throw std::invalid_argument(
                    "LevenbergMarquardt failed: Jacobian shape must be residual_size x parameter_count.");
            }

            const DynamicMatrix<T> JT =
                J.Transpose();

            const DynamicMatrix<T> gradient =
                JT * r;

            const T gradientNorm =
                optimization_detail::Norm(
                    gradient);

            if (gradientNorm <= settings.GradientTolerance)
            {
                return
                {
                    x,
                    value,
                    gradientNorm,
                    T(0),
                    iteration,
                    true,
                    "LevenbergMarquardt converged: gradient norm below tolerance."
                };
            }

            DynamicMatrix<T> damped =
                JT * J;

            for (std::size_t i = 0; i < damped.Rows(); ++i)
            {
                damped(i, i) += lambda;
            }

            const DynamicMatrix<T> rhs =
                optimization_detail::Negative(
                    gradient);

            DynamicMatrix<T> direction;

            try
            {
                direction =
                    LinearSolve(
                        damped,
                        rhs);
            }
            catch (...)
            {
                lambda *=
                    lambdaUp;

                continue;
            }

            const DynamicMatrix<T> candidateX =
                x + direction;

            const DynamicMatrix<T> candidateResidual =
                residual(candidateX);

            const T candidateValue =
                optimization_detail::LeastSquaresValue(
                    candidateResidual);

            const T stepNorm =
                optimization_detail::Norm(
                    direction);

            if (std::isfinite(candidateValue) &&
                candidateValue < value)
            {
                const T valueChange =
                    std::abs(value - candidateValue);

                x =
                    candidateX;

                r =
                    candidateResidual;

                value =
                    candidateValue;

                lambda =
                    std::max(
                        lambda * lambdaDown,
                        std::numeric_limits<T>::epsilon());

                if (stepNorm <= settings.StepTolerance ||
                    valueChange <= settings.ValueTolerance)
                {
                    return
                    {
                        x,
                        value,
                        gradientNorm,
                        stepNorm,
                        iteration + 1,
                        true,
                        "LevenbergMarquardt converged: step/value tolerance reached."
                    };
                }
            }
            else
            {
                lambda *=
                    lambdaUp;
            }
        }

        const DynamicMatrix<T> finalJ =
            jacobian(x);

        const DynamicMatrix<T> finalGradient =
            finalJ.Transpose() *
            residual(x);

        return
        {
            x,
            value,
            optimization_detail::Norm(finalGradient),
            T(0),
            settings.MaxIterations,
            false,
            "LevenbergMarquardt stopped: maximum iterations reached."
        };
    }

} // namespace kairo::foundation::math
