module;

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

export module Kairo.Foundation.Math.Optimization;

import Kairo.Foundation.Math.Vector;
import Kairo.Foundation.Math.DynamicMatrix;
import Kairo.Foundation.Math.LinearAlgebra.LinearSolve;

export namespace kairo::foundation::math
{
    /// Input: convergence tolerances and iteration limits for optimization routines.
    /// Output: configuration object copied into iterative solvers.
    /// Task: keep optimizer controls explicit without forcing every algorithm to
    /// grow a long parameter list. `GradientTolerance` measures the first-order
    /// stationarity residual and `StepTolerance` stops when updates become too
    /// small to matter for the supplied scalar type.
    template<FloatingPoint T>
    struct OptimizationOptions final
    {
        std::size_t MaxIterations = 256;
        T LearningRate = T(1e-2);
        T GradientTolerance = T(1e-6);
        T StepTolerance = T(1e-9);
        T Damping = T(1e-3);
        T Beta1 = T(0.9);
        T Beta2 = T(0.999);
        T Epsilon = T(1e-8);
    };

    /// Input: final iterate, objective value, and convergence metadata.
    /// Output: value object returned by nonlinear optimizers.
    /// Task: make optimizer termination inspectable instead of returning only
    /// the final vector and hiding whether tolerance or iteration budget ended
    /// the solve.
    template<FloatingPoint T>
    struct OptimizationResult final
    {
        std::vector<T> Position;
        T Value = T(0);
        std::size_t Iterations = 0;
        T GradientNorm = T(0);
        bool Converged = false;
    };

    /// Input: final Krylov solution and residual metadata.
    /// Output: value object returned by iterative linear solvers.
    /// Task: expose solver quality for sparse systems where a caller may choose
    /// between more iterations, a different preconditioner, or a direct fallback.
    template<FloatingPoint T>
    struct IterativeSolveResult final
    {
        std::vector<T> Solution;
        std::size_t Iterations = 0;
        T ResidualNorm = T(0);
        bool Converged = false;
    };

    /// Input: equality-constrained quadratic program solution.
    /// Output: primal variables, Lagrange multipliers, objective value, and KKT residual.
    /// Task: preserve both sides of the KKT solve so physics/optimization callers
    /// can inspect constraint forces or sensitivities later.
    template<FloatingPoint T>
    struct EqualityConstrainedQPResult final
    {
        std::vector<T> Primal;
        std::vector<T> Multipliers;
        T ObjectiveValue = T(0);
        T KKTResidualNorm = T(0);
    };

    namespace optimization_detail
    {
        template<FloatingPoint T>
        void RequireSameSize(
            const std::vector<T>& lhs,
            const std::vector<T>& rhs,
            const char* message)
        {
            if (lhs.size() != rhs.size())
            {
                throw std::invalid_argument(message);
            }
        }

        template<FloatingPoint T>
        [[nodiscard]]
        T DotVector(
            const std::vector<T>& lhs,
            const std::vector<T>& rhs)
        {
            RequireSameSize(lhs, rhs, "Vector dot product failed: sizes must match.");

            T sum = T(0);
            for (std::size_t i = 0; i < lhs.size(); ++i)
            {
                sum += lhs[i] * rhs[i];
            }

            return sum;
        }

        template<FloatingPoint T>
        [[nodiscard]]
        T Norm(
            const std::vector<T>& values)
        {
            return std::sqrt(DotVector(values, values));
        }

        template<FloatingPoint T>
        [[nodiscard]]
        std::vector<T> AddScaled(
            const std::vector<T>& lhs,
            const std::vector<T>& rhs,
            T scale)
        {
            RequireSameSize(lhs, rhs, "Vector scaled addition failed: sizes must match.");

            std::vector<T> result(lhs.size());
            for (std::size_t i = 0; i < lhs.size(); ++i)
            {
                result[i] = lhs[i] + (rhs[i] * scale);
            }

            return result;
        }

        template<FloatingPoint T>
        [[nodiscard]]
        std::vector<T> Subtract(
            const std::vector<T>& lhs,
            const std::vector<T>& rhs)
        {
            return AddScaled(lhs, rhs, T(-1));
        }

        template<FloatingPoint T>
        [[nodiscard]]
        std::vector<T> MatrixVector(
            const DynamicMatrix<T>& matrix,
            const std::vector<T>& vector)
        {
            if (matrix.Columns() != vector.size())
            {
                throw std::invalid_argument("Matrix-vector multiply failed: matrix columns must match vector size.");
            }

            std::vector<T> result(matrix.Rows(), T(0));
            for (std::size_t row = 0; row < matrix.Rows(); ++row)
            {
                for (std::size_t col = 0; col < matrix.Columns(); ++col)
                {
                    result[row] += matrix(row, col) * vector[col];
                }
            }

            return result;
        }

        template<FloatingPoint T>
        [[nodiscard]]
        std::vector<T> TransposeMatrixVector(
            const DynamicMatrix<T>& matrix,
            const std::vector<T>& vector)
        {
            if (matrix.Rows() != vector.size())
            {
                throw std::invalid_argument("Transpose matrix-vector multiply failed: matrix rows must match vector size.");
            }

            std::vector<T> result(matrix.Columns(), T(0));
            for (std::size_t row = 0; row < matrix.Rows(); ++row)
            {
                for (std::size_t col = 0; col < matrix.Columns(); ++col)
                {
                    result[col] += matrix(row, col) * vector[row];
                }
            }

            return result;
        }

        template<FloatingPoint T>
        [[nodiscard]]
        DynamicMatrix<T> OuterProduct(
            const DynamicMatrix<T>& lhs,
            const DynamicMatrix<T>& rhs)
        {
            return lhs.Transpose() * rhs;
        }

        template<FloatingPoint T>
        [[nodiscard]]
        T QuadraticObjectiveValue(
            const DynamicMatrix<T>& H,
            const std::vector<T>& g,
            const std::vector<T>& x)
        {
            const std::vector<T> Hx =
                MatrixVector(H, x);

            return (T(0.5) * DotVector(x, Hx)) +
                DotVector(g, x);
        }
    }

    /// Input: objective `f(x)`, gradient `grad(x)`, initial point, and options.
    /// Output: iterative minimization result.
    /// Task: provide the simplest first-order optimizer for smooth objectives.
    /// Invalid gradient dimensions are runtime input errors and throw.
    template<FloatingPoint T, typename Objective, typename Gradient>
    [[nodiscard]]
    OptimizationResult<T> GradientDescent(
        Objective objective,
        Gradient gradient,
        std::vector<T> initial,
        OptimizationOptions<T> options = {})
    {
        std::vector<T> x =
            std::move(initial);

        OptimizationResult<T> result;
        result.Position = x;

        for (std::size_t iteration = 0; iteration < options.MaxIterations; ++iteration)
        {
            const std::vector<T> grad =
                gradient(x);

            optimization_detail::RequireSameSize(
                x,
                grad,
                "GradientDescent failed: gradient size must match position size.");

            const T gradientNorm =
                optimization_detail::Norm(grad);

            result.Iterations = iteration;
            result.GradientNorm = gradientNorm;
            if (gradientNorm <= options.GradientTolerance)
            {
                result.Converged = true;
                break;
            }

            const std::vector<T> next =
                optimization_detail::AddScaled(
                    x,
                    grad,
                    -options.LearningRate);

            const T stepNorm =
                optimization_detail::Norm(
                    optimization_detail::Subtract(next, x));

            x = next;
            result.Position = x;
            result.Value = objective(x);
            result.Iterations = iteration + 1;

            if (stepNorm <= options.StepTolerance)
            {
                result.Converged = true;
                break;
            }
        }

        result.Position = x;
        result.Value = objective(x);
        result.GradientNorm =
            optimization_detail::Norm(gradient(x));
        return result;
    }

    /// Input: objective, gradient, initial point, and momentum options.
    /// Output: iterative minimization result.
    /// Task: accelerate gradient descent by accumulating an exponential moving
    /// direction. `Beta1` is the momentum coefficient.
    template<FloatingPoint T, typename Objective, typename Gradient>
    [[nodiscard]]
    OptimizationResult<T> Momentum(
        Objective objective,
        Gradient gradient,
        std::vector<T> initial,
        OptimizationOptions<T> options = {})
    {
        std::vector<T> x =
            std::move(initial);

        std::vector<T> velocity(x.size(), T(0));
        OptimizationResult<T> result;

        for (std::size_t iteration = 0; iteration < options.MaxIterations; ++iteration)
        {
            const std::vector<T> grad =
                gradient(x);

            optimization_detail::RequireSameSize(
                x,
                grad,
                "Momentum failed: gradient size must match position size.");

            const T gradientNorm =
                optimization_detail::Norm(grad);

            if (gradientNorm <= options.GradientTolerance)
            {
                result.Converged = true;
                result.Iterations = iteration;
                break;
            }

            for (std::size_t i = 0; i < x.size(); ++i)
            {
                velocity[i] = (options.Beta1 * velocity[i]) -
                    (options.LearningRate * grad[i]);
                x[i] += velocity[i];
            }

            result.Iterations = iteration + 1;
            if (optimization_detail::Norm(velocity) <= options.StepTolerance)
            {
                result.Converged = true;
                break;
            }
        }

        result.Position = x;
        result.Value = objective(x);
        result.GradientNorm =
            optimization_detail::Norm(gradient(x));
        return result;
    }

    /// Input: objective, gradient, initial point, and Nesterov options.
    /// Output: iterative minimization result.
    /// Task: evaluate the gradient at the lookahead position to reduce lag in
    /// the accumulated momentum direction.
    template<FloatingPoint T, typename Objective, typename Gradient>
    [[nodiscard]]
    OptimizationResult<T> Nesterov(
        Objective objective,
        Gradient gradient,
        std::vector<T> initial,
        OptimizationOptions<T> options = {})
    {
        std::vector<T> x =
            std::move(initial);

        std::vector<T> velocity(x.size(), T(0));
        OptimizationResult<T> result;

        for (std::size_t iteration = 0; iteration < options.MaxIterations; ++iteration)
        {
            const std::vector<T> lookahead =
                optimization_detail::AddScaled(
                    x,
                    velocity,
                    options.Beta1);

            const std::vector<T> grad =
                gradient(lookahead);

            optimization_detail::RequireSameSize(
                x,
                grad,
                "Nesterov failed: gradient size must match position size.");

            const T gradientNorm =
                optimization_detail::Norm(grad);

            if (gradientNorm <= options.GradientTolerance)
            {
                result.Converged = true;
                result.Iterations = iteration;
                break;
            }

            for (std::size_t i = 0; i < x.size(); ++i)
            {
                velocity[i] = (options.Beta1 * velocity[i]) -
                    (options.LearningRate * grad[i]);
                x[i] += velocity[i];
            }

            result.Iterations = iteration + 1;
            if (optimization_detail::Norm(velocity) <= options.StepTolerance)
            {
                result.Converged = true;
                break;
            }
        }

        result.Position = x;
        result.Value = objective(x);
        result.GradientNorm =
            optimization_detail::Norm(gradient(x));
        return result;
    }

    /// Input: objective, gradient, initial point, and Adam options.
    /// Output: iterative minimization result.
    /// Task: provide an adaptive first-order optimizer with bias-corrected first
    /// and second gradient moments.
    template<FloatingPoint T, typename Objective, typename Gradient>
    [[nodiscard]]
    OptimizationResult<T> Adam(
        Objective objective,
        Gradient gradient,
        std::vector<T> initial,
        OptimizationOptions<T> options = {})
    {
        std::vector<T> x =
            std::move(initial);

        std::vector<T> m(x.size(), T(0));
        std::vector<T> v(x.size(), T(0));
        OptimizationResult<T> result;

        for (std::size_t iteration = 0; iteration < options.MaxIterations; ++iteration)
        {
            const std::vector<T> grad =
                gradient(x);

            optimization_detail::RequireSameSize(
                x,
                grad,
                "Adam failed: gradient size must match position size.");

            const T gradientNorm =
                optimization_detail::Norm(grad);

            if (gradientNorm <= options.GradientTolerance)
            {
                result.Converged = true;
                result.Iterations = iteration;
                break;
            }

            const T beta1Correction =
                T(1) - std::pow(options.Beta1, static_cast<T>(iteration + 1));

            const T beta2Correction =
                T(1) - std::pow(options.Beta2, static_cast<T>(iteration + 1));

            std::vector<T> step(x.size(), T(0));
            for (std::size_t i = 0; i < x.size(); ++i)
            {
                m[i] = (options.Beta1 * m[i]) +
                    ((T(1) - options.Beta1) * grad[i]);

                v[i] = (options.Beta2 * v[i]) +
                    ((T(1) - options.Beta2) * grad[i] * grad[i]);

                const T mHat =
                    m[i] / beta1Correction;

                const T vHat =
                    v[i] / beta2Correction;

                step[i] =
                    -options.LearningRate * mHat /
                    (std::sqrt(vHat) + options.Epsilon);

                x[i] += step[i];
            }

            result.Iterations = iteration + 1;
            if (optimization_detail::Norm(step) <= options.StepTolerance)
            {
                result.Converged = true;
                break;
            }
        }

        result.Position = x;
        result.Value = objective(x);
        result.GradientNorm =
            optimization_detail::Norm(gradient(x));
        return result;
    }

    /// Input: objective, gradient, Hessian, initial point, and options.
    /// Output: Newton minimization result.
    /// Task: solve `H(x) step = grad(x)` and subtract the step. This is for
    /// smooth problems where callers can provide a well-conditioned Hessian.
    template<FloatingPoint T, typename Objective, typename Gradient, typename Hessian>
    [[nodiscard]]
    OptimizationResult<T> Newton(
        Objective objective,
        Gradient gradient,
        Hessian hessian,
        std::vector<T> initial,
        OptimizationOptions<T> options = {})
    {
        std::vector<T> x =
            std::move(initial);

        OptimizationResult<T> result;

        for (std::size_t iteration = 0; iteration < options.MaxIterations; ++iteration)
        {
            const std::vector<T> grad =
                gradient(x);

            optimization_detail::RequireSameSize(
                x,
                grad,
                "Newton failed: gradient size must match position size.");

            const T gradientNorm =
                optimization_detail::Norm(grad);

            if (gradientNorm <= options.GradientTolerance)
            {
                result.Converged = true;
                result.Iterations = iteration;
                break;
            }

            DynamicMatrix<T> H =
                hessian(x);

            if (H.Rows() != x.size() || H.Columns() != x.size())
            {
                throw std::invalid_argument("Newton failed: Hessian must be square with one row per variable.");
            }

            const std::vector<T> step =
                LinearSolve(H, grad);

            x =
                optimization_detail::AddScaled(
                    x,
                    step,
                    T(-1));

            result.Iterations = iteration + 1;
            if (optimization_detail::Norm(step) <= options.StepTolerance)
            {
                result.Converged = true;
                break;
            }
        }

        result.Position = x;
        result.Value = objective(x);
        result.GradientNorm =
            optimization_detail::Norm(gradient(x));
        return result;
    }

    /// Input: residual function, Jacobian function, initial point, and options.
    /// Output: least-squares minimization result.
    /// Task: solve nonlinear least-squares through the normal equations
    /// `(J^T J) step = -J^T r`. Residual count may differ from variable count.
    template<FloatingPoint T, typename Residual, typename Jacobian>
    [[nodiscard]]
    OptimizationResult<T> GaussNewton(
        Residual residual,
        Jacobian jacobian,
        std::vector<T> initial,
        OptimizationOptions<T> options = {})
    {
        std::vector<T> x =
            std::move(initial);

        OptimizationResult<T> result;

        for (std::size_t iteration = 0; iteration < options.MaxIterations; ++iteration)
        {
            const std::vector<T> r =
                residual(x);

            const DynamicMatrix<T> J =
                jacobian(x);

            if (J.Rows() != r.size() || J.Columns() != x.size())
            {
                throw std::invalid_argument("GaussNewton failed: Jacobian shape must be residuals x variables.");
            }

            const DynamicMatrix<T> JTJ =
                J.Transpose() * J;

            const std::vector<T> gradient =
                optimization_detail::TransposeMatrixVector(J, r);

            const T gradientNorm =
                optimization_detail::Norm(gradient);

            if (gradientNorm <= options.GradientTolerance)
            {
                result.Converged = true;
                result.Iterations = iteration;
                break;
            }

            std::vector<T> rhs(gradient.size());
            for (std::size_t i = 0; i < gradient.size(); ++i)
            {
                rhs[i] = -gradient[i];
            }

            const std::vector<T> step =
                LinearSolve(JTJ, rhs);

            x =
                optimization_detail::AddScaled(
                    x,
                    step,
                    T(1));

            result.Iterations = iteration + 1;
            if (optimization_detail::Norm(step) <= options.StepTolerance)
            {
                result.Converged = true;
                break;
            }
        }

        const std::vector<T> finalResidual =
            residual(x);

        result.Position = x;
        result.Value =
            T(0.5) * optimization_detail::DotVector(finalResidual, finalResidual);
        result.GradientNorm =
            optimization_detail::Norm(
                optimization_detail::TransposeMatrixVector(
                    jacobian(x),
                    finalResidual));
        return result;
    }

    /// Input: residual function, Jacobian function, initial point, and damping options.
    /// Output: damped least-squares minimization result.
    /// Task: blend Gauss-Newton with gradient descent behavior by solving
    /// `(J^T J + lambda I) step = -J^T r`. This is safer for ill-conditioned
    /// residual problems than a pure normal-equation step.
    template<FloatingPoint T, typename Residual, typename Jacobian>
    [[nodiscard]]
    OptimizationResult<T> LevenbergMarquardt(
        Residual residual,
        Jacobian jacobian,
        std::vector<T> initial,
        OptimizationOptions<T> options = {})
    {
        std::vector<T> x =
            std::move(initial);

        T damping =
            options.Damping;

        OptimizationResult<T> result;

        auto objectiveFromResidual =
            [](const std::vector<T>& r) -> T
            {
                return T(0.5) * optimization_detail::DotVector(r, r);
            };

        for (std::size_t iteration = 0; iteration < options.MaxIterations; ++iteration)
        {
            const std::vector<T> r =
                residual(x);

            const DynamicMatrix<T> J =
                jacobian(x);

            if (J.Rows() != r.size() || J.Columns() != x.size())
            {
                throw std::invalid_argument("LevenbergMarquardt failed: Jacobian shape must be residuals x variables.");
            }

            DynamicMatrix<T> system =
                J.Transpose() * J;

            const std::vector<T> gradient =
                optimization_detail::TransposeMatrixVector(J, r);

            const T gradientNorm =
                optimization_detail::Norm(gradient);

            if (gradientNorm <= options.GradientTolerance)
            {
                result.Converged = true;
                result.Iterations = iteration;
                break;
            }

            for (std::size_t diagonal = 0; diagonal < system.Rows(); ++diagonal)
            {
                system(diagonal, diagonal) += damping;
            }

            std::vector<T> rhs(gradient.size());
            for (std::size_t i = 0; i < gradient.size(); ++i)
            {
                rhs[i] = -gradient[i];
            }

            const std::vector<T> step =
                LinearSolve(system, rhs);

            const std::vector<T> candidate =
                optimization_detail::AddScaled(
                    x,
                    step,
                    T(1));

            if (objectiveFromResidual(residual(candidate)) <= objectiveFromResidual(r))
            {
                x = candidate;
                damping = std::max(damping * T(0.5), std::numeric_limits<T>::epsilon());
            }
            else
            {
                damping *= T(2);
            }

            result.Iterations = iteration + 1;
            if (optimization_detail::Norm(step) <= options.StepTolerance)
            {
                result.Converged = true;
                break;
            }
        }

        const std::vector<T> finalResidual =
            residual(x);

        result.Position = x;
        result.Value =
            T(0.5) * optimization_detail::DotVector(finalResidual, finalResidual);
        result.GradientNorm =
            optimization_detail::Norm(
                optimization_detail::TransposeMatrixVector(
                    jacobian(x),
                    finalResidual));
        return result;
    }

    /// Input: square system matrix, right-hand side, optional initial guess, and options.
    /// Output: approximate solution of `A x = b`.
    /// Task: solve symmetric positive-definite systems without forming a
    /// decomposition. This is the sparse-friendly baseline used by physics and
    /// simulation systems.
    template<FloatingPoint T>
    [[nodiscard]]
    IterativeSolveResult<T> ConjugateGradient(
        const DynamicMatrix<T>& A,
        const std::vector<T>& b,
        std::vector<T> initial = {},
        OptimizationOptions<T> options = {})
    {
        if (A.Rows() != A.Columns())
        {
            throw std::invalid_argument("ConjugateGradient failed: matrix must be square.");
        }
        if (A.Rows() != b.size())
        {
            throw std::invalid_argument("ConjugateGradient failed: matrix rows must match b size.");
        }

        std::vector<T> x =
            initial.empty()
                ? std::vector<T>(b.size(), T(0))
                : std::move(initial);

        optimization_detail::RequireSameSize(
            x,
            b,
            "ConjugateGradient failed: initial guess size must match b size.");

        std::vector<T> r =
            optimization_detail::Subtract(
                b,
                optimization_detail::MatrixVector(A, x));

        std::vector<T> p =
            r;

        T rsOld =
            optimization_detail::DotVector(r, r);

        IterativeSolveResult<T> result;
        for (std::size_t iteration = 0; iteration < options.MaxIterations; ++iteration)
        {
            const std::vector<T> Ap =
                optimization_detail::MatrixVector(A, p);

            const T denominator =
                optimization_detail::DotVector(p, Ap);

            if (std::abs(denominator) <= std::numeric_limits<T>::epsilon())
            {
                throw std::runtime_error("ConjugateGradient failed: search direction became singular.");
            }

            const T alpha =
                rsOld / denominator;

            x =
                optimization_detail::AddScaled(
                    x,
                    p,
                    alpha);

            r =
                optimization_detail::AddScaled(
                    r,
                    Ap,
                    -alpha);

            const T residualNorm =
                optimization_detail::Norm(r);

            result.Iterations = iteration + 1;
            result.ResidualNorm = residualNorm;
            if (residualNorm <= options.GradientTolerance)
            {
                result.Converged = true;
                break;
            }

            const T rsNew =
                optimization_detail::DotVector(r, r);

            const T beta =
                rsNew / rsOld;

            for (std::size_t i = 0; i < p.size(); ++i)
            {
                p[i] = r[i] + (beta * p[i]);
            }

            rsOld = rsNew;
        }

        result.Solution = x;
        result.ResidualNorm =
            optimization_detail::Norm(
                optimization_detail::Subtract(
                    b,
                    optimization_detail::MatrixVector(A, x)));
        return result;
    }

    /// Input: square system, right-hand side, inverse diagonal preconditioner,
    /// optional initial guess, and options.
    /// Output: approximate solution of `A x = b`.
    /// Task: provide a preconditioned CG path where callers can pass `M^-1` as
    /// a vector. For Jacobi preconditioning, each entry is `1 / A(i,i)`.
    template<FloatingPoint T>
    [[nodiscard]]
    IterativeSolveResult<T> PreconditionedConjugateGradient(
        const DynamicMatrix<T>& A,
        const std::vector<T>& b,
        const std::vector<T>& inversePreconditionerDiagonal,
        std::vector<T> initial = {},
        OptimizationOptions<T> options = {})
    {
        if (A.Rows() != A.Columns())
        {
            throw std::invalid_argument("PreconditionedConjugateGradient failed: matrix must be square.");
        }
        optimization_detail::RequireSameSize(
            b,
            inversePreconditionerDiagonal,
            "PreconditionedConjugateGradient failed: preconditioner size must match b size.");

        std::vector<T> x =
            initial.empty()
                ? std::vector<T>(b.size(), T(0))
                : std::move(initial);

        optimization_detail::RequireSameSize(
            x,
            b,
            "PreconditionedConjugateGradient failed: initial guess size must match b size.");

        std::vector<T> r =
            optimization_detail::Subtract(
                b,
                optimization_detail::MatrixVector(A, x));

        std::vector<T> z(r.size());
        for (std::size_t i = 0; i < r.size(); ++i)
        {
            z[i] = inversePreconditionerDiagonal[i] * r[i];
        }

        std::vector<T> p =
            z;

        T rzOld =
            optimization_detail::DotVector(r, z);

        IterativeSolveResult<T> result;
        for (std::size_t iteration = 0; iteration < options.MaxIterations; ++iteration)
        {
            const std::vector<T> Ap =
                optimization_detail::MatrixVector(A, p);

            const T denominator =
                optimization_detail::DotVector(p, Ap);

            if (std::abs(denominator) <= std::numeric_limits<T>::epsilon())
            {
                throw std::runtime_error("PreconditionedConjugateGradient failed: search direction became singular.");
            }

            const T alpha =
                rzOld / denominator;

            x =
                optimization_detail::AddScaled(
                    x,
                    p,
                    alpha);

            r =
                optimization_detail::AddScaled(
                    r,
                    Ap,
                    -alpha);

            const T residualNorm =
                optimization_detail::Norm(r);

            result.Iterations = iteration + 1;
            result.ResidualNorm = residualNorm;
            if (residualNorm <= options.GradientTolerance)
            {
                result.Converged = true;
                break;
            }

            for (std::size_t i = 0; i < r.size(); ++i)
            {
                z[i] = inversePreconditionerDiagonal[i] * r[i];
            }

            const T rzNew =
                optimization_detail::DotVector(r, z);

            const T beta =
                rzNew / rzOld;

            for (std::size_t i = 0; i < p.size(); ++i)
            {
                p[i] = z[i] + (beta * p[i]);
            }

            rzOld = rzNew;
        }

        result.Solution = x;
        result.ResidualNorm =
            optimization_detail::Norm(
                optimization_detail::Subtract(
                    b,
                    optimization_detail::MatrixVector(A, x)));
        return result;
    }

    /// Input: square system, right-hand side, optional initial guess, options, and restart size.
    /// Output: approximate solution of `A x = b`.
    /// Task: solve general nonsymmetric systems through restarted GMRES. The
    /// implementation stores a compact Krylov basis and solves the small least
    /// squares problem through normal equations, which is appropriate for the
    /// foundation-scale matrices covered by this module.
    template<FloatingPoint T>
    [[nodiscard]]
    IterativeSolveResult<T> GMRES(
        const DynamicMatrix<T>& A,
        const std::vector<T>& b,
        std::vector<T> initial = {},
        OptimizationOptions<T> options = {},
        std::size_t restart = 30)
    {
        if (A.Rows() != A.Columns())
        {
            throw std::invalid_argument("GMRES failed: matrix must be square.");
        }
        if (A.Rows() != b.size())
        {
            throw std::invalid_argument("GMRES failed: matrix rows must match b size.");
        }
        if (restart == 0)
        {
            throw std::invalid_argument("GMRES failed: restart must be non-zero.");
        }

        const std::size_t n =
            b.size();

        std::vector<T> x =
            initial.empty()
                ? std::vector<T>(n, T(0))
                : std::move(initial);

        optimization_detail::RequireSameSize(
            x,
            b,
            "GMRES failed: initial guess size must match b size.");

        IterativeSolveResult<T> result;
        std::size_t totalIterations = 0;

        while (totalIterations < options.MaxIterations)
        {
            const std::vector<T> residual =
                optimization_detail::Subtract(
                    b,
                    optimization_detail::MatrixVector(A, x));

            const T beta =
                optimization_detail::Norm(residual);

            if (beta <= options.GradientTolerance)
            {
                result.Converged = true;
                result.ResidualNorm = beta;
                break;
            }

            const std::size_t innerLimit =
                std::min(restart, options.MaxIterations - totalIterations);

            DynamicMatrix<T> V(n, innerLimit + 1, T(0));
            for (std::size_t row = 0; row < n; ++row)
            {
                V(row, 0) = residual[row] / beta;
            }

            DynamicMatrix<T> H(innerLimit + 1, innerLimit, T(0));
            std::size_t usedInner = 0;

            for (std::size_t j = 0; j < innerLimit; ++j)
            {
                std::vector<T> vj(n);
                for (std::size_t row = 0; row < n; ++row)
                {
                    vj[row] = V(row, j);
                }

                std::vector<T> w =
                    optimization_detail::MatrixVector(A, vj);

                for (std::size_t i = 0; i <= j; ++i)
                {
                    std::vector<T> vi(n);
                    for (std::size_t row = 0; row < n; ++row)
                    {
                        vi[row] = V(row, i);
                    }

                    H(i, j) =
                        optimization_detail::DotVector(w, vi);

                    w =
                        optimization_detail::AddScaled(
                            w,
                            vi,
                            -H(i, j));
                }

                H(j + 1, j) =
                    optimization_detail::Norm(w);

                if (H(j + 1, j) > options.Epsilon)
                {
                    for (std::size_t row = 0; row < n; ++row)
                    {
                        V(row, j + 1) = w[row] / H(j + 1, j);
                    }
                }

                usedInner = j + 1;

                DynamicMatrix<T> Hk(usedInner + 1, usedInner, T(0));
                for (std::size_t row = 0; row < usedInner + 1; ++row)
                {
                    for (std::size_t col = 0; col < usedInner; ++col)
                    {
                        Hk(row, col) = H(row, col);
                    }
                }

                std::vector<T> rhs(usedInner + 1, T(0));
                rhs[0] = beta;

                const std::vector<T> normalRhs =
                    optimization_detail::TransposeMatrixVector(Hk, rhs);

                const std::vector<T> y =
                    LinearSolve(
                        Hk.Transpose() * Hk,
                        normalRhs);

                std::vector<T> candidate =
                    x;

                for (std::size_t col = 0; col < usedInner; ++col)
                {
                    for (std::size_t row = 0; row < n; ++row)
                    {
                        candidate[row] += V(row, col) * y[col];
                    }
                }

                const T candidateResidual =
                    optimization_detail::Norm(
                        optimization_detail::Subtract(
                            b,
                            optimization_detail::MatrixVector(A, candidate)));

                ++totalIterations;

                if (candidateResidual <= options.GradientTolerance)
                {
                    x = candidate;
                    result.Converged = true;
                    result.ResidualNorm = candidateResidual;
                    break;
                }

                if (H(j + 1, j) <= options.Epsilon)
                {
                    x = candidate;
                    break;
                }

                if (j + 1 == innerLimit)
                {
                    x = candidate;
                }
            }

            if (usedInner == 0 || result.Converged)
            {
                break;
            }
        }

        result.Solution = x;
        result.Iterations = totalIterations;
        result.ResidualNorm =
            optimization_detail::Norm(
                optimization_detail::Subtract(
                    b,
                    optimization_detail::MatrixVector(A, x)));
        result.Converged =
            result.Converged ||
            result.ResidualNorm <= options.GradientTolerance;
        return result;
    }

    /// Input: Hessian `H`, linear term `g`, equality matrix `A`, and target `b`.
    /// Output: primal variables and Lagrange multipliers solving the KKT system.
    /// Task: solve equality constrained quadratic objectives:
    /// `min 0.5*x^T H x + g^T x` subject to `A*x = b`.
    template<FloatingPoint T>
    [[nodiscard]]
    EqualityConstrainedQPResult<T> LagrangeMultipliers(
        const DynamicMatrix<T>& H,
        const std::vector<T>& g,
        const DynamicMatrix<T>& A,
        const std::vector<T>& b)
    {
        if (H.Rows() != H.Columns())
        {
            throw std::invalid_argument("LagrangeMultipliers failed: Hessian must be square.");
        }
        if (H.Rows() != g.size())
        {
            throw std::invalid_argument("LagrangeMultipliers failed: Hessian rows must match gradient size.");
        }
        if (A.Columns() != g.size())
        {
            throw std::invalid_argument("LagrangeMultipliers failed: constraint columns must match variable count.");
        }
        if (A.Rows() != b.size())
        {
            throw std::invalid_argument("LagrangeMultipliers failed: constraint rows must match b size.");
        }

        const std::size_t variableCount =
            g.size();

        const std::size_t constraintCount =
            b.size();

        DynamicMatrix<T> kkt(
            variableCount + constraintCount,
            variableCount + constraintCount,
            T(0));

        std::vector<T> rhs(variableCount + constraintCount, T(0));

        for (std::size_t row = 0; row < variableCount; ++row)
        {
            rhs[row] = -g[row];
            for (std::size_t col = 0; col < variableCount; ++col)
            {
                kkt(row, col) = H(row, col);
            }
            for (std::size_t constraint = 0; constraint < constraintCount; ++constraint)
            {
                kkt(row, variableCount + constraint) = A(constraint, row);
            }
        }

        for (std::size_t constraint = 0; constraint < constraintCount; ++constraint)
        {
            rhs[variableCount + constraint] = b[constraint];
            for (std::size_t col = 0; col < variableCount; ++col)
            {
                kkt(variableCount + constraint, col) = A(constraint, col);
            }
        }

        const std::vector<T> solution =
            LinearSolve(kkt, rhs);

        EqualityConstrainedQPResult<T> result;
        result.Primal.assign(
            solution.begin(),
            solution.begin() + static_cast<std::ptrdiff_t>(variableCount));

        result.Multipliers.assign(
            solution.begin() + static_cast<std::ptrdiff_t>(variableCount),
            solution.end());

        result.ObjectiveValue =
            optimization_detail::QuadraticObjectiveValue(
                H,
                g,
                result.Primal);

        result.KKTResidualNorm =
            optimization_detail::Norm(
                optimization_detail::Subtract(
                    optimization_detail::MatrixVector(kkt, solution),
                    rhs));

        return result;
    }

    /// Input: Hessian, linear term, equality constraints, and equality targets.
    /// Output: equality-constrained quadratic-program solution.
    /// Task: provide the named QP entry point for convex equality-constrained
    /// problems. Inequality/active-set handling belongs to a later optimization
    /// layer; this function intentionally solves the complete equality KKT
    /// system exactly through the dynamic linear algebra module.
    template<FloatingPoint T>
    [[nodiscard]]
    EqualityConstrainedQPResult<T> QuadraticProgramming(
        const DynamicMatrix<T>& H,
        const std::vector<T>& g,
        const DynamicMatrix<T>& A,
        const std::vector<T>& b)
    {
        return LagrangeMultipliers(
            H,
            g,
            A,
            b);
    }
}
