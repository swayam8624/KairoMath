module;

#include <concepts>
#include <cstddef>
#include <stdexcept>

export module Kairo.Foundation.Math.TensorExecution;

import Kairo.Foundation.Math.Tensor;
import Kairo.Scheduler;

export namespace kairo::foundation::math
{
    /// Runs elementwise addition over scheduler-owned disjoint output ranges.
    /// Tensor views are read logically; the returned Tensor is contiguous.
    template<typename T>
        requires std::floating_point<T>
    [[nodiscard]]
    Tensor<T> ParallelAdd(
        kairo::scheduler::Scheduler& scheduler,
        const Tensor<T>& lhs,
        const Tensor<T>& rhs,
        kairo::scheduler::ExecutionPolicy policy = kairo::scheduler::ExecutionPolicy::Parallel)
    {
        if (lhs.Rank() != rhs.Rank())
        {
            throw std::invalid_argument("ParallelAdd requires equal tensor shapes.");
        }
        for (std::size_t axis = 0; axis < lhs.Rank(); ++axis)
        {
            if (lhs.Dim(axis) != rhs.Dim(axis))
            {
                throw std::invalid_argument("ParallelAdd requires equal tensor shapes.");
            }
        }
        Tensor<T> result(lhs.GetShape());
        scheduler.For(lhs.Size(), [&](kairo::scheduler::Range range)
        {
            for (std::size_t index = range.begin; index < range.end; ++index)
            {
                result[index] = lhs[index] + rhs[index];
            }
        }, policy);
        return result;
    }

    /// Parallelizes rank-2 matrix multiplication by independent result rows.
    /// The inner loop preserves the scalar Tensor kernel's row-major locality.
    template<typename T>
        requires std::floating_point<T>
    [[nodiscard]]
    Tensor<T> ParallelMatMul(
        kairo::scheduler::Scheduler& scheduler,
        const Tensor<T>& lhs,
        const Tensor<T>& rhs,
        kairo::scheduler::ExecutionPolicy policy = kairo::scheduler::ExecutionPolicy::Parallel)
    {
        if (lhs.Rank() != 2 || rhs.Rank() != 2 || lhs.Dim(1) != rhs.Dim(0))
        {
            throw std::invalid_argument("ParallelMatMul expects [M,K] x [K,N] tensors.");
        }
        Tensor<T> result({ lhs.Dim(0), rhs.Dim(1) }, T(0));
        scheduler.For(lhs.Dim(0), [&](kairo::scheduler::Range rows)
        {
            for (std::size_t row = rows.begin; row < rows.end; ++row)
            {
                for (std::size_t inner = 0; inner < lhs.Dim(1); ++inner)
                {
                    const T factor = lhs(row, inner);
                    for (std::size_t column = 0; column < rhs.Dim(1); ++column)
                    {
                        result(row, column) += factor * rhs(inner, column);
                    }
                }
            }
        }, policy);
        return result;
    }
}
