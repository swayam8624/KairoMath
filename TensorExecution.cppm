module;

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <mutex>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

export module Kairo.Foundation.Math.TensorExecution;

import Kairo.Foundation.Math.Tensor;
import Kairo.Scheduler;
import Kairo.SIMD;

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
            if constexpr (std::same_as<T, float>)
            {
                if (lhs.IsContiguous() && rhs.IsContiguous())
                {
                    kairo::simd::Add(
                        std::span<float>(result.Data() + range.begin, range.Size()),
                        std::span<const float>(lhs.Data() + range.begin, range.Size()),
                        std::span<const float>(rhs.Data() + range.begin, range.Size()));
                    return;
                }
            }
            for (std::size_t index = range.begin; index < range.end; ++index)
                result[index] = lhs[index] + rhs[index];
        }, policy);
        result.SetBackend(TensorBackend::SIMD);
        return result;
    }

    template<typename T>
        requires std::floating_point<T>
    [[nodiscard]]
    Tensor<T> ParallelReLU(
        kairo::scheduler::Scheduler& scheduler,
        const Tensor<T>& input,
        kairo::scheduler::ExecutionPolicy policy = kairo::scheduler::ExecutionPolicy::Parallel)
    {
        Tensor<T> result(input.GetShape());
        scheduler.For(input.Size(), [&](kairo::scheduler::Range range)
        {
            if constexpr (std::same_as<T, float>)
            {
                if (input.IsContiguous())
                {
                    kairo::simd::ReLU(
                        std::span<float>(result.Data() + range.begin, range.Size()),
                        std::span<const float>(input.Data() + range.begin, range.Size()));
                    return;
                }
            }
            for (std::size_t index = range.begin; index < range.end; ++index)
                result[index] = std::max(T(0), input[index]);
        }, policy);
        result.SetBackend(TensorBackend::SIMD);
        return result;
    }

    /// Deterministic parallel reduction: ranges compute independently and the
    /// caller combines partials in ascending source-offset order.
    template<typename T>
        requires std::floating_point<T>
    [[nodiscard]]
    T ParallelSum(
        kairo::scheduler::Scheduler& scheduler,
        const Tensor<T>& input,
        kairo::scheduler::ExecutionPolicy policy = kairo::scheduler::ExecutionPolicy::Parallel)
    {
        std::vector<std::pair<std::size_t, T>> partials;
        std::mutex partialMutex;
        scheduler.For(input.Size(), [&](kairo::scheduler::Range range)
        {
            T sum = T(0);
            if constexpr (std::same_as<T, float>)
            {
                if (input.IsContiguous())
                    sum = kairo::simd::Sum(
                        std::span<const float>(input.Data() + range.begin, range.Size()));
                else
                    for (std::size_t index = range.begin; index < range.end; ++index)
                        sum += input[index];
            }
            else
                for (std::size_t index = range.begin; index < range.end; ++index)
                    sum += input[index];
            std::scoped_lock lock(partialMutex);
            partials.emplace_back(range.begin, sum);
        }, policy);
        std::sort(partials.begin(), partials.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
        T sum = T(0);
        for (const auto& partial : partials) sum += partial.second;
        return sum;
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
        const Tensor<T> left = lhs.Contiguous();
        const Tensor<T> right = rhs.Contiguous();
        Tensor<T> packedRight({ rhs.Dim(1), rhs.Dim(0) }, T(0));
        for (std::size_t inner = 0; inner < rhs.Dim(0); ++inner)
            for (std::size_t column = 0; column < rhs.Dim(1); ++column)
                packedRight(column, inner) = right(inner, column);
        Tensor<T> result({ lhs.Dim(0), rhs.Dim(1) }, T(0));
        scheduler.For(lhs.Dim(0), [&](kairo::scheduler::Range rows)
        {
            for (std::size_t row = rows.begin; row < rows.end; ++row)
                for (std::size_t column = 0; column < rhs.Dim(1); ++column)
                    if constexpr (std::same_as<T, float>)
                        result(row, column) = kairo::simd::Dot(
                            std::span<const float>(left.Data() + row * lhs.Dim(1), lhs.Dim(1)),
                            std::span<const float>(
                                packedRight.Data() + column * lhs.Dim(1), lhs.Dim(1)));
                    else
                        for (std::size_t inner = 0; inner < lhs.Dim(1); ++inner)
                            result(row, column) += left(row, inner) * packedRight(column, inner);
        }, policy);
        result.SetBackend(TensorBackend::SIMD);
        return result;
    }

    template<typename T>
        requires std::floating_point<T>
    [[nodiscard]]
    Tensor<T> ParallelBatchedMatMul(
        kairo::scheduler::Scheduler& scheduler,
        const Tensor<T>& lhs,
        const Tensor<T>& rhs,
        kairo::scheduler::ExecutionPolicy policy = kairo::scheduler::ExecutionPolicy::Parallel)
    {
        if (lhs.Rank() < 3 || rhs.Rank() != lhs.Rank()
            || lhs.Dim(lhs.Rank() - 1) != rhs.Dim(rhs.Rank() - 2))
            throw std::invalid_argument("ParallelBatchedMatMul expects [...,M,K] x [...,K,N].");
        for (std::size_t axis = 0; axis + 2 < lhs.Rank(); ++axis)
            if (lhs.Dim(axis) != rhs.Dim(axis))
                throw std::invalid_argument("ParallelBatchedMatMul batch dimensions must match.");
        const Tensor<T> left = lhs.Contiguous();
        const Tensor<T> right = rhs.Contiguous();
        const std::size_t rows = lhs.Dim(lhs.Rank() - 2);
        const std::size_t inner = lhs.Dim(lhs.Rank() - 1);
        const std::size_t columns = rhs.Dim(rhs.Rank() - 1);
        const std::size_t batches = lhs.Size() / (rows * inner);
        typename Tensor<T>::Shape outputShape = lhs.GetShape();
        outputShape.back() = columns;
        Tensor<T> output(outputShape, T(0));
        Tensor<T> packed({ batches, columns, inner }, T(0));
        for (std::size_t batch = 0; batch < batches; ++batch)
            for (std::size_t k = 0; k < inner; ++k)
                for (std::size_t column = 0; column < columns; ++column)
                    packed[batch * columns * inner + column * inner + k] =
                        right[batch * inner * columns + k * columns + column];
        scheduler.For(batches * rows, [&](kairo::scheduler::Range range)
        {
            for (std::size_t work = range.begin; work < range.end; ++work)
            {
                const std::size_t batch = work / rows;
                const std::size_t row = work % rows;
                for (std::size_t column = 0; column < columns; ++column)
                    if constexpr (std::same_as<T, float>)
                        output[batch * rows * columns + row * columns + column] =
                            kairo::simd::Dot(
                                std::span<const float>(
                                    left.Data() + batch * rows * inner + row * inner, inner),
                                std::span<const float>(
                                    packed.Data() + batch * columns * inner + column * inner, inner));
                    else
                        for (std::size_t k = 0; k < inner; ++k)
                            output[batch * rows * columns + row * columns + column]
                                += left[batch * rows * inner + row * inner + k]
                                * packed[batch * columns * inner + column * inner + k];
            }
        }, policy);
        output.SetBackend(TensorBackend::SIMD);
        return output;
    }

    template<typename T>
        requires std::floating_point<T>
    [[nodiscard]]
    Tensor<T> ParallelConv2DValidNHWC(
        kairo::scheduler::Scheduler& scheduler,
        const Tensor<T>& input,
        const Tensor<T>& filters,
        std::size_t strideHeight = 1,
        std::size_t strideWidth = 1,
        kairo::scheduler::ExecutionPolicy policy = kairo::scheduler::ExecutionPolicy::Parallel)
    {
        if (input.Rank() != 4 || filters.Rank() != 4 || strideHeight == 0 || strideWidth == 0
            || input.Dim(3) != filters.Dim(3) || input.Dim(1) < filters.Dim(1)
            || input.Dim(2) < filters.Dim(2))
            throw std::invalid_argument("ParallelConv2DValidNHWC received incompatible tensors.");
        const Tensor<T> activations = input.Contiguous();
        const Tensor<T> kernels = filters.Contiguous();
        const std::size_t outputHeight =
            (input.Dim(1) - filters.Dim(1)) / strideHeight + 1;
        const std::size_t outputWidth =
            (input.Dim(2) - filters.Dim(2)) / strideWidth + 1;
        const std::size_t outputChannels = filters.Dim(0);
        const std::size_t channels = input.Dim(3);
        Tensor<T> output({
            input.Dim(0), outputHeight, outputWidth, outputChannels
        }, T(0));
        scheduler.For(output.Size(), [&](kairo::scheduler::Range range)
        {
            for (std::size_t work = range.begin; work < range.end; ++work)
            {
                std::size_t coordinate = work;
                const std::size_t outputChannel = coordinate % outputChannels;
                coordinate /= outputChannels;
                const std::size_t outputX = coordinate % outputWidth;
                coordinate /= outputWidth;
                const std::size_t outputY = coordinate % outputHeight;
                const std::size_t batch = coordinate / outputHeight;
                T sum = T(0);
                for (std::size_t kernelY = 0; kernelY < filters.Dim(1); ++kernelY)
                    for (std::size_t kernelX = 0; kernelX < filters.Dim(2); ++kernelX)
                    {
                        const std::size_t inputOffset =
                            ((batch * input.Dim(1) + outputY * strideHeight + kernelY)
                                * input.Dim(2) + outputX * strideWidth + kernelX) * channels;
                        const std::size_t filterOffset =
                            ((outputChannel * filters.Dim(1) + kernelY)
                                * filters.Dim(2) + kernelX) * channels;
                        if constexpr (std::same_as<T, float>)
                            sum += kairo::simd::Dot(
                                std::span<const float>(
                                    activations.Data() + inputOffset, channels),
                                std::span<const float>(
                                    kernels.Data() + filterOffset, channels));
                        else
                            for (std::size_t channel = 0; channel < channels; ++channel)
                                sum += activations[inputOffset + channel]
                                    * kernels[filterOffset + channel];
                    }
                output[work] = sum;
            }
        }, policy);
        output.SetBackend(TensorBackend::SIMD);
        return output;
    }

    class TensorExecutionContext final
    {
    public:
        explicit TensorExecutionContext(kairo::scheduler::SchedulerConfig config = {})
            : scheduler_(config) {}

        [[nodiscard]] Tensor<float> Add(
            const Tensor<float>& lhs, const Tensor<float>& rhs)
        {
            return ShouldAccelerate(lhs.Size())
                ? ParallelAdd(scheduler_, lhs, rhs) : lhs + rhs;
        }

        [[nodiscard]] Tensor<float> ReLU(const Tensor<float>& input)
        {
            return ShouldAccelerate(input.Size())
                ? ParallelReLU(scheduler_, input) : kairo::foundation::math::ReLU(input);
        }

        [[nodiscard]] Tensor<float> MatMul(
            const Tensor<float>& lhs, const Tensor<float>& rhs)
        {
            return ShouldAccelerate(lhs.Dim(0) * rhs.Dim(1))
                ? ParallelMatMul(scheduler_, lhs, rhs) : kairo::foundation::math::MatMul(lhs, rhs);
        }

        [[nodiscard]] kairo::simd::CpuFeature Feature() const noexcept
        {
            return kairo::simd::DetectedFeature();
        }

    private:
        kairo::scheduler::Scheduler scheduler_;
        static bool ShouldAccelerate(std::size_t work) noexcept { return work >= 256; }
    };
}
