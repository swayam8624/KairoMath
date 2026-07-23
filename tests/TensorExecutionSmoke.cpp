#include <cmath>
#include <cstddef>

import Kairo.Foundation.Math.Tensor;
import Kairo.Foundation.Math.TensorExecution;
import Kairo.Scheduler;
import Kairo.SIMD;

namespace
{
    template<typename T>
    bool Equal(
        const kairo::foundation::math::Tensor<T>& lhs,
        const kairo::foundation::math::Tensor<T>& rhs,
        T tolerance = T(1e-5))
    {
        if (lhs.Rank() != rhs.Rank()) return false;
        for (std::size_t axis = 0; axis < lhs.Rank(); ++axis)
            if (lhs.Dim(axis) != rhs.Dim(axis)) return false;
        for (std::size_t index = 0; index < lhs.Size(); ++index)
            if (std::abs(lhs[index] - rhs[index]) > tolerance) return false;
        return true;
    }
}

int main()
{
    using namespace kairo::foundation::math;
    kairo::scheduler::Scheduler scheduler({
        .workerCount = 3,
        .defaultMinChunkSize = 2
    });
    const Tensor<float> lhs({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });
    const Tensor<float> rhs({ 2, 3 }, { 6, 5, 4, 3, 2, 1 });
    const auto added = ParallelAdd(scheduler, lhs, rhs);
    if (!Equal(added, lhs + rhs) || added.Backend() != TensorBackend::SIMD) return 1;
    if (ParallelSum(scheduler, lhs) != lhs.Sum()) return 2;

    const Tensor<float> right({ 3, 2 }, { 1, 0, 0, 1, 1, 1 });
    const auto product = ParallelMatMul(scheduler, lhs, right);
    if (!Equal(product, MatMul(lhs, right))) return 3;

    const Tensor<float> batchedLeft({ 2, 2, 3 }, {
        1, 2, 3, 4, 5, 6,
        1, 0, 2, 0, 1, 3
    });
    const Tensor<float> batchedRight({ 2, 3, 2 }, {
        1, 0, 0, 1, 1, 1,
        2, 1, 1, 0, 0, 1
    });
    if (!Equal(
        ParallelBatchedMatMul(scheduler, batchedLeft, batchedRight),
        BatchedMatMul(batchedLeft, batchedRight))) return 4;

    const Tensor<float> convolutionInput({ 1, 3, 3, 2 }, {
        1, 2, 3, 4, 5, 6,
        7, 8, 9, 10, 11, 12,
        13, 14, 15, 16, 17, 18
    });
    const Tensor<float> filters({ 2, 2, 2, 2 }, {
        1, 0, 0, 1, 1, 1, -1, 0,
        0, 1, 1, 0, -1, 1, 0, -1
    });
    if (!Equal(
        ParallelConv2DValidNHWC(scheduler, convolutionInput, filters),
        Conv2DValidNHWC(convolutionInput, filters))) return 5;

    Tensor<float> activation({ 1, 512 }, -1.0f);
    for (std::size_t index = 256; index < activation.Size(); ++index)
        activation[index] = static_cast<float>(index);
    TensorExecutionContext context({
        .workerCount = 2,
        .defaultMinChunkSize = 32
    });
    const Tensor<float> activated = context.ReLU(activation);
    if (!Equal(activated, ReLU(activation))
        || activated.Backend() != TensorBackend::SIMD
        || context.Feature() != kairo::simd::DetectedFeature()) return 6;
    return 0;
}
