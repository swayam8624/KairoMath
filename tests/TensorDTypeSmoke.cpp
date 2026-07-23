#include <cmath>
#include <cstddef>
#include <cstdint>

import Kairo.Foundation.Math.Tensor;

int main()
{
    using namespace kairo::foundation::math;
    const Tensor<float> source({ 2, 3 }, {
        1.0f, -2.0f, 3.5f,
        0.25f, 4.0f, -1.5f
    });
    const TensorFloat16 half = TensorCast<Float16>(source);
    const TensorBFloat16 brain = TensorCast<BFloat16>(source);
    const Tensor<float> halfRoundTrip = TensorCast<float>(half);
    const Tensor<float> brainRoundTrip = TensorCast<float>(brain);
    for (std::size_t index = 0; index < source.Size(); ++index)
    {
        if (std::abs(halfRoundTrip[index] - source[index]) > 0.002f) return 1;
        if (std::abs(brainRoundTrip[index] - source[index]) > 0.02f) return 2;
    }

    const TensorFloat16 lhs({ 2, 2, 3 }, {
        1, 2, 3, 4, 5, 6,
        1, 0, 2, 0, 1, 3
    });
    const TensorFloat16 rhs({ 2, 3, 2 }, {
        1, 0, 0, 1, 1, 1,
        2, 1, 1, 0, 0, 1
    });
    const TensorFloat16 product = BatchedMatMul(lhs, rhs);
    const Tensor<float> productFloat = TensorCast<float>(product);
    const float expected[] = { 4, 5, 10, 11, 2, 3, 1, 3 };
    for (std::size_t index = 0; index < productFloat.Size(); ++index)
        if (productFloat[index] != expected[index]) return 3;

    const Tensor<float> normalized = LayerNormLastDim(source);
    for (std::size_t row = 0; row < 2; ++row)
    {
        float mean = 0.0f;
        float squareMean = 0.0f;
        for (std::size_t column = 0; column < 3; ++column)
        {
            mean += normalized(row, column);
            squareMean += normalized(row, column) * normalized(row, column);
        }
        mean /= 3.0f;
        squareMean /= 3.0f;
        if (std::abs(mean) > 1e-5f || std::abs(squareMean - 1.0f) > 1e-4f) return 4;
    }

    const TensorBFloat16 rms = RMSNormLastDim(brain);
    const Tensor<float> rmsFloat = TensorCast<float>(rms);
    for (std::size_t row = 0; row < 2; ++row)
    {
        float squareMean = 0.0f;
        for (std::size_t column = 0; column < 3; ++column)
            squareMean += rmsFloat(row, column) * rmsFloat(row, column);
        if (std::abs(squareMean / 3.0f - 1.0f) > 0.02f) return 5;
    }

    TensorInt8 quantized({ 3 }, { std::int8_t{-3}, std::int8_t{0}, std::int8_t{7} });
    TensorIndex indices({ 3 }, { std::int64_t{2}, std::int64_t{0}, std::int64_t{1} });
    return quantized[2] == 7 && indices[0] == 2 ? 0 : 6;
}
