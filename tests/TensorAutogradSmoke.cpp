#include <cmath>
#include <cstddef>

import Kairo.Foundation.Math.Tensor;
import Kairo.Foundation.Math.TensorAutograd;

int main()
{
    using namespace kairo::foundation::math;
    const Tensor<float> gradientInputs({ 2, 2 }, { 1, 2, -1, 3 });
    const Tensor<float> gradientTargets({ 2, 1 }, { 0.5f, -0.25f });
    Tensor<float> baseWeights({ 2, 1 }, { 0.2f, -0.4f });
    Variable gradientWeights(baseWeights, true);
    const Variable gradientPrediction = AutogradMatMul(Variable(gradientInputs), gradientWeights);
    const Variable gradientLoss = MeanSquaredLoss(gradientPrediction, gradientTargets);
    gradientLoss.Backward();
    constexpr float epsilon = 1e-3f;
    for (std::size_t index = 0; index < baseWeights.Size(); ++index)
    {
        Tensor<float> positive = baseWeights.Contiguous();
        Tensor<float> negative = baseWeights.Contiguous();
        positive[index] += epsilon;
        negative[index] -= epsilon;
        const float positiveLoss = MeanSquaredError(MatMul(gradientInputs, positive), gradientTargets);
        const float negativeLoss = MeanSquaredError(MatMul(gradientInputs, negative), gradientTargets);
        const float numerical = (positiveLoss - negativeLoss) / (2.0f * epsilon);
        if (std::abs(numerical - gradientWeights.Gradient()[index]) > 2e-4f) return 4;
    }

    const Tensor<float> inputs({ 4, 2 }, {
        0, 0,
        0, 1,
        1, 0,
        1, 1
    });
    const Tensor<float> labels({ 4, 2 }, {
        1, 0,
        0, 1,
        0, 1,
        1, 0
    });
    Variable firstWeight(Tensor<float>({ 2, 4 }, {
        0.2f, -0.3f, 0.4f, 0.1f,
        -0.4f, 0.2f, 0.1f, 0.3f
    }), true);
    Variable firstBias(Tensor<float>({ 4 }, 0.0f), true);
    Variable secondWeight(Tensor<float>({ 4, 2 }, {
        0.3f, -0.2f,
        -0.1f, 0.4f,
        0.2f, -0.3f,
        -0.4f, 0.1f
    }), true);
    Variable secondBias(Tensor<float>({ 2 }, 0.0f), true);
    const Variable inputVariable(inputs, false);

    float initialLoss = 0.0f;
    float finalLoss = 0.0f;
    for (std::size_t step = 0; step < 2'000; ++step)
    {
        firstWeight.ZeroGradient();
        firstBias.ZeroGradient();
        secondWeight.ZeroGradient();
        secondBias.ZeroGradient();
        const Variable hidden = AutogradReLU(AddRowBias(AutogradMatMul(inputVariable, firstWeight), firstBias));
        const Variable logits = AddRowBias(AutogradMatMul(hidden, secondWeight), secondBias);
        const Variable loss = SoftmaxCrossEntropyLoss(logits, labels);
        if (step == 0) initialLoss = loss.Value()[0];
        finalLoss = loss.Value()[0];
        loss.Backward();
        firstWeight.ApplySGD(0.2f);
        firstBias.ApplySGD(0.2f);
        secondWeight.ApplySGD(0.2f);
        secondBias.ApplySGD(0.2f);
    }

    const Variable hidden = AutogradReLU(AddRowBias(AutogradMatMul(inputVariable, firstWeight), firstBias));
    const Variable logits = AddRowBias(AutogradMatMul(hidden, secondWeight), secondBias);
    if (!(finalLoss < initialLoss && finalLoss < 0.05f)) return 1;
    for (std::size_t row = 0; row < 4; ++row)
    {
        const std::size_t predicted = logits.Value()(row, 1) > logits.Value()(row, 0) ? 1 : 0;
        const std::size_t expected = labels(row, 1) > labels(row, 0) ? 1 : 0;
        if (predicted != expected) return 2;
    }

    Variable scalar(Tensor<float>({ 1 }, 3.0f), true);
    const Variable squared = Multiply(scalar, scalar);
    squared.Backward();
    return std::abs(scalar.Gradient()[0] - 6.0f) < 1e-5f ? 0 : 3;
}
