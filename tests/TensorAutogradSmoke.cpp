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

    const Tensor<float> convolutionInput({ 1, 3, 3, 1 }, {
        0.2f, -0.1f, 0.3f,
        0.7f, 0.5f, -0.2f,
        -0.4f, 0.6f, 0.8f
    });
    Tensor<float> baseFilter({ 1, 2, 2, 1 }, { 0.3f, -0.2f, 0.4f, 0.1f });
    const Tensor<float> convolutionTarget({ 1, 2, 2, 1 }, {
        0.1f, -0.2f,
        0.3f, 0.4f
    });
    Variable gradientFilter(baseFilter, true);
    const Variable convolution = AutogradConv2DValidNHWC(
        Variable(convolutionInput), gradientFilter, 1, 1);
    MeanSquaredLoss(convolution, convolutionTarget).Backward();
    for (std::size_t index = 0; index < baseFilter.Size(); ++index)
    {
        Tensor<float> positive = baseFilter.Contiguous();
        Tensor<float> negative = baseFilter.Contiguous();
        positive[index] += epsilon;
        negative[index] -= epsilon;
        const float positiveLoss = MeanSquaredError(
            Conv2DValidNHWC(convolutionInput, positive, 1, 1), convolutionTarget);
        const float negativeLoss = MeanSquaredError(
            Conv2DValidNHWC(convolutionInput, negative, 1, 1), convolutionTarget);
        const float numerical = (positiveLoss - negativeLoss) / (2.0f * epsilon);
        if (std::abs(numerical - gradientFilter.Gradient()[index]) > 3e-4f) return 5;
    }

    Variable poolingInput(Tensor<float>({ 1, 2, 2, 1 }, { 1, 3, 2, 4 }), true);
    const Variable pooling = AutogradMaxPool2DValidNHWC(poolingInput, 2, 2, 2, 2);
    MeanSquaredLoss(pooling, Tensor<float>({ 1, 1, 1, 1 }, 0.0f)).Backward();
    for (std::size_t index = 0; index < poolingInput.Gradient().Size(); ++index)
    {
        const float expected = index == 3 ? 8.0f : 0.0f;
        if (std::abs(poolingInput.Gradient()[index] - expected) > 1e-6f) return 6;
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

    const Tensor<float> imageInputs({ 4, 3, 3, 1 }, {
        1, 1, 1,  0, 0, 0,  0, 0, 0,
        0, 0, 0,  0, 0, 0,  1, 1, 1,
        1, 0, 0,  1, 0, 0,  1, 0, 0,
        0, 0, 1,  0, 0, 1,  0, 0, 1
    });
    const Tensor<float> imageLabels({ 4, 2 }, {
        1, 0,
        1, 0,
        0, 1,
        0, 1
    });
    Variable cnnFilters(Tensor<float>({ 2, 2, 2, 1 }, {
        0.15f, -0.05f, 0.10f, -0.10f,
        -0.10f, 0.10f, -0.05f, 0.15f
    }), true);
    Variable cnnWeights(Tensor<float>({ 8, 2 }, {
        0.10f, -0.10f, -0.05f, 0.08f,
        0.07f, -0.04f, -0.09f, 0.06f,
        -0.06f, 0.09f, 0.04f, -0.07f,
        0.08f, -0.05f, -0.10f, 0.10f
    }), true);
    Variable cnnBias(Tensor<float>({ 2 }, 0.0f), true);
    const Variable imageInputVariable(imageInputs);
    float initialCnnLoss = 0.0f;
    float finalCnnLoss = 0.0f;
    for (std::size_t step = 0; step < 1'500; ++step)
    {
        cnnFilters.ZeroGradient();
        cnnWeights.ZeroGradient();
        cnnBias.ZeroGradient();
        const Variable features =
            AutogradReLU(AutogradConv2DValidNHWC(imageInputVariable, cnnFilters, 1, 1));
        const Variable flattened = AutogradReshape(features, { 4, 8 });
        const Variable cnnLogits = AddRowBias(
            AutogradMatMul(flattened, cnnWeights), cnnBias);
        const Variable cnnLoss = SoftmaxCrossEntropyLoss(cnnLogits, imageLabels);
        if (step == 0) initialCnnLoss = cnnLoss.Value()[0];
        finalCnnLoss = cnnLoss.Value()[0];
        cnnLoss.Backward();
        cnnFilters.ApplySGD(0.15f);
        cnnWeights.ApplySGD(0.15f);
        cnnBias.ApplySGD(0.15f);
    }
    const Variable finalFeatures =
        AutogradReLU(AutogradConv2DValidNHWC(imageInputVariable, cnnFilters, 1, 1));
    const Variable finalCnnLogits = AddRowBias(
        AutogradMatMul(AutogradReshape(finalFeatures, { 4, 8 }), cnnWeights), cnnBias);
    if (!(finalCnnLoss < initialCnnLoss && finalCnnLoss < 0.08f)) return 7;
    for (std::size_t row = 0; row < 4; ++row)
    {
        const std::size_t predicted =
            finalCnnLogits.Value()(row, 1) > finalCnnLogits.Value()(row, 0) ? 1 : 0;
        const std::size_t expected = imageLabels(row, 1) > imageLabels(row, 0) ? 1 : 0;
        if (predicted != expected) return 8;
    }

    Variable scalar(Tensor<float>({ 1 }, 3.0f), true);
    const Variable squared = Multiply(scalar, scalar);
    squared.Backward();
    return std::abs(scalar.Gradient()[0] - 6.0f) < 1e-5f ? 0 : 3;
}
