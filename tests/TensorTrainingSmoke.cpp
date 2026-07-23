#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <limits>

import Kairo.Foundation.Math.Tensor;
import Kairo.Foundation.Math.TensorAutograd;
import Kairo.Foundation.Math.TensorTraining;

namespace
{
    using namespace kairo::foundation::math;

    void TrainStep(Variable& parameter, TensorOptimizer& optimizer, TrainingRandom& random)
    {
        parameter.ZeroGradient();
        const float target = random.Uniform() * 0.5f + 0.5f;
        MeanSquaredLoss(parameter, Tensor<float>({ 1 }, target)).Backward();
        std::array<Variable*, 1> parameters{ &parameter };
        optimizer.Step(parameters);
    }
}

int main()
{
    using namespace kairo::foundation::math;
    const TensorOptimizerKind kinds[] = {
        TensorOptimizerKind::SGD,
        TensorOptimizerKind::Momentum,
        TensorOptimizerKind::Nesterov,
        TensorOptimizerKind::RMSProp,
        TensorOptimizerKind::Adam,
        TensorOptimizerKind::AdamW
    };
    for (const TensorOptimizerKind kind : kinds)
    {
        Variable parameter(Tensor<float>({ 1 }, 2.0f), true);
        TensorOptimizerConfig config;
        config.kind = kind;
        config.schedule.baseRate = 0.05f;
        config.weightDecay = kind == TensorOptimizerKind::AdamW ? 0.01f : 0.0f;
        config.maximumGradientNorm = 0.5f;
        TensorOptimizer optimizer(config);
        TrainingRandom random(17);
        TrainStep(parameter, optimizer, random);
        if (!(parameter.Value()[0] < 2.0f) || optimizer.CompletedSteps() != 1) return 1;
    }

    LearningRateSchedule schedule{
        .kind = LearningRateScheduleKind::CosineDecay,
        .baseRate = 0.1f,
        .minimumRate = 0.01f,
        .decayFactor = 0.1f,
        .decaySteps = 10,
        .warmupSteps = 2
    };
    if (std::abs(schedule.Rate(0) - 0.05f) > 1e-6f
        || std::abs(schedule.Rate(1) - 0.1f) > 1e-6f
        || std::abs(schedule.Rate(12) - 0.01f) > 1e-6f) return 2;

    TensorOptimizerConfig config;
    config.kind = TensorOptimizerKind::AdamW;
    config.schedule = schedule;
    config.weightDecay = 0.02f;
    config.maximumGradientNorm = 1.0f;

    Variable uninterrupted(Tensor<float>({ 1 }, 1.75f), true);
    TensorOptimizer uninterruptedOptimizer(config);
    TrainingRandom uninterruptedRandom(0x12345678);
    for (std::size_t step = 0; step < 40; ++step)
        TrainStep(uninterrupted, uninterruptedOptimizer, uninterruptedRandom);

    Variable interrupted(Tensor<float>({ 1 }, 1.75f), true);
    TensorOptimizer interruptedOptimizer(config);
    TrainingRandom interruptedRandom(0x12345678);
    for (std::size_t step = 0; step < 13; ++step)
        TrainStep(interrupted, interruptedOptimizer, interruptedRandom);

    const std::filesystem::path checkpoint =
        std::filesystem::temp_directory_path() / "kairo-tensor-training-checkpoint.bin";
    std::filesystem::remove(checkpoint);
    std::array<Variable*, 1> interruptedParameters{ &interrupted };
    TensorTrainingCheckpoint::Save(
        checkpoint, interruptedParameters, interruptedOptimizer, interruptedRandom);

    Variable resumed(Tensor<float>({ 1 }, -99.0f), true);
    TensorOptimizer resumedOptimizer;
    TrainingRandom resumedRandom(1);
    std::array<Variable*, 1> resumedParameters{ &resumed };
    TensorTrainingCheckpoint::Load(
        checkpoint, resumedParameters, resumedOptimizer, resumedRandom);
    for (std::size_t step = 13; step < 40; ++step)
        TrainStep(resumed, resumedOptimizer, resumedRandom);

    std::filesystem::remove(checkpoint);
    if (resumed.Value()[0] != uninterrupted.Value()[0]
        || resumedOptimizer.CompletedSteps() != uninterruptedOptimizer.CompletedSteps()
        || resumedRandom.State() != uninterruptedRandom.State()) return 3;
    const TensorOptimizerState resumedState = resumedOptimizer.State();
    const TensorOptimizerState uninterruptedState = uninterruptedOptimizer.State();
    if (resumedState.firstMoments[0][0] != uninterruptedState.firstMoments[0][0]
        || resumedState.secondMoments[0][0] != uninterruptedState.secondMoments[0][0])
        return 4;

    TensorOptimizerConfig scaledConfig;
    scaledConfig.kind = TensorOptimizerKind::SGD;
    scaledConfig.schedule.baseRate = 0.1f;
    Variable regular(Tensor<float>({ 1 }, 2.0f), true);
    Variable scaled(Tensor<float>({ 1 }, 2.0f), true);
    TensorOptimizer regularOptimizer(scaledConfig);
    TensorOptimizer scaledOptimizer(scaledConfig);
    MeanSquaredLoss(regular, Tensor<float>({ 1 }, 0.5f)).Backward();
    std::array<Variable*, 1> regularParameters{ &regular };
    regularOptimizer.Step(regularParameters);
    const Variable scaledLoss = MeanSquaredLoss(scaled, Tensor<float>({ 1 }, 0.5f));
    DynamicLossScaler scaler(1'024.0f, 2.0f, 0.5f, 2);
    scaler.Backward(scaledLoss);
    std::array<Variable*, 1> scaledParameters{ &scaled };
    if (!scaler.Step(scaledOptimizer, scaledParameters)
        || scaled.Value()[0] != regular.Value()[0]) return 5;

    Variable overflowing(
        Tensor<float>({ 1 }, std::numeric_limits<float>::max()), true);
    const Variable overflowLoss =
        MeanSquaredLoss(overflowing, Tensor<float>({ 1 }, 0.0f));
    scaler.Backward(overflowLoss);
    std::array<Variable*, 1> overflowParameters{ &overflowing };
    const float beforeOverflow = overflowing.Value()[0];
    if (scaler.Step(scaledOptimizer, overflowParameters)
        || overflowing.Value()[0] != beforeOverflow || scaler.Scale() != 512.0f)
        return 6;
    return 0;
}
