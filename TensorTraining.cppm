module;

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

export module Kairo.Foundation.Math.TensorTraining;

import Kairo.Foundation.Math.Tensor;
import Kairo.Foundation.Math.TensorAutograd;

export namespace kairo::foundation::math
{
    enum class TensorOptimizerKind : std::uint32_t
    {
        SGD,
        Momentum,
        Nesterov,
        RMSProp,
        Adam,
        AdamW
    };

    enum class LearningRateScheduleKind : std::uint32_t
    {
        Constant,
        StepDecay,
        CosineDecay
    };

    struct LearningRateSchedule final
    {
        LearningRateScheduleKind kind = LearningRateScheduleKind::Constant;
        float baseRate = 1e-3f;
        float minimumRate = 0.0f;
        float decayFactor = 0.1f;
        std::uint64_t decaySteps = 1;
        std::uint64_t warmupSteps = 0;

        [[nodiscard]] float Rate(std::uint64_t completedSteps) const
        {
            if (!(baseRate > 0.0f) || minimumRate < 0.0f || minimumRate > baseRate)
                throw std::invalid_argument("Learning-rate schedule contains invalid rates.");
            if (decaySteps == 0)
                throw std::invalid_argument("Learning-rate schedule requires non-zero decaySteps.");
            if (warmupSteps > 0 && completedSteps < warmupSteps)
                return baseRate * static_cast<float>(completedSteps + 1)
                    / static_cast<float>(warmupSteps);
            const std::uint64_t decayStep = completedSteps >= warmupSteps
                ? completedSteps - warmupSteps : 0;
            switch (kind)
            {
            case LearningRateScheduleKind::Constant:
                return baseRate;
            case LearningRateScheduleKind::StepDecay:
                return std::max(
                    minimumRate,
                    baseRate * std::pow(decayFactor,
                        static_cast<float>(decayStep / decaySteps)));
            case LearningRateScheduleKind::CosineDecay:
            {
                const float progress = std::min(
                    1.0f, static_cast<float>(decayStep) / static_cast<float>(decaySteps));
                constexpr float pi = 3.14159265358979323846f;
                return minimumRate + 0.5f * (baseRate - minimumRate)
                    * (1.0f + std::cos(pi * progress));
            }
            }
            throw std::logic_error("Unknown learning-rate schedule.");
        }
    };

    struct TensorOptimizerConfig final
    {
        TensorOptimizerKind kind = TensorOptimizerKind::AdamW;
        LearningRateSchedule schedule{};
        float momentum = 0.9f;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float epsilon = 1e-8f;
        float weightDecay = 0.0f;
        float maximumGradientNorm = 0.0f;
    };

    struct TensorOptimizerState final
    {
        TensorOptimizerConfig config;
        std::uint64_t completedSteps = 0;
        std::vector<Tensor<float>> firstMoments;
        std::vector<Tensor<float>> secondMoments;
    };

    /// Stateful optimizer for Float32 autograd Variables.
    ///
    /// Parameter order and shapes are part of the state contract and must remain
    /// stable across calls and checkpoint restoration. Weight decay is coupled
    /// for Adam and decoupled for AdamW.
    class TensorOptimizer final
    {
    public:
        explicit TensorOptimizer(TensorOptimizerConfig config = {}) : config_(config)
        {
            ValidateConfig(config_);
        }

        [[nodiscard]] std::uint64_t CompletedSteps() const noexcept { return completedSteps_; }
        [[nodiscard]] float CurrentLearningRate() const
        {
            return config_.schedule.Rate(completedSteps_);
        }
        [[nodiscard]] const TensorOptimizerConfig& Config() const noexcept { return config_; }

        /// Input: stable ordered trainable variables with accumulated gradients.
        /// Output: one optimizer update, persistent state advancement, and no
        /// implicit gradient clearing.
        /// Failure: rejects absent gradients, non-trainable variables, shape
        /// drift, non-finite norms, and invalid hyperparameters before mutation.
        void Step(std::span<Variable*> parameters, float gradientDivisor = 1.0f)
        {
            if (parameters.empty()) throw std::invalid_argument("Optimizer Step requires parameters.");
            if (!(gradientDivisor > 0.0f) || !std::isfinite(gradientDivisor))
                throw std::invalid_argument("Optimizer gradient divisor must be finite and positive.");
            EnsureState(parameters);
            float squaredNorm = 0.0f;
            for (Variable* parameter : parameters)
            {
                if (!parameter || !parameter->RequiresGradient() || !parameter->HasGradient())
                    throw std::logic_error("Optimizer parameters must be trainable and have gradients.");
                for (std::size_t index = 0; index < parameter->Gradient().Size(); ++index)
                {
                    const float gradient = parameter->Gradient()[index] / gradientDivisor;
                    squaredNorm += gradient * gradient;
                }
            }
            if (!std::isfinite(squaredNorm))
                throw std::domain_error("Optimizer gradient norm is not finite.");
            float gradientScale = 1.0f;
            if (config_.maximumGradientNorm > 0.0f)
            {
                const float norm = std::sqrt(squaredNorm);
                if (norm > config_.maximumGradientNorm)
                    gradientScale = config_.maximumGradientNorm / norm;
            }

            const float rate = CurrentLearningRate();
            const std::uint64_t nextStep = completedSteps_ + 1;
            const float bias1 = 1.0f - std::pow(config_.beta1, static_cast<float>(nextStep));
            const float bias2 = 1.0f - std::pow(config_.beta2, static_cast<float>(nextStep));
            for (std::size_t parameterIndex = 0; parameterIndex < parameters.size(); ++parameterIndex)
            {
                Variable& parameter = *parameters[parameterIndex];
                Tensor<float> update(parameter.Value().GetShape(), 0.0f);
                Tensor<float>& first = firstMoments_[parameterIndex];
                Tensor<float>& second = secondMoments_[parameterIndex];
                for (std::size_t index = 0; index < parameter.Value().Size(); ++index)
                {
                    float gradient =
                        parameter.Gradient()[index] / gradientDivisor * gradientScale;
                    if (config_.kind != TensorOptimizerKind::AdamW && config_.weightDecay != 0.0f)
                        gradient += config_.weightDecay * parameter.Value()[index];
                    switch (config_.kind)
                    {
                    case TensorOptimizerKind::SGD:
                        update[index] = rate * gradient;
                        break;
                    case TensorOptimizerKind::Momentum:
                        first[index] = config_.momentum * first[index] + gradient;
                        update[index] = rate * first[index];
                        break;
                    case TensorOptimizerKind::Nesterov:
                        first[index] = config_.momentum * first[index] + gradient;
                        update[index] = rate * (config_.momentum * first[index] + gradient);
                        break;
                    case TensorOptimizerKind::RMSProp:
                        second[index] = config_.beta2 * second[index]
                            + (1.0f - config_.beta2) * gradient * gradient;
                        update[index] = rate * gradient
                            / (std::sqrt(second[index]) + config_.epsilon);
                        break;
                    case TensorOptimizerKind::Adam:
                    case TensorOptimizerKind::AdamW:
                        first[index] = config_.beta1 * first[index]
                            + (1.0f - config_.beta1) * gradient;
                        second[index] = config_.beta2 * second[index]
                            + (1.0f - config_.beta2) * gradient * gradient;
                        update[index] = rate * (first[index] / bias1)
                            / (std::sqrt(second[index] / bias2) + config_.epsilon);
                        if (config_.kind == TensorOptimizerKind::AdamW)
                            update[index] += rate * config_.weightDecay * parameter.Value()[index];
                        break;
                    }
                }
                parameter.ApplyUpdate(update);
            }
            completedSteps_ = nextStep;
        }

        [[nodiscard]] TensorOptimizerState State() const
        {
            return {
                config_,
                completedSteps_,
                DeepCopy(firstMoments_),
                DeepCopy(secondMoments_)
            };
        }

        void Restore(const TensorOptimizerState& state, std::span<Variable*> parameters)
        {
            ValidateConfig(state.config);
            ValidateStateShapes(state, parameters);
            config_ = state.config;
            completedSteps_ = state.completedSteps;
            firstMoments_ = DeepCopy(state.firstMoments);
            secondMoments_ = DeepCopy(state.secondMoments);
        }

    private:
        TensorOptimizerConfig config_;
        std::uint64_t completedSteps_ = 0;
        std::vector<Tensor<float>> firstMoments_;
        std::vector<Tensor<float>> secondMoments_;

        static void ValidateConfig(const TensorOptimizerConfig& config)
        {
            (void)config.schedule.Rate(0);
            if (!(config.momentum >= 0.0f && config.momentum < 1.0f)
                || !(config.beta1 >= 0.0f && config.beta1 < 1.0f)
                || !(config.beta2 >= 0.0f && config.beta2 < 1.0f)
                || !(config.epsilon > 0.0f) || config.weightDecay < 0.0f
                || config.maximumGradientNorm < 0.0f)
                throw std::invalid_argument("Optimizer hyperparameters are invalid.");
        }

        static std::vector<Tensor<float>> DeepCopy(const std::vector<Tensor<float>>& source)
        {
            std::vector<Tensor<float>> copy;
            copy.reserve(source.size());
            for (const auto& tensor : source) copy.push_back(tensor.Contiguous());
            return copy;
        }

        static void ValidateStateShapes(
            const TensorOptimizerState& state, std::span<Variable*> parameters)
        {
            if (state.firstMoments.size() != parameters.size()
                || state.secondMoments.size() != parameters.size())
                throw std::invalid_argument("Optimizer state parameter count mismatch.");
            for (std::size_t parameterIndex = 0; parameterIndex < parameters.size(); ++parameterIndex)
            {
                if (!parameters[parameterIndex])
                    throw std::invalid_argument("Optimizer state contains a null parameter.");
                for (const Tensor<float>* moment : {
                    &state.firstMoments[parameterIndex], &state.secondMoments[parameterIndex] })
                {
                    if (moment->Rank() != parameters[parameterIndex]->Value().Rank())
                        throw std::invalid_argument("Optimizer state rank mismatch.");
                    for (std::size_t axis = 0; axis < moment->Rank(); ++axis)
                        if (moment->Dim(axis) != parameters[parameterIndex]->Value().Dim(axis))
                            throw std::invalid_argument("Optimizer state shape mismatch.");
                }
            }
        }

        void EnsureState(std::span<Variable*> parameters)
        {
            if (firstMoments_.empty())
            {
                firstMoments_.reserve(parameters.size());
                secondMoments_.reserve(parameters.size());
                for (Variable* parameter : parameters)
                {
                    if (!parameter) throw std::invalid_argument("Optimizer parameter cannot be null.");
                    firstMoments_.emplace_back(parameter->Value().GetShape(), 0.0f);
                    secondMoments_.emplace_back(parameter->Value().GetShape(), 0.0f);
                }
                return;
            }
            ValidateStateShapes(State(), parameters);
        }
    };

    /// Dynamic loss scaler for Float32-master mixed-precision training.
    ///
    /// `Backward` seeds a scaled reverse pass. `Step` rejects non-finite
    /// gradients without modifying parameters, backs off the scale, and clears
    /// rejected gradients. Successful steps unscale exactly once inside the
    /// optimizer and periodically grow the scale.
    class DynamicLossScaler final
    {
    public:
        explicit DynamicLossScaler(
            float initialScale = 65'536.0f,
            float growthFactor = 2.0f,
            float backoffFactor = 0.5f,
            std::uint64_t growthInterval = 2'000)
            : scale_(initialScale),
              growthFactor_(growthFactor),
              backoffFactor_(backoffFactor),
              growthInterval_(growthInterval)
        {
            if (!(scale_ >= 1.0f) || !(growthFactor_ > 1.0f)
                || !(backoffFactor_ > 0.0f && backoffFactor_ < 1.0f)
                || growthInterval_ == 0)
                throw std::invalid_argument("DynamicLossScaler configuration is invalid.");
        }

        [[nodiscard]] float Scale() const noexcept { return scale_; }
        [[nodiscard]] std::uint64_t ConsecutiveFiniteSteps() const noexcept
        {
            return finiteSteps_;
        }

        void Backward(const Variable& scalarLoss) const
        {
            scalarLoss.Backward(scale_);
        }

        [[nodiscard]] bool Step(TensorOptimizer& optimizer, std::span<Variable*> parameters)
        {
            bool finite = true;
            for (Variable* parameter : parameters)
            {
                if (!parameter || !parameter->HasGradient())
                    throw std::logic_error("Loss scaler requires parameters with gradients.");
                for (std::size_t index = 0; index < parameter->Gradient().Size(); ++index)
                    finite = finite && std::isfinite(parameter->Gradient()[index]);
            }
            if (!finite)
            {
                for (Variable* parameter : parameters) parameter->ZeroGradient();
                scale_ = std::max(1.0f, scale_ * backoffFactor_);
                finiteSteps_ = 0;
                return false;
            }
            optimizer.Step(parameters, scale_);
            ++finiteSteps_;
            if (finiteSteps_ == growthInterval_)
            {
                scale_ *= growthFactor_;
                finiteSteps_ = 0;
            }
            return true;
        }

    private:
        float scale_;
        float growthFactor_;
        float backoffFactor_;
        std::uint64_t growthInterval_;
        std::uint64_t finiteSteps_ = 0;
    };

    /// Small reproducible generator whose complete state is checkpointable.
    class TrainingRandom final
    {
    public:
        explicit TrainingRandom(std::uint64_t seed = 0x4B4149524F4D4CULL)
            : state_(seed == 0 ? 0x9E3779B97F4A7C15ULL : seed) {}

        [[nodiscard]] std::uint64_t State() const noexcept { return state_; }
        void Restore(std::uint64_t state)
        {
            if (state == 0) throw std::invalid_argument("Training RNG state cannot be zero.");
            state_ = state;
        }
        [[nodiscard]] std::uint64_t Next()
        {
            std::uint64_t value = state_;
            value ^= value >> 12;
            value ^= value << 25;
            value ^= value >> 27;
            state_ = value;
            return value * 0x2545F4914F6CDD1DULL;
        }
        [[nodiscard]] float Uniform()
        {
            return static_cast<float>(Next() >> 40) / static_cast<float>(1ULL << 24);
        }

    private:
        std::uint64_t state_;
    };

    /// Versioned Float32 training checkpoint. Save uses a sibling temporary
    /// file and atomic rename. Load validates the entire payload and parameter
    /// shapes before replacing any live state.
    class TensorTrainingCheckpoint final
    {
    public:
        static void Save(
            const std::filesystem::path& path,
            std::span<Variable* const> parameters,
            const TensorOptimizer& optimizer,
            const TrainingRandom& random);

        static void Load(
            const std::filesystem::path& path,
            std::span<Variable*> parameters,
            TensorOptimizer& optimizer,
            TrainingRandom& random);
    };
}

namespace kairo::foundation::math::training_checkpoint_detail
{
    constexpr char Magic[8] = { 'K', 'A', 'I', 'R', 'O', 'T', 'R', 'N' };
    constexpr std::uint32_t Version = 2;

    [[nodiscard]] std::FILE* OpenBinaryFile(
        const std::filesystem::path& path, bool write) noexcept
    {
#ifdef _WIN32
        return ::_wfopen(path.c_str(), write ? L"wb" : L"rb");
#else
        return std::fopen(path.c_str(), write ? "wb" : "rb");
#endif
    }

    template<class T>
    bool Write(std::FILE* file, const T& value)
    {
        return std::fwrite(&value, sizeof(T), 1, file) == 1;
    }

    template<class T>
    bool Read(std::FILE* file, T& value)
    {
        return std::fread(&value, sizeof(T), 1, file) == 1;
    }

    bool WriteTensor(std::FILE* file, const Tensor<float>& tensor)
    {
        const std::uint64_t rank = tensor.Rank();
        if (!Write(file, rank)) return false;
        for (std::size_t axis = 0; axis < tensor.Rank(); ++axis)
        {
            const std::uint64_t dimension = tensor.Dim(axis);
            if (!Write(file, dimension)) return false;
        }
        const std::uint64_t size = tensor.Size();
        return Write(file, size)
            && std::fwrite(tensor.Contiguous().Data(), sizeof(float), size, file) == size;
    }

    bool ReadTensor(std::FILE* file, Tensor<float>& tensor)
    {
        std::uint64_t rank = 0;
        if (!Read(file, rank) || rank == 0 || rank > 16) return false;
        Tensor<float>::Shape shape(static_cast<std::size_t>(rank));
        std::uint64_t expectedSize = 1;
        for (std::uint64_t axis = 0; axis < rank; ++axis)
        {
            std::uint64_t dimension = 0;
            if (!Read(file, dimension) || dimension == 0
                || expectedSize > std::numeric_limits<std::uint64_t>::max() / dimension)
                return false;
            shape[static_cast<std::size_t>(axis)] = static_cast<std::size_t>(dimension);
            expectedSize *= dimension;
        }
        std::uint64_t size = 0;
        if (!Read(file, size) || size != expectedSize
            || size > std::numeric_limits<std::size_t>::max() / sizeof(float))
            return false;
        Tensor<float> loaded(std::move(shape), 0.0f);
        if (std::fread(loaded.Data(), sizeof(float), static_cast<std::size_t>(size), file) != size)
            return false;
        tensor = std::move(loaded);
        return true;
    }
}

namespace kairo::foundation::math
{
    inline void TensorTrainingCheckpoint::Save(
        const std::filesystem::path& path,
        std::span<Variable* const> parameters,
        const TensorOptimizer& optimizer,
        const TrainingRandom& random)
    {
        namespace detail = training_checkpoint_detail;
        if (path.empty() || parameters.empty())
            throw std::invalid_argument("Checkpoint Save requires a path and parameters.");
        const std::filesystem::path temporary = path.string() + ".tmp";
        std::FILE* file = detail::OpenBinaryFile(temporary, true);
        if (!file) throw std::runtime_error("Cannot open checkpoint temporary file.");
        const TensorOptimizerState state = optimizer.State();
        const std::uint64_t parameterCount = parameters.size();
        const std::uint32_t optimizerKind = static_cast<std::uint32_t>(state.config.kind);
        const std::uint32_t scheduleKind = static_cast<std::uint32_t>(state.config.schedule.kind);
        bool success = std::fwrite(detail::Magic, 1, sizeof(detail::Magic), file) == sizeof(detail::Magic)
            && detail::Write(file, detail::Version)
            && detail::Write(file, parameterCount)
            && detail::Write(file, optimizerKind)
            && detail::Write(file, scheduleKind)
            && detail::Write(file, state.config.schedule.baseRate)
            && detail::Write(file, state.config.schedule.minimumRate)
            && detail::Write(file, state.config.schedule.decayFactor)
            && detail::Write(file, state.config.schedule.decaySteps)
            && detail::Write(file, state.config.schedule.warmupSteps)
            && detail::Write(file, state.config.momentum)
            && detail::Write(file, state.config.beta1)
            && detail::Write(file, state.config.beta2)
            && detail::Write(file, state.config.epsilon)
            && detail::Write(file, state.config.weightDecay)
            && detail::Write(file, state.config.maximumGradientNorm)
            && detail::Write(file, state.completedSteps)
            && detail::Write(file, random.State());
        for (std::size_t index = 0; success && index < parameters.size(); ++index)
            success = parameters[index] && detail::WriteTensor(file, parameters[index]->Value())
                && detail::WriteTensor(file, state.firstMoments[index])
                && detail::WriteTensor(file, state.secondMoments[index]);
        success = std::fclose(file) == 0 && success;
        if (!success)
        {
            std::filesystem::remove(temporary);
            throw std::runtime_error("Checkpoint write failed.");
        }
        std::error_code error;
        std::filesystem::rename(temporary, path, error);
        if (error)
        {
            std::filesystem::remove(temporary);
            throw std::runtime_error("Checkpoint atomic rename failed: " + error.message());
        }
    }

    inline void TensorTrainingCheckpoint::Load(
        const std::filesystem::path& path,
        std::span<Variable*> parameters,
        TensorOptimizer& optimizer,
        TrainingRandom& random)
    {
        namespace detail = training_checkpoint_detail;
        if (path.empty() || parameters.empty())
            throw std::invalid_argument("Checkpoint Load requires a path and parameters.");
        std::FILE* file = detail::OpenBinaryFile(path, false);
        if (!file) throw std::runtime_error("Cannot open training checkpoint.");
        char magic[sizeof(detail::Magic)]{};
        std::uint32_t version = 0;
        std::uint64_t parameterCount = 0;
        std::uint32_t optimizerKind = 0, scheduleKind = 0;
        TensorOptimizerState loaded;
        std::uint64_t randomState = 0;
        bool success = std::fread(magic, 1, sizeof(magic), file) == sizeof(magic)
            && std::memcmp(magic, detail::Magic, sizeof(magic)) == 0
            && detail::Read(file, version) && version == detail::Version
            && detail::Read(file, parameterCount) && parameterCount == parameters.size()
            && detail::Read(file, optimizerKind)
            && optimizerKind <= static_cast<std::uint32_t>(TensorOptimizerKind::AdamW)
            && detail::Read(file, scheduleKind)
            && scheduleKind <= static_cast<std::uint32_t>(LearningRateScheduleKind::CosineDecay);
        loaded.config.kind = static_cast<TensorOptimizerKind>(optimizerKind);
        loaded.config.schedule.kind = static_cast<LearningRateScheduleKind>(scheduleKind);
        success = success
            && detail::Read(file, loaded.config.schedule.baseRate)
            && detail::Read(file, loaded.config.schedule.minimumRate)
            && detail::Read(file, loaded.config.schedule.decayFactor)
            && detail::Read(file, loaded.config.schedule.decaySteps)
            && detail::Read(file, loaded.config.schedule.warmupSteps)
            && detail::Read(file, loaded.config.momentum)
            && detail::Read(file, loaded.config.beta1)
            && detail::Read(file, loaded.config.beta2)
            && detail::Read(file, loaded.config.epsilon)
            && detail::Read(file, loaded.config.weightDecay)
            && detail::Read(file, loaded.config.maximumGradientNorm)
            && detail::Read(file, loaded.completedSteps)
            && detail::Read(file, randomState);
        std::vector<Tensor<float>> values(parameters.size());
        loaded.firstMoments.resize(parameters.size());
        loaded.secondMoments.resize(parameters.size());
        for (std::size_t index = 0; success && index < parameters.size(); ++index)
            success = detail::ReadTensor(file, values[index])
                && detail::ReadTensor(file, loaded.firstMoments[index])
                && detail::ReadTensor(file, loaded.secondMoments[index]);
        const int trailing = success ? std::fgetc(file) : EOF;
        std::fclose(file);
        if (!success || trailing != EOF || randomState == 0)
            throw std::runtime_error("Training checkpoint is malformed or incompatible.");

        // Validate everything against live shapes before the first mutation.
        TensorOptimizer validation(loaded.config);
        validation.Restore(loaded, parameters);
        for (std::size_t index = 0; index < parameters.size(); ++index)
        {
            if (values[index].Rank() != parameters[index]->Value().Rank())
                throw std::runtime_error("Checkpoint parameter rank mismatch.");
            for (std::size_t axis = 0; axis < values[index].Rank(); ++axis)
                if (values[index].Dim(axis) != parameters[index]->Value().Dim(axis))
                    throw std::runtime_error("Checkpoint parameter shape mismatch.");
        }
        for (std::size_t index = 0; index < parameters.size(); ++index)
            parameters[index]->LoadValue(std::move(values[index]));
        optimizer.Restore(loaded, parameters);
        random.Restore(randomState);
    }
}
