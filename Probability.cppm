module;

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

export module Kairo.Foundation.Math.Probability;

import Kairo.Foundation.Math.Vector;

export namespace kairo::foundation::math
{
    /// Input: optional deterministic seed.
    /// Output: reusable pseudo-random number generator wrapper.
    /// Task: centralize sampling state for probability utilities. The generator
    /// is explicit so math code stays deterministic under tests and never hides
    /// global mutable state behind distribution functions.
    class RandomGenerator final
    {
    private:
        std::mt19937_64 m_engine;

    public:
        using EngineType = std::mt19937_64;

        /// Input: seed value.
        /// Output: generator initialized with that seed.
        /// Task: make random sequences reproducible by default.
        explicit RandomGenerator(std::uint64_t seed = 0x4B4149524F4D4154ULL)
            : m_engine(seed)
        {
        }

        /// Input: seed value.
        /// Output: none; resets the internal engine state.
        /// Task: allow deterministic replay of sampling-heavy tests or tools.
        void Seed(std::uint64_t seed)
        {
            m_engine.seed(seed);
        }

        /// Input: none.
        /// Output: reference to the underlying standard engine.
        /// Task: allow advanced callers to interoperate with `<random>`
        /// distributions without exposing hidden global state.
        [[nodiscard]]
        EngineType& Engine() noexcept
        {
            return m_engine;
        }

        /// Input: none.
        /// Output: const reference to the underlying standard engine.
        /// Task: expose engine metadata without mutating the sequence.
        [[nodiscard]]
        const EngineType& Engine() const noexcept
        {
            return m_engine;
        }

        /// Input: inclusive integer bounds.
        /// Output: uniformly sampled integer in [minValue, maxValue].
        /// Task: provide deterministic integer sampling for algorithms that need
        /// random indices, bootstrap resampling, or randomized tests.
        [[nodiscard]]
        std::uint64_t UniformUInt(
            std::uint64_t minValue,
            std::uint64_t maxValue)
        {
            if (minValue > maxValue)
            {
                throw std::invalid_argument("UniformUInt failed: minValue must be <= maxValue.");
            }

            std::uniform_int_distribution<std::uint64_t> distribution(
                minValue,
                maxValue);

            return distribution(m_engine);
        }

        /// Input: floating bounds.
        /// Output: uniformly sampled floating-point value in [minValue, maxValue).
        /// Task: provide deterministic continuous sampling with runtime guards
        /// for data-driven ranges.
        template<FloatingPoint T>
        [[nodiscard]]
        T UniformReal(
            T minValue = T(0),
            T maxValue = T(1))
        {
            if (!(minValue < maxValue))
            {
                throw std::invalid_argument("UniformReal failed: minValue must be < maxValue.");
            }

            std::uniform_real_distribution<T> distribution(
                minValue,
                maxValue);

            return distribution(m_engine);
        }
    };

    /// Input: continuous interval [Min, Max].
    /// Output: uniform distribution value object.
    /// Task: provide PDF/CDF/moment/sampling helpers for bounded random values.
    template<FloatingPoint T>
    struct UniformDistribution final
    {
        T Min = T(0);
        T Max = T(1);

        /// Input: min and max bounds.
        /// Output: validated uniform distribution.
        /// Task: reject zero-width or inverted ranges as public runtime input errors.
        constexpr UniformDistribution(T minValue = T(0), T maxValue = T(1))
            : Min(minValue)
            , Max(maxValue)
        {
            if (!(Min < Max))
            {
                throw std::invalid_argument("UniformDistribution failed: Min must be < Max.");
            }
        }

        /// Input: sample value.
        /// Output: probability density at that value.
        /// Task: evaluate the analytic uniform PDF.
        [[nodiscard]]
        constexpr T Pdf(T x) const noexcept
        {
            return (x >= Min && x <= Max)
                ? T(1) / (Max - Min)
                : T(0);
        }

        /// Input: sample value.
        /// Output: cumulative probability up to that value.
        /// Task: evaluate the analytic uniform CDF with clamped tails.
        [[nodiscard]]
        constexpr T Cdf(T x) const noexcept
        {
            if (x <= Min)
            {
                return T(0);
            }
            if (x >= Max)
            {
                return T(1);
            }

            return (x - Min) / (Max - Min);
        }

        /// Input: none.
        /// Output: distribution mean.
        [[nodiscard]]
        constexpr T MeanValue() const noexcept
        {
            return (Min + Max) / T(2);
        }

        /// Input: none.
        /// Output: distribution variance.
        [[nodiscard]]
        constexpr T Variance() const noexcept
        {
            const T width =
                Max - Min;

            return (width * width) / T(12);
        }

        /// Input: random generator.
        /// Output: sample drawn from the distribution.
        /// Task: connect analytic distribution metadata to explicit sampling state.
        [[nodiscard]]
        T Sample(RandomGenerator& generator) const
        {
            return generator.UniformReal(Min, Max);
        }
    };

    /// Input: mean and positive standard deviation.
    /// Output: normal distribution value object.
    /// Task: provide Gaussian PDF/CDF/moment/sampling helpers for statistics,
    /// optimization diagnostics, and simulation noise models.
    template<FloatingPoint T>
    struct NormalDistribution final
    {
        T Mean = T(0);
        T StandardDeviation = T(1);

        /// Input: mean and standard deviation.
        /// Output: validated normal distribution.
        /// Task: reject invalid scale as public runtime input.
        NormalDistribution(T mean = T(0), T standardDeviation = T(1))
            : Mean(mean)
            , StandardDeviation(standardDeviation)
        {
            if (!(StandardDeviation > T(0)))
            {
                throw std::invalid_argument("NormalDistribution failed: standard deviation must be positive.");
            }
        }

        /// Input: sample value.
        /// Output: probability density at that value.
        /// Task: evaluate the analytic Gaussian PDF.
        [[nodiscard]]
        T Pdf(T x) const noexcept
        {
            constexpr T pi =
                T(3.141592653589793238462643383279502884L);

            const T z =
                (x - Mean) / StandardDeviation;

            return std::exp(-T(0.5) * z * z) /
                (StandardDeviation * std::sqrt(T(2) * pi));
        }

        /// Input: sample value.
        /// Output: cumulative probability up to that value.
        /// Task: evaluate the Gaussian CDF using the standard error function.
        [[nodiscard]]
        T Cdf(T x) const noexcept
        {
            const T z =
                (x - Mean) / (StandardDeviation * std::sqrt(T(2)));

            return T(0.5) * (T(1) + std::erf(z));
        }

        /// Input: none.
        /// Output: distribution mean.
        [[nodiscard]]
        constexpr T MeanValue() const noexcept
        {
            return Mean;
        }

        /// Input: none.
        /// Output: distribution variance.
        [[nodiscard]]
        constexpr T Variance() const noexcept
        {
            return StandardDeviation * StandardDeviation;
        }

        /// Input: random generator.
        /// Output: sample drawn from the distribution.
        /// Task: use the standard library's normal sampler while preserving
        /// explicit caller-owned random state.
        [[nodiscard]]
        T Sample(RandomGenerator& generator) const
        {
            std::normal_distribution<T> distribution(
                Mean,
                StandardDeviation);

            return distribution(generator.Engine());
        }
    };

    /// Input: success probability p in [0, 1].
    /// Output: Bernoulli distribution value object.
    /// Task: model binary events and deterministic sampling for tests,
    /// classifiers, and randomized algorithms.
    template<FloatingPoint T>
    struct BernoulliDistribution final
    {
        T Probability = T(0.5);

        /// Input: success probability.
        /// Output: validated Bernoulli distribution.
        explicit constexpr BernoulliDistribution(T probability = T(0.5))
            : Probability(probability)
        {
            if (Probability < T(0) || Probability > T(1))
            {
                throw std::invalid_argument("BernoulliDistribution failed: probability must be in [0, 1].");
            }
        }

        /// Input: event outcome.
        /// Output: probability mass for that outcome.
        [[nodiscard]]
        constexpr T Pmf(bool value) const noexcept
        {
            return value
                ? Probability
                : T(1) - Probability;
        }

        /// Input: none.
        /// Output: expected value.
        [[nodiscard]]
        constexpr T MeanValue() const noexcept
        {
            return Probability;
        }

        /// Input: none.
        /// Output: variance.
        [[nodiscard]]
        constexpr T Variance() const noexcept
        {
            return Probability * (T(1) - Probability);
        }

        /// Input: random generator.
        /// Output: sampled boolean event.
        [[nodiscard]]
        bool Sample(RandomGenerator& generator) const
        {
            std::bernoulli_distribution distribution(
                static_cast<double>(Probability));

            return distribution(generator.Engine());
        }
    };

    /// Input: positive rate lambda.
    /// Output: exponential distribution value object.
    /// Task: model waiting times and decay processes with analytic PDF/CDF and
    /// inverse-transform sampling.
    template<FloatingPoint T>
    struct ExponentialDistribution final
    {
        T Lambda = T(1);

        /// Input: positive rate.
        /// Output: validated exponential distribution.
        explicit constexpr ExponentialDistribution(T lambda = T(1))
            : Lambda(lambda)
        {
            if (!(Lambda > T(0)))
            {
                throw std::invalid_argument("ExponentialDistribution failed: lambda must be positive.");
            }
        }

        /// Input: sample value.
        /// Output: probability density at that value.
        [[nodiscard]]
        T Pdf(T x) const noexcept
        {
            return x < T(0)
                ? T(0)
                : Lambda * std::exp(-Lambda * x);
        }

        /// Input: sample value.
        /// Output: cumulative probability up to that value.
        [[nodiscard]]
        T Cdf(T x) const noexcept
        {
            return x < T(0)
                ? T(0)
                : T(1) - std::exp(-Lambda * x);
        }

        /// Input: none.
        /// Output: expected value.
        [[nodiscard]]
        constexpr T MeanValue() const noexcept
        {
            return T(1) / Lambda;
        }

        /// Input: none.
        /// Output: variance.
        [[nodiscard]]
        constexpr T Variance() const noexcept
        {
            return T(1) / (Lambda * Lambda);
        }

        /// Input: random generator.
        /// Output: sampled waiting time.
        [[nodiscard]]
        T Sample(RandomGenerator& generator) const
        {
            const T u =
                std::max(
                    generator.UniformReal<T>(),
                    std::numeric_limits<T>::min());

            return -std::log(u) / Lambda;
        }
    };

    /// Input: non-empty list of non-negative weights.
    /// Output: total weight.
    /// Task: validate data-driven discrete sampling input before random choice.
    template<FloatingPoint T>
    [[nodiscard]]
    T ValidateWeights(const std::vector<T>& weights)
    {
        if (weights.empty())
        {
            throw std::invalid_argument("ValidateWeights failed: weights must not be empty.");
        }

        T total = T(0);
        for (const T weight : weights)
        {
            if (weight < T(0))
            {
                throw std::invalid_argument("ValidateWeights failed: weights must be non-negative.");
            }

            total += weight;
        }

        if (!(total > T(0)))
        {
            throw std::invalid_argument("ValidateWeights failed: at least one weight must be positive.");
        }

        return total;
    }

    /// Input: non-empty non-negative weights and a random generator.
    /// Output: sampled index proportional to the supplied weights.
    /// Task: provide discrete sampling without forcing callers to construct a
    /// heavyweight distribution object for one-off choices.
    template<FloatingPoint T>
    [[nodiscard]]
    std::size_t SampleWeightedIndex(
        const std::vector<T>& weights,
        RandomGenerator& generator)
    {
        const T total =
            ValidateWeights(weights);

        const T target =
            generator.UniformReal<T>(T(0), total);

        T cumulative = T(0);
        for (std::size_t i = 0; i < weights.size(); ++i)
        {
            cumulative += weights[i];
            if (target <= cumulative)
            {
                return i;
            }
        }

        return weights.size() - 1;
    }

    /// Input: sample vector.
    /// Output: arithmetic mean.
    /// Task: provide a small probability/statistics helper for sampled data.
    template<FloatingPoint T>
    [[nodiscard]]
    T Mean(const std::vector<T>& samples)
    {
        if (samples.empty())
        {
            throw std::invalid_argument("Mean failed: samples must not be empty.");
        }

        T sum = T(0);
        for (const T sample : samples)
        {
            sum += sample;
        }

        return sum / static_cast<T>(samples.size());
    }

    /// Input: sample vector and Bessel-correction flag.
    /// Output: variance estimate.
    /// Task: support both population and unbiased sample variance calculations.
    template<FloatingPoint T>
    [[nodiscard]]
    T Variance(
        const std::vector<T>& samples,
        bool unbiased = true)
    {
        if (samples.empty())
        {
            throw std::invalid_argument("Variance failed: samples must not be empty.");
        }
        if (unbiased && samples.size() < 2)
        {
            throw std::invalid_argument("Variance failed: unbiased variance requires at least two samples.");
        }

        const T mean =
            Mean(samples);

        T sumSquares = T(0);
        for (const T sample : samples)
        {
            const T delta =
                sample - mean;

            sumSquares += delta * delta;
        }

        const std::size_t divisor =
            unbiased
                ? samples.size() - 1
                : samples.size();

        return sumSquares / static_cast<T>(divisor);
    }

    /// Input: sample vector and Bessel-correction flag.
    /// Output: standard deviation estimate.
    /// Task: convenience wrapper around `Variance()` for sampled data analysis.
    template<FloatingPoint T>
    [[nodiscard]]
    T StandardDeviation(
        const std::vector<T>& samples,
        bool unbiased = true)
    {
        return std::sqrt(
            Variance(
                samples,
                unbiased));
    }
}
