module;

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

export module Kairo.Foundation.Math.TensorData;

import Kairo.Foundation.Math.Tensor;
import Kairo.Foundation.Math.TensorTraining;

export namespace kairo::foundation::math
{
    struct TensorBatch final
    {
        Tensor<float> samples;
        Tensor<float> labels;
        std::vector<std::size_t> sourceIndices;
    };

    /// An immutable indexed dataset over contiguous Float32 sample and label
    /// tensors. Axis zero is the sample axis; all remaining dimensions are
    /// preserved by batching. Dataset transforms reorder indices rather than
    /// duplicating source storage.
    class TensorDataset final
    {
    public:
        TensorDataset(Tensor<float> samples, Tensor<float> labels)
            : samples_(samples.Contiguous()), labels_(labels.Contiguous())
        {
            if (samples_.Rank() == 0 || labels_.Rank() == 0
                || samples_.Dim(0) == 0 || samples_.Dim(0) != labels_.Dim(0))
                throw std::invalid_argument(
                    "TensorDataset requires non-empty tensors with equal axis-zero size.");
            indices_.resize(samples_.Dim(0));
            for (std::size_t index = 0; index < indices_.size(); ++index)
                indices_[index] = index;
        }

        [[nodiscard]] std::size_t Size() const noexcept { return indices_.size(); }
        [[nodiscard]] const std::vector<std::size_t>& Indices() const noexcept
        {
            return indices_;
        }

        /// Input: deterministic seed.
        /// Output: a dataset sharing source tensors with Fisher-Yates reordered
        /// indices.
        [[nodiscard]] TensorDataset Shuffled(std::uint64_t seed) const
        {
            TensorDataset result(*this);
            TrainingRandom random(seed);
            for (std::size_t remaining = result.indices_.size(); remaining > 1; --remaining)
            {
                const std::size_t selected =
                    static_cast<std::size_t>(random.Next() % remaining);
                std::swap(result.indices_[remaining - 1], result.indices_[selected]);
            }
            return result;
        }

        /// Input: validation fraction strictly between zero and one plus seed.
        /// Output: disjoint train and validation index sets covering the source
        /// exactly once. Both partitions are guaranteed non-empty.
        [[nodiscard]] std::pair<TensorDataset, TensorDataset> Split(
            float validationFraction, std::uint64_t seed) const
        {
            if (!(validationFraction > 0.0f && validationFraction < 1.0f) || Size() < 2)
                throw std::invalid_argument(
                    "TensorDataset Split requires at least two samples and a fraction in (0,1).");
            TensorDataset shuffled = Shuffled(seed);
            const std::size_t validationSize = std::clamp(
                static_cast<std::size_t>(
                    static_cast<double>(Size()) * validationFraction + 0.5),
                std::size_t{1}, Size() - 1);
            TensorDataset training(*this);
            TensorDataset validation(*this);
            training.indices_.assign(
                shuffled.indices_.begin() + static_cast<std::ptrdiff_t>(validationSize),
                shuffled.indices_.end());
            validation.indices_.assign(
                shuffled.indices_.begin(),
                shuffled.indices_.begin() + static_cast<std::ptrdiff_t>(validationSize));
            return { std::move(training), std::move(validation) };
        }

        /// Input: offset and count in this dataset's index order.
        /// Output: owning contiguous tensors whose leading dimension is count.
        /// Failure: rejects empty or out-of-range requests.
        [[nodiscard]] TensorBatch Gather(std::size_t offset, std::size_t count) const
        {
            if (count == 0 || offset > Size() || count > Size() - offset)
                throw std::out_of_range("TensorDataset Gather range is invalid.");
            Tensor<float>::Shape sampleShape = samples_.GetShape();
            Tensor<float>::Shape labelShape = labels_.GetShape();
            sampleShape[0] = count;
            labelShape[0] = count;
            Tensor<float> samples(sampleShape, 0.0f);
            Tensor<float> labels(labelShape, 0.0f);
            const std::size_t sampleWidth = samples_.Size() / samples_.Dim(0);
            const std::size_t labelWidth = labels_.Size() / labels_.Dim(0);
            std::vector<std::size_t> gatheredIndices;
            gatheredIndices.reserve(count);
            for (std::size_t output = 0; output < count; ++output)
            {
                const std::size_t source = indices_[offset + output];
                gatheredIndices.push_back(source);
                std::copy_n(
                    samples_.Data() + source * sampleWidth,
                    sampleWidth,
                    samples.Data() + output * sampleWidth);
                std::copy_n(
                    labels_.Data() + source * labelWidth,
                    labelWidth,
                    labels.Data() + output * labelWidth);
            }
            return { std::move(samples), std::move(labels), std::move(gatheredIndices) };
        }

    private:
        Tensor<float> samples_;
        Tensor<float> labels_;
        std::vector<std::size_t> indices_;
    };

    /// Single-pass deterministic batch loader. The final short batch is
    /// returned unless dropLast is enabled.
    class TensorBatchLoader final
    {
    public:
        TensorBatchLoader(TensorDataset dataset, std::size_t batchSize, bool dropLast = false)
            : dataset_(std::move(dataset)), batchSize_(batchSize), dropLast_(dropLast)
        {
            if (batchSize_ == 0)
                throw std::invalid_argument("TensorBatchLoader requires non-zero batch size.");
        }

        [[nodiscard]] std::optional<TensorBatch> Next()
        {
            if (cursor_ >= dataset_.Size()) return std::nullopt;
            const std::size_t remaining = dataset_.Size() - cursor_;
            if (dropLast_ && remaining < batchSize_)
            {
                cursor_ = dataset_.Size();
                return std::nullopt;
            }
            const std::size_t count = std::min(batchSize_, remaining);
            TensorBatch batch = dataset_.Gather(cursor_, count);
            cursor_ += count;
            return batch;
        }

        [[nodiscard]] std::size_t Remaining() const noexcept
        {
            return dataset_.Size() - cursor_;
        }

    private:
        TensorDataset dataset_;
        std::size_t batchSize_;
        bool dropLast_;
        std::size_t cursor_ = 0;
    };

    /// Bounded producer/consumer wrapper around TensorBatchLoader. One jthread
    /// performs shape-preserving gathers while the training thread computes.
    /// Destruction requests cancellation and joins; worker exceptions are
    /// rethrown by Next.
    class TensorPrefetchLoader final
    {
    public:
        TensorPrefetchLoader(
            TensorDataset dataset,
            std::size_t batchSize,
            std::size_t capacity = 2,
            bool dropLast = false)
            : state_(std::make_shared<State>(capacity)),
              worker_([state = state_, loader = TensorBatchLoader(
                  std::move(dataset), batchSize, dropLast)](std::stop_token stop) mutable
              {
                  try
                  {
                      while (!stop.stop_requested())
                      {
                          std::optional<TensorBatch> batch = loader.Next();
                          std::unique_lock lock(state->mutex);
                          state->space.wait(lock, stop, [&]
                          {
                              return state->queue.size() < state->capacity;
                          });
                          if (stop.stop_requested()) break;
                          if (!batch)
                          {
                              state->finished = true;
                              lock.unlock();
                              state->ready.notify_all();
                              return;
                          }
                          state->queue.push_back(std::move(*batch));
                          lock.unlock();
                          state->ready.notify_one();
                      }
                  }
                  catch (...)
                  {
                      std::lock_guard lock(state->mutex);
                      state->failure = std::current_exception();
                      state->finished = true;
                      state->ready.notify_all();
                  }
              })
        {
            if (capacity == 0)
                throw std::invalid_argument("TensorPrefetchLoader capacity must be non-zero.");
        }

        TensorPrefetchLoader(const TensorPrefetchLoader&) = delete;
        TensorPrefetchLoader& operator=(const TensorPrefetchLoader&) = delete;

        ~TensorPrefetchLoader()
        {
            worker_.request_stop();
            state_->space.notify_all();
            state_->ready.notify_all();
        }

        [[nodiscard]] std::optional<TensorBatch> Next()
        {
            std::unique_lock lock(state_->mutex);
            state_->ready.wait(lock, [&]
            {
                return !state_->queue.empty() || state_->finished;
            });
            if (state_->failure) std::rethrow_exception(state_->failure);
            if (state_->queue.empty()) return std::nullopt;
            TensorBatch batch = std::move(state_->queue.front());
            state_->queue.pop_front();
            lock.unlock();
            state_->space.notify_one();
            return batch;
        }

    private:
        struct State final
        {
            explicit State(std::size_t requestedCapacity) : capacity(requestedCapacity) {}
            std::size_t capacity;
            std::mutex mutex;
            std::condition_variable_any ready;
            std::condition_variable_any space;
            std::deque<TensorBatch> queue;
            std::exception_ptr failure;
            bool finished = false;
        };

        std::shared_ptr<State> state_;
        std::jthread worker_;
    };
}
