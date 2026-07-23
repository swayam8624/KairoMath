module;

#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

export module Kairo.Foundation.Math.TensorAutograd;

import Kairo.Foundation.Math.Tensor;

namespace kairo::foundation::math::autograd_detail
{
    struct Node final
    {
        Tensor<float> value;
        Tensor<float> gradient;
        bool requiresGradient = false;
        bool hasGradient = false;
        std::vector<std::shared_ptr<Node>> parents;
        std::function<void(const Tensor<float>&)> backward;
    };

    void Accumulate(const std::shared_ptr<Node>& node, const Tensor<float>& contribution)
    {
        if (!node->requiresGradient) return;
        if (node->value.Rank() != contribution.Rank())
            throw std::logic_error("Autograd contribution rank differs from its variable.");
        for (std::size_t axis = 0; axis < node->value.Rank(); ++axis)
            if (node->value.Dim(axis) != contribution.Dim(axis))
                throw std::logic_error("Autograd contribution shape differs from its variable.");
        if (!node->hasGradient)
        {
            node->gradient = Tensor<float>(node->value.GetShape(), 0.0f);
            node->hasGradient = true;
        }
        for (std::size_t index = 0; index < contribution.Size(); ++index)
            node->gradient[index] += contribution[index];
    }
}

export namespace kairo::foundation::math
{
    class Variable;
    [[nodiscard]] Variable Add(const Variable&, const Variable&);
    [[nodiscard]] Variable AddRowBias(const Variable&, const Variable&);
    [[nodiscard]] Variable Multiply(const Variable&, const Variable&);
    [[nodiscard]] Variable AutogradMatMul(const Variable&, const Variable&);
    [[nodiscard]] Variable AutogradReLU(const Variable&);
    [[nodiscard]] Variable MeanSquaredLoss(const Variable&, const Tensor<float>&);
    [[nodiscard]] Variable SoftmaxCrossEntropyLoss(const Variable&, const Tensor<float>&);

    /// A Float32 Tensor value participating in a dynamically constructed
    /// reverse-mode graph. Graph operations retain their parent states; leaf
    /// parameters remain valid across independently constructed training steps.
    class Variable final
    {
    public:
        explicit Variable(Tensor<float> value, bool requiresGradient = false)
            : state_(std::make_shared<autograd_detail::Node>())
        {
            if (value.Empty()) throw std::invalid_argument("Autograd variables require non-empty Tensor values.");
            state_->value = std::move(value);
            state_->requiresGradient = requiresGradient;
        }

        [[nodiscard]] const Tensor<float>& Value() const noexcept { return state_->value; }
        [[nodiscard]] bool RequiresGradient() const noexcept { return state_->requiresGradient; }
        [[nodiscard]] bool HasGradient() const noexcept { return state_->hasGradient; }

        [[nodiscard]] const Tensor<float>& Gradient() const
        {
            if (!state_->hasGradient) throw std::logic_error("Variable has no accumulated gradient.");
            return state_->gradient;
        }

        void ZeroGradient() noexcept
        {
            state_->gradient = Tensor<float>();
            state_->hasGradient = false;
        }

        /// Input: a scalar variable with exactly one element.
        /// Output: gradients accumulated into every reachable trainable leaf.
        /// Task: execute a fresh reverse topological traversal. Call
        /// `ZeroGradient` explicitly when accumulation across steps is unwanted.
        void Backward() const
        {
            if (state_->value.Size() != 1)
                throw std::invalid_argument("Backward requires a scalar one-element output.");
            std::vector<std::shared_ptr<autograd_detail::Node>> topology;
            std::unordered_set<const autograd_detail::Node*> visited;
            std::function<void(const std::shared_ptr<autograd_detail::Node>&)> visit;
            visit = [&](const std::shared_ptr<autograd_detail::Node>& node)
            {
                if (!visited.emplace(node.get()).second) return;
                for (const auto& parent : node->parents) visit(parent);
                topology.push_back(node);
            };
            visit(state_);
            autograd_detail::Accumulate(state_, Tensor<float>(state_->value.GetShape(), 1.0f));
            for (auto iterator = topology.rbegin(); iterator != topology.rend(); ++iterator)
                if ((*iterator)->backward && (*iterator)->hasGradient)
                    (*iterator)->backward((*iterator)->gradient);
        }

        void ApplySGD(float learningRate)
        {
            if (!state_->requiresGradient || !state_->hasGradient || !(learningRate > 0.0f))
                throw std::logic_error("ApplySGD requires a positive rate and an accumulated trainable gradient.");
            for (std::size_t index = 0; index < state_->value.Size(); ++index)
                state_->value[index] -= learningRate * state_->gradient[index];
        }

    private:
        explicit Variable(std::shared_ptr<autograd_detail::Node> state) : state_(std::move(state)) {}
        std::shared_ptr<autograd_detail::Node> state_;

        friend Variable Add(const Variable&, const Variable&);
        friend Variable AddRowBias(const Variable&, const Variable&);
        friend Variable Multiply(const Variable&, const Variable&);
        friend Variable AutogradMatMul(const Variable&, const Variable&);
        friend Variable AutogradReLU(const Variable&);
        friend Variable MeanSquaredLoss(const Variable&, const Tensor<float>&);
        friend Variable SoftmaxCrossEntropyLoss(const Variable&, const Tensor<float>&);
    };

    [[nodiscard]]
    inline Variable Add(const Variable& lhs, const Variable& rhs)
    {
        const Tensor<float> value = lhs.Value() + rhs.Value();
        auto result = std::make_shared<autograd_detail::Node>();
        result->value = value;
        result->requiresGradient = lhs.RequiresGradient() || rhs.RequiresGradient();
        if (result->requiresGradient)
        {
            result->parents = { lhs.state_, rhs.state_ };
            const auto left = lhs.state_;
            const auto right = rhs.state_;
            result->backward = [left, right](const Tensor<float>& upstream)
            {
                autograd_detail::Accumulate(left, upstream);
                autograd_detail::Accumulate(right, upstream);
            };
        }
        return Variable(std::move(result));
    }

    [[nodiscard]]
    inline Variable AddRowBias(const Variable& matrix, const Variable& bias)
    {
        if (matrix.Value().Rank() != 2 || bias.Value().Rank() != 1
            || matrix.Value().Dim(1) != bias.Value().Dim(0))
            throw std::invalid_argument("AddRowBias expects [rows,columns] and [columns].");
        Tensor<float> value = matrix.Value();
        for (std::size_t row = 0; row < value.Dim(0); ++row)
            for (std::size_t column = 0; column < value.Dim(1); ++column)
                value(row, column) += bias.Value()[column];
        auto result = std::make_shared<autograd_detail::Node>();
        result->value = std::move(value);
        result->requiresGradient = matrix.RequiresGradient() || bias.RequiresGradient();
        if (result->requiresGradient)
        {
            result->parents = { matrix.state_, bias.state_ };
            const auto matrixState = matrix.state_;
            const auto biasState = bias.state_;
            result->backward = [matrixState, biasState](const Tensor<float>& upstream)
            {
                autograd_detail::Accumulate(matrixState, upstream);
                if (biasState->requiresGradient)
                {
                    Tensor<float> biasGradient({ upstream.Dim(1) }, 0.0f);
                    for (std::size_t row = 0; row < upstream.Dim(0); ++row)
                        for (std::size_t column = 0; column < upstream.Dim(1); ++column)
                            biasGradient[column] += upstream(row, column);
                    autograd_detail::Accumulate(biasState, biasGradient);
                }
            };
        }
        return Variable(std::move(result));
    }

    [[nodiscard]]
    inline Variable Multiply(const Variable& lhs, const Variable& rhs)
    {
        const Tensor<float> value = lhs.Value() * rhs.Value();
        auto result = std::make_shared<autograd_detail::Node>();
        result->value = value;
        result->requiresGradient = lhs.RequiresGradient() || rhs.RequiresGradient();
        if (result->requiresGradient)
        {
            result->parents = { lhs.state_, rhs.state_ };
            const auto left = lhs.state_;
            const auto right = rhs.state_;
            result->backward = [left, right](const Tensor<float>& upstream)
            {
                if (left->requiresGradient)
                    autograd_detail::Accumulate(left, upstream * right->value);
                if (right->requiresGradient)
                    autograd_detail::Accumulate(right, upstream * left->value);
            };
        }
        return Variable(std::move(result));
    }

    [[nodiscard]]
    inline Variable AutogradMatMul(const Variable& lhs, const Variable& rhs)
    {
        const Tensor<float> value = MatMul(lhs.Value(), rhs.Value());
        auto result = std::make_shared<autograd_detail::Node>();
        result->value = value;
        result->requiresGradient = lhs.RequiresGradient() || rhs.RequiresGradient();
        if (result->requiresGradient)
        {
            result->parents = { lhs.state_, rhs.state_ };
            const auto left = lhs.state_;
            const auto right = rhs.state_;
            result->backward = [left, right](const Tensor<float>& upstream)
            {
                if (left->requiresGradient)
                {
                    Tensor<float> gradient(left->value.GetShape(), 0.0f);
                    for (std::size_t row = 0; row < left->value.Dim(0); ++row)
                        for (std::size_t inner = 0; inner < left->value.Dim(1); ++inner)
                            for (std::size_t column = 0; column < right->value.Dim(1); ++column)
                                gradient(row, inner) += upstream(row, column) * right->value(inner, column);
                    autograd_detail::Accumulate(left, gradient);
                }
                if (right->requiresGradient)
                {
                    Tensor<float> gradient(right->value.GetShape(), 0.0f);
                    for (std::size_t inner = 0; inner < right->value.Dim(0); ++inner)
                        for (std::size_t column = 0; column < right->value.Dim(1); ++column)
                            for (std::size_t row = 0; row < left->value.Dim(0); ++row)
                                gradient(inner, column) += left->value(row, inner) * upstream(row, column);
                    autograd_detail::Accumulate(right, gradient);
                }
            };
        }
        return Variable(std::move(result));
    }

    [[nodiscard]]
    inline Variable AutogradReLU(const Variable& input)
    {
        auto result = std::make_shared<autograd_detail::Node>();
        result->value = ReLU(input.Value());
        result->requiresGradient = input.RequiresGradient();
        if (result->requiresGradient)
        {
            result->parents = { input.state_ };
            const auto parent = input.state_;
            result->backward = [parent](const Tensor<float>& upstream)
            {
                Tensor<float> gradient(parent->value.GetShape(), 0.0f);
                for (std::size_t index = 0; index < gradient.Size(); ++index)
                    gradient[index] = parent->value[index] > 0.0f ? upstream[index] : 0.0f;
                autograd_detail::Accumulate(parent, gradient);
            };
        }
        return Variable(std::move(result));
    }

    [[nodiscard]]
    inline Variable MeanSquaredLoss(const Variable& predictions, const Tensor<float>& labels)
    {
        const float loss = MeanSquaredError(predictions.Value(), labels);
        auto result = std::make_shared<autograd_detail::Node>();
        result->value = Tensor<float>({ 1 }, loss);
        result->requiresGradient = predictions.RequiresGradient();
        if (result->requiresGradient)
        {
            result->parents = { predictions.state_ };
            const auto parent = predictions.state_;
            result->backward = [parent, labels](const Tensor<float>& upstream)
            {
                Tensor<float> gradient(parent->value.GetShape(), 0.0f);
                const float scale = 2.0f * upstream[0] / static_cast<float>(gradient.Size());
                for (std::size_t index = 0; index < gradient.Size(); ++index)
                    gradient[index] = scale * (parent->value[index] - labels[index]);
                autograd_detail::Accumulate(parent, gradient);
            };
        }
        return Variable(std::move(result));
    }

    [[nodiscard]]
    inline Variable SoftmaxCrossEntropyLoss(const Variable& logits, const Tensor<float>& labels)
    {
        const Tensor<float> probabilities = SoftmaxLastDim(logits.Value());
        const float loss = CrossEntropyMean(labels, probabilities);
        auto result = std::make_shared<autograd_detail::Node>();
        result->value = Tensor<float>({ 1 }, loss);
        result->requiresGradient = logits.RequiresGradient();
        if (result->requiresGradient)
        {
            result->parents = { logits.state_ };
            const auto parent = logits.state_;
            result->backward = [parent, labels, probabilities](const Tensor<float>& upstream)
            {
                const std::size_t samples = labels.Rank() >= 2 ? labels.Dim(0) : labels.Size();
                Tensor<float> gradient = probabilities - labels;
                const float scale = upstream[0] / static_cast<float>(samples);
                for (std::size_t index = 0; index < gradient.Size(); ++index)
                    gradient[index] *= scale;
                autograd_detail::Accumulate(parent, gradient);
            };
        }
        return Variable(std::move(result));
    }
}
