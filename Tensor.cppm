module;

#include <algorithm>
#include <cassert>
#include <bit>
#include <cmath>
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

export module Kairo.Foundation.Math.Tensor;

import Kairo.Foundation.Math.Vector;
import Kairo.Foundation.Math.DynamicMatrix;

export namespace kairo::foundation::math
{
    namespace tensor_dtype_detail
    {
        [[nodiscard]] constexpr std::uint16_t FloatToHalfBits(float value) noexcept
        {
            const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
            const std::uint32_t sign = (bits >> 16) & 0x8000u;
            const std::uint32_t exponent = (bits >> 23) & 0xFFu;
            const std::uint32_t mantissa = bits & 0x7FFFFFu;
            if (exponent == 0xFFu)
                return static_cast<std::uint16_t>(
                    sign | 0x7C00u | (mantissa == 0 ? 0u : 0x0200u));
            const int halfExponent = static_cast<int>(exponent) - 127 + 15;
            if (halfExponent >= 31) return static_cast<std::uint16_t>(sign | 0x7C00u);
            if (halfExponent <= 0)
            {
                if (halfExponent < -10) return static_cast<std::uint16_t>(sign);
                const std::uint32_t normalized = mantissa | 0x800000u;
                const int shift = 14 - halfExponent;
                std::uint32_t rounded = normalized >> shift;
                const std::uint32_t remainder = normalized & ((1u << shift) - 1u);
                const std::uint32_t halfway = 1u << (shift - 1);
                if (remainder > halfway || (remainder == halfway && (rounded & 1u))) ++rounded;
                return static_cast<std::uint16_t>(sign | rounded);
            }
            std::uint32_t roundedMantissa = mantissa >> 13;
            const std::uint32_t remainder = mantissa & 0x1FFFu;
            if (remainder > 0x1000u || (remainder == 0x1000u && (roundedMantissa & 1u)))
            {
                ++roundedMantissa;
                if (roundedMantissa == 0x400u)
                    return static_cast<std::uint16_t>(
                        sign | (static_cast<std::uint32_t>(halfExponent + 1) << 10));
            }
            return static_cast<std::uint16_t>(
                sign | (static_cast<std::uint32_t>(halfExponent) << 10)
                | (roundedMantissa & 0x3FFu));
        }

        [[nodiscard]] constexpr float HalfBitsToFloat(std::uint16_t value) noexcept
        {
            const std::uint32_t sign = (static_cast<std::uint32_t>(value & 0x8000u)) << 16;
            std::uint32_t exponent = (value >> 10) & 0x1Fu;
            std::uint32_t mantissa = value & 0x3FFu;
            std::uint32_t bits = 0;
            if (exponent == 0)
            {
                if (mantissa == 0) bits = sign;
                else
                {
                    int unbiased = -14;
                    while ((mantissa & 0x400u) == 0)
                    {
                        mantissa <<= 1;
                        --unbiased;
                    }
                    mantissa &= 0x3FFu;
                    bits = sign | (static_cast<std::uint32_t>(unbiased + 127) << 23)
                        | (mantissa << 13);
                }
            }
            else if (exponent == 0x1Fu)
                bits = sign | 0x7F800000u | (mantissa << 13);
            else
                bits = sign | ((exponent - 15u + 127u) << 23) | (mantissa << 13);
            return std::bit_cast<float>(bits);
        }
    }

    struct Float16 final
    {
        std::uint16_t bits = 0;
        constexpr Float16() noexcept = default;
        template<typename T> requires std::is_arithmetic_v<T>
        constexpr Float16(T value) noexcept
            : bits(tensor_dtype_detail::FloatToHalfBits(static_cast<float>(value))) {}
        [[nodiscard]] constexpr operator float() const noexcept
        {
            return tensor_dtype_detail::HalfBitsToFloat(bits);
        }
        constexpr Float16& operator+=(Float16 rhs) noexcept { return *this = Float16(float(*this) + float(rhs)); }
        constexpr Float16& operator-=(Float16 rhs) noexcept { return *this = Float16(float(*this) - float(rhs)); }
        constexpr Float16& operator*=(Float16 rhs) noexcept { return *this = Float16(float(*this) * float(rhs)); }
        constexpr Float16& operator/=(Float16 rhs) noexcept { return *this = Float16(float(*this) / float(rhs)); }
        friend constexpr Float16 operator+(Float16 a, Float16 b) noexcept { return Float16(float(a) + float(b)); }
        friend constexpr Float16 operator-(Float16 a, Float16 b) noexcept { return Float16(float(a) - float(b)); }
        friend constexpr Float16 operator-(Float16 value) noexcept { return Float16(-float(value)); }
        friend constexpr Float16 operator*(Float16 a, Float16 b) noexcept { return Float16(float(a) * float(b)); }
        friend constexpr Float16 operator/(Float16 a, Float16 b) noexcept { return Float16(float(a) / float(b)); }
        friend constexpr bool operator==(Float16, Float16) noexcept = default;
        friend constexpr auto operator<=>(Float16 a, Float16 b) noexcept { return float(a) <=> float(b); }
    };

    struct BFloat16 final
    {
        std::uint16_t bits = 0;
        constexpr BFloat16() noexcept = default;
        template<typename T> requires std::is_arithmetic_v<T>
        constexpr BFloat16(T value) noexcept
        {
            std::uint32_t raw = std::bit_cast<std::uint32_t>(static_cast<float>(value));
            raw += 0x7FFFu + ((raw >> 16) & 1u);
            bits = static_cast<std::uint16_t>(raw >> 16);
        }
        [[nodiscard]] constexpr operator float() const noexcept
        {
            return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
        }
        constexpr BFloat16& operator+=(BFloat16 rhs) noexcept { return *this = BFloat16(float(*this) + float(rhs)); }
        constexpr BFloat16& operator-=(BFloat16 rhs) noexcept { return *this = BFloat16(float(*this) - float(rhs)); }
        constexpr BFloat16& operator*=(BFloat16 rhs) noexcept { return *this = BFloat16(float(*this) * float(rhs)); }
        constexpr BFloat16& operator/=(BFloat16 rhs) noexcept { return *this = BFloat16(float(*this) / float(rhs)); }
        friend constexpr BFloat16 operator+(BFloat16 a, BFloat16 b) noexcept { return BFloat16(float(a) + float(b)); }
        friend constexpr BFloat16 operator-(BFloat16 a, BFloat16 b) noexcept { return BFloat16(float(a) - float(b)); }
        friend constexpr BFloat16 operator-(BFloat16 value) noexcept { return BFloat16(-float(value)); }
        friend constexpr BFloat16 operator*(BFloat16 a, BFloat16 b) noexcept { return BFloat16(float(a) * float(b)); }
        friend constexpr BFloat16 operator/(BFloat16 a, BFloat16 b) noexcept { return BFloat16(float(a) / float(b)); }
        friend constexpr bool operator==(BFloat16, BFloat16) noexcept = default;
        friend constexpr auto operator<=>(BFloat16 a, BFloat16 b) noexcept { return float(a) <=> float(b); }
    };

    template<typename T>
    concept TensorScalar = Arithmetic<T>
        || std::same_as<T, Float16> || std::same_as<T, BFloat16>;

    template<typename T>
    concept TensorFloating = FloatingPoint<T>
        || std::same_as<T, Float16> || std::same_as<T, BFloat16>;

    /// Describes the execution backend used by tensor kernels.
    ///
    /// Current implementation status:
    /// - Scalar: implemented and used by every kernel.
    /// - Threaded, SIMD, GPU: reserved API surface for later backend dispatch.
    ///
    /// Task: keep the public tensor API stable while allowing future optimized
    /// kernels to replace scalar loops without changing model/layer code.
    enum class TensorBackend
    {
        Scalar,
        Threaded,
        SIMD,
        GPU
    };

    /// A dynamically ranked row-major tensor with optional strided views.
    ///
    /// Design notes:
    /// - Logical indexing is row-major.
    /// - Owned tensors store data contiguously.
    /// - Views share the same storage and carry shape/stride/offset metadata.
    /// - `Data()` exposes the first physical element; use `IsContiguous()` when
    ///   passing the pointer to APIs that expect dense memory.
    /// - The scalar backend is intentionally simple and correct. Optimized
    ///   SIMD/thread/GPU kernels should be added behind these operations later.
    template<TensorScalar T>
    class Tensor final
    {
    public:
        using ValueType = T;
        using Shape = std::vector<std::size_t>;

    private:
        std::shared_ptr<std::vector<T>> m_storage;
        Shape m_shape;
        Shape m_strides;
        std::size_t m_offset = 0;
        TensorBackend m_backend = TensorBackend::Scalar;

    public:
        /// Input: none.
        /// Output: empty tensor with rank 0 and zero logical elements.
        /// Task: support deferred construction and container usage.
        /// Note: rank-0 scalar tensors should be constructed explicitly if that
        /// semantic is added later; the default tensor is an empty value.
        Tensor()
            : m_storage(std::make_shared<std::vector<T>>())
        {
        }

        /// Input: tensor shape.
        /// Output: zero-filled contiguous tensor.
        /// Task: allocate a dense row-major tensor for model parameters,
        /// activations, gradients, labels, and data batches.
        explicit Tensor(Shape shape)
            : m_storage(std::make_shared<std::vector<T>>(ElementCount(shape), T(0)))
            , m_shape(std::move(shape))
            , m_strides(MakeContiguousStrides(m_shape))
        {
        }

        /// Input: tensor shape and fill value.
        /// Output: contiguous tensor filled with value.
        /// Task: support explicit initialization for tests and constant tensors.
        Tensor(Shape shape, T value)
            : m_storage(std::make_shared<std::vector<T>>(ElementCount(shape), value))
            , m_shape(std::move(shape))
            , m_strides(MakeContiguousStrides(m_shape))
        {
        }

        /// Input: tensor shape and owned values in row-major order.
        /// Output: contiguous tensor containing the supplied values.
        /// Task: bridge datasets, converted weights, and serialized tensors into
        /// the tensor runtime.
        Tensor(Shape shape, std::vector<T> values)
            : m_storage(std::make_shared<std::vector<T>>(std::move(values)))
            , m_shape(std::move(shape))
            , m_strides(MakeContiguousStrides(m_shape))
        {
            if (m_storage->size() != ElementCount(m_shape))
            {
                throw std::invalid_argument("Tensor constructor error: data size does not match shape product.");
            }
        }

        /// Input: tensor shape.
        /// Output: zero-filled contiguous tensor.
        [[nodiscard]]
        static Tensor Zeros(Shape shape)
        {
            return Tensor(std::move(shape), T(0));
        }

        /// Input: tensor shape.
        /// Output: one-filled contiguous tensor.
        [[nodiscard]]
        static Tensor Ones(Shape shape)
        {
            return Tensor(std::move(shape), T(1));
        }

        /// Input: tensor shape and fill value.
        /// Output: value-filled contiguous tensor.
        [[nodiscard]]
        static Tensor Full(Shape shape, T value)
        {
            return Tensor(std::move(shape), value);
        }

        /// Input: count.
        /// Output: 1D tensor [0, 1, ..., count - 1].
        /// Task: support small tests, indexing examples, and deterministic
        /// synthetic data generation.
        [[nodiscard]]
        static Tensor Arange(std::size_t count)
        {
            Tensor result({ count });
            for (std::size_t i = 0; i < count; ++i)
            {
                result[i] = static_cast<T>(i);
            }
            return result;
        }

        /// Input: none.
        /// Output: tensor rank.
        [[nodiscard]]
        std::size_t Rank() const noexcept
        {
            return m_shape.size();
        }

        /// Input: none.
        /// Output: logical tensor shape.
        [[nodiscard]]
        const Shape& GetShape() const noexcept
        {
            return m_shape;
        }

        /// Input: axis index.
        /// Output: size of that axis.
        [[nodiscard]]
        std::size_t Dim(std::size_t axis) const noexcept
        {
            assert(axis < m_shape.size());
            return m_shape[axis];
        }

        /// Input: none.
        /// Output: row-major physical strides.
        [[nodiscard]]
        const Shape& Strides() const noexcept
        {
            return m_strides;
        }

        /// Input: none.
        /// Output: logical element count.
        [[nodiscard]]
        std::size_t Size() const noexcept
        {
            return ElementCount(m_shape);
        }

        /// Input: none.
        /// Output: true if there are no logical elements.
        [[nodiscard]]
        bool Empty() const noexcept
        {
            return Size() == 0;
        }

        /// Input: none.
        /// Output: current backend tag.
        [[nodiscard]]
        TensorBackend Backend() const noexcept
        {
            return m_backend;
        }

        /// Input: backend tag.
        /// Output: none.
        /// Task: record the desired backend. Today all kernels still run on the
        /// scalar implementation; future dispatch can use this flag.
        void SetBackend(TensorBackend backend) noexcept
        {
            m_backend = backend;
        }

        /// Input: none.
        /// Output: true if logical row-major order maps to contiguous memory.
        [[nodiscard]]
        bool IsContiguous() const noexcept
        {
            return m_offset == 0 && m_strides == MakeContiguousStrides(m_shape);
        }

        /// Input: none.
        /// Output: pointer to first physical element.
        /// Task: expose raw storage for kernels and interop. Check
        /// `IsContiguous()` before treating this as a dense logical range.
        [[nodiscard]]
        T* Data() noexcept
        {
            return m_storage->data() + m_offset;
        }

        /// Input: none.
        /// Output: const pointer to first physical element.
        [[nodiscard]]
        const T* Data() const noexcept
        {
            return m_storage->data() + m_offset;
        }

        /// Input: logical linear index in row-major order.
        /// Output: mutable element reference.
        /// Task: support generic kernels that iterate logical tensor order.
        [[nodiscard]]
        T& operator[](std::size_t linearIndex) noexcept
        {
            assert(linearIndex < Size());
            return (*m_storage)[PhysicalOffsetFromLinear(linearIndex)];
        }

        /// Input: logical linear index in row-major order.
        /// Output: const element reference.
        [[nodiscard]]
        const T& operator[](std::size_t linearIndex) const noexcept
        {
            assert(linearIndex < Size());
            return (*m_storage)[PhysicalOffsetFromLinear(linearIndex)];
        }

        /// Input: tensor indices, one per axis.
        /// Output: mutable element reference.
        [[nodiscard]]
        T& At(std::span<const std::size_t> indices) noexcept
        {
            return (*m_storage)[PhysicalOffset(indices)];
        }

        /// Input: tensor indices, one per axis.
        /// Output: const element reference.
        [[nodiscard]]
        const T& At(std::span<const std::size_t> indices) const noexcept
        {
            return (*m_storage)[PhysicalOffset(indices)];
        }

        /// Input: tensor indices as an initializer list.
        /// Output: mutable element reference.
        [[nodiscard]]
        T& At(std::initializer_list<std::size_t> indices) noexcept
        {
            return At(std::span<const std::size_t>(indices.begin(), indices.size()));
        }

        /// Input: tensor indices as an initializer list.
        /// Output: const element reference.
        [[nodiscard]]
        const T& At(std::initializer_list<std::size_t> indices) const noexcept
        {
            return At(std::span<const std::size_t>(indices.begin(), indices.size()));
        }

        /// Input: row and column for a rank-2 tensor.
        /// Output: mutable element reference.
        [[nodiscard]]
        T& operator()(std::size_t row, std::size_t column) noexcept
        {
            assert(Rank() == 2);
            std::size_t indices[2] = { row, column };
            return At(std::span<const std::size_t>(indices, 2));
        }

        /// Input: row and column for a rank-2 tensor.
        /// Output: const element reference.
        [[nodiscard]]
        const T& operator()(std::size_t row, std::size_t column) const noexcept
        {
            assert(Rank() == 2);
            std::size_t indices[2] = { row, column };
            return At(std::span<const std::size_t>(indices, 2));
        }

        /// Input: new shape with the same element count.
        /// Output: a view over the same storage.
        /// Task: reshape activations without copying when the tensor is
        /// contiguous.
        [[nodiscard]]
        Tensor Reshape(Shape newShape) const
        {
            if (!IsContiguous())
            {
                throw std::logic_error("Tensor::Reshape requires a contiguous tensor.");
            }
            if (ElementCount(newShape) != Size())
            {
                throw std::invalid_argument("Tensor::Reshape shape product mismatch.");
            }

            Tensor result;
            result.m_storage = m_storage;
            result.m_shape = std::move(newShape);
            result.m_strides = MakeContiguousStrides(result.m_shape);
            result.m_offset = m_offset;
            result.m_backend = m_backend;
            return result;
        }

        /// Input: axis, start, count.
        /// Output: strided view over the requested slice.
        /// Task: support batching and sub-tensor access without copying.
        [[nodiscard]]
        Tensor Slice(std::size_t axis, std::size_t start, std::size_t count) const
        {
            if (axis >= Rank() || start + count > m_shape[axis])
            {
                throw std::out_of_range("Tensor::Slice range is outside the tensor shape.");
            }

            Tensor result;
            result.m_storage = m_storage;
            result.m_shape = m_shape;
            result.m_shape[axis] = count;
            result.m_strides = m_strides;
            result.m_offset = m_offset + start * m_strides[axis];
            result.m_backend = m_backend;
            return result;
        }

        /// Input: none.
        /// Output: contiguous copy if needed, otherwise a dense copy.
        /// Task: normalize views before passing to kernels that require dense
        /// memory or serialization.
        [[nodiscard]]
        Tensor Contiguous() const
        {
            Tensor result(m_shape);
            for (std::size_t i = 0; i < Size(); ++i)
            {
                result[i] = (*this)[i];
            }
            result.m_backend = m_backend;
            return result;
        }

        /// Input: fill value.
        /// Output: this tensor after assignment.
        Tensor& Fill(T value) noexcept
        {
            for (std::size_t i = 0; i < Size(); ++i)
            {
                (*this)[i] = value;
            }
            return *this;
        }

        /// Input: unary function object.
        /// Output: new tensor with function applied element-wise.
        template<typename Fn>
        [[nodiscard]]
        Tensor Map(Fn&& fn) const
        {
            Tensor result(m_shape);
            for (std::size_t i = 0; i < Size(); ++i)
            {
                result[i] = static_cast<T>(fn((*this)[i]));
            }
            return result;
        }

        /// Input: unary function object.
        /// Output: this tensor after in-place element-wise mutation.
        template<typename Fn>
        Tensor& ApplyInPlace(Fn&& fn)
        {
            for (std::size_t i = 0; i < Size(); ++i)
            {
                (*this)[i] = static_cast<T>(fn((*this)[i]));
            }
            return *this;
        }

        /// Input: none.
        /// Output: sum of all logical elements.
        [[nodiscard]]
        T Sum() const noexcept
        {
            T sum = T(0);
            for (std::size_t i = 0; i < Size(); ++i)
            {
                sum += (*this)[i];
            }
            return sum;
        }

        /// Input: none.
        /// Output: arithmetic mean of all logical elements.
        [[nodiscard]]
        T Mean() const noexcept
            requires TensorFloating<T>
        {
            return Empty() ? T(0) : Sum() / static_cast<T>(Size());
        }

        /// Input: none.
        /// Output: maximum element, or zero for empty tensors.
        [[nodiscard]]
        T Max() const noexcept
        {
            if (Empty())
            {
                return T(0);
            }
            T maxValue = (*this)[0];
            for (std::size_t i = 1; i < Size(); ++i)
            {
                maxValue = std::max(maxValue, (*this)[i]);
            }
            return maxValue;
        }

    private:
        Tensor(
            std::shared_ptr<std::vector<T>> storage,
            Shape shape,
            Shape strides,
            std::size_t offset,
            TensorBackend backend)
            : m_storage(std::move(storage))
            , m_shape(std::move(shape))
            , m_strides(std::move(strides))
            , m_offset(offset)
            , m_backend(backend)
        {
        }

        [[nodiscard]]
        static std::size_t ElementCount(const Shape& shape)
        {
            if (shape.empty())
            {
                return 0;
            }
            return std::accumulate(
                shape.begin(),
                shape.end(),
                std::size_t(1),
                [](std::size_t a, std::size_t b)
                {
                    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
                    {
                        throw std::overflow_error("Tensor shape product overflow.");
                    }
                    return a * b;
                });
        }

        [[nodiscard]]
        static Shape MakeContiguousStrides(const Shape& shape)
        {
            Shape strides(shape.size(), 1);
            if (shape.empty())
            {
                return strides;
            }
            for (std::size_t i = shape.size() - 1; i > 0; --i)
            {
                strides[i - 1] = strides[i] * shape[i];
            }
            return strides;
        }

        [[nodiscard]]
        std::size_t PhysicalOffset(std::span<const std::size_t> indices) const noexcept
        {
            assert(indices.size() == m_shape.size());
            std::size_t offset = m_offset;
            for (std::size_t axis = 0; axis < indices.size(); ++axis)
            {
                assert(indices[axis] < m_shape[axis]);
                offset += indices[axis] * m_strides[axis];
            }
            return offset;
        }

        [[nodiscard]]
        std::size_t PhysicalOffsetFromLinear(std::size_t linearIndex) const noexcept
        {
            if (IsContiguous())
            {
                return m_offset + linearIndex;
            }

            const Shape logicalStrides = MakeContiguousStrides(m_shape);
            std::size_t offset = m_offset;
            for (std::size_t axis = 0; axis < m_shape.size(); ++axis)
            {
                const std::size_t stride = logicalStrides[axis];
                const std::size_t index = linearIndex / stride;
                linearIndex %= stride;
                offset += index * m_strides[axis];
            }
            return offset;
        }

    };

    template<TensorScalar T, typename Fn>
    [[nodiscard]]
    Tensor<T> ElementwiseBinary(const Tensor<T>& lhs, const Tensor<T>& rhs, Fn&& op)
    {
        if (lhs.GetShape() != rhs.GetShape())
        {
            throw std::invalid_argument("Tensor elementwise operation shape mismatch.");
        }

        Tensor<T> result(lhs.GetShape());
        for (std::size_t i = 0; i < lhs.Size(); ++i)
        {
            result[i] = op(lhs[i], rhs[i]);
        }
        return result;
    }

    template<TensorScalar T>
    [[nodiscard]]
    Tensor<T> operator+(const Tensor<T>& lhs, const Tensor<T>& rhs)
    {
        return ElementwiseBinary(lhs, rhs, [](T a, T b) { return a + b; });
    }

    template<TensorScalar T>
    [[nodiscard]]
    Tensor<T> operator-(const Tensor<T>& lhs, const Tensor<T>& rhs)
    {
        return ElementwiseBinary(lhs, rhs, [](T a, T b) { return a - b; });
    }

    template<TensorScalar T>
    [[nodiscard]]
    Tensor<T> operator*(const Tensor<T>& lhs, const Tensor<T>& rhs)
    {
        return ElementwiseBinary(lhs, rhs, [](T a, T b) { return a * b; });
    }

    template<TensorScalar T>
    [[nodiscard]]
    Tensor<T> operator/(const Tensor<T>& lhs, const Tensor<T>& rhs)
    {
        return ElementwiseBinary(lhs, rhs, [](T a, T b) { return a / b; });
    }

    template<TensorScalar T>
    [[nodiscard]]
    Tensor<T> operator*(const Tensor<T>& tensor, T scalar)
    {
        return tensor.Map([scalar](T value) { return value * scalar; });
    }

    template<TensorScalar T>
    [[nodiscard]]
    Tensor<T> operator*(T scalar, const Tensor<T>& tensor)
    {
        return tensor * scalar;
    }

    template<TensorScalar T>
    [[nodiscard]]
    Tensor<T> operator+(const Tensor<T>& tensor, T scalar)
    {
        return tensor.Map([scalar](T value) { return value + scalar; });
    }

    /// Input: rank-2 tensors lhs[M, K] and rhs[K, N].
    /// Output: rank-2 tensor result[M, N].
    /// Task: provide the baseline GEMM kernel used by dense layers and
    /// attention blocks. The loop order is i-k-j for row-major locality.
    template<TensorScalar T>
    [[nodiscard]]
    Tensor<T> MatMul(const Tensor<T>& lhs, const Tensor<T>& rhs)
    {
        if (lhs.Rank() != 2 || rhs.Rank() != 2 || lhs.Dim(1) != rhs.Dim(0))
        {
            throw std::invalid_argument("MatMul expects [M,K] x [K,N] tensors.");
        }

        const std::size_t rows = lhs.Dim(0);
        const std::size_t inner = lhs.Dim(1);
        const std::size_t cols = rhs.Dim(1);

        using Accumulator = std::conditional_t<
            std::same_as<T, Float16> || std::same_as<T, BFloat16>, float, T>;
        Tensor<T> result({ rows, cols }, T(0));
        for (std::size_t i = 0; i < rows; ++i)
        {
            for (std::size_t j = 0; j < cols; ++j)
            {
                Accumulator sum = Accumulator(0);
                for (std::size_t k = 0; k < inner; ++k)
                    sum += static_cast<Accumulator>(lhs(i, k))
                        * static_cast<Accumulator>(rhs(k, j));
                result(i, j) = static_cast<T>(sum);
            }
        }
        return result;
    }

    /// Input: tensors [...,M,K] and [...,K,N] with identical batch dimensions.
    /// Output: tensor [...,M,N]. Low-precision inputs accumulate in Float32.
    /// Failure: broadcasting batch dimensions is deliberately not implicit.
    template<TensorScalar T>
    [[nodiscard]]
    Tensor<T> BatchedMatMul(const Tensor<T>& lhs, const Tensor<T>& rhs)
    {
        if (lhs.Rank() < 3 || rhs.Rank() != lhs.Rank()
            || lhs.Dim(lhs.Rank() - 1) != rhs.Dim(rhs.Rank() - 2))
            throw std::invalid_argument("BatchedMatMul expects [...,M,K] x [...,K,N].");
        for (std::size_t axis = 0; axis + 2 < lhs.Rank(); ++axis)
            if (lhs.Dim(axis) != rhs.Dim(axis))
                throw std::invalid_argument("BatchedMatMul batch dimensions must match.");
        const std::size_t rows = lhs.Dim(lhs.Rank() - 2);
        const std::size_t inner = lhs.Dim(lhs.Rank() - 1);
        const std::size_t columns = rhs.Dim(rhs.Rank() - 1);
        const std::size_t batches = lhs.Size() / (rows * inner);
        typename Tensor<T>::Shape shape = lhs.GetShape();
        shape[shape.size() - 1] = columns;
        Tensor<T> output(shape, T(0));
        using Accumulator = std::conditional_t<
            std::same_as<T, Float16> || std::same_as<T, BFloat16>, float, T>;
        for (std::size_t batch = 0; batch < batches; ++batch)
            for (std::size_t row = 0; row < rows; ++row)
                for (std::size_t column = 0; column < columns; ++column)
                {
                    Accumulator sum = Accumulator(0);
                    for (std::size_t k = 0; k < inner; ++k)
                        sum += static_cast<Accumulator>(
                            lhs[batch * rows * inner + row * inner + k])
                            * static_cast<Accumulator>(
                                rhs[batch * inner * columns + k * columns + column]);
                    output[batch * rows * columns + row * columns + column] =
                        static_cast<T>(sum);
                }
        return output;
    }

    /// Input: any TensorScalar source and destination type.
    /// Output: owning contiguous tensor with value-wise checked C++ conversion.
    template<TensorScalar To, TensorScalar From>
    [[nodiscard]]
    Tensor<To> TensorCast(const Tensor<From>& input)
    {
        Tensor<To> output(input.GetShape());
        for (std::size_t index = 0; index < input.Size(); ++index)
            output[index] = static_cast<To>(input[index]);
        return output;
    }

    /// Input: floating tensor and epsilon; normalization is over the final axis.
    /// Output: zero-mean, unit-variance groups in the same dtype. Statistics for
    /// Float16 and BFloat16 are accumulated in Float32.
    template<TensorFloating T>
    [[nodiscard]]
    Tensor<T> LayerNormLastDim(const Tensor<T>& input, float epsilon = 1e-5f)
    {
        if (input.Rank() == 0 || input.Dim(input.Rank() - 1) == 0 || !(epsilon > 0.0f))
            throw std::invalid_argument("LayerNormLastDim requires a final axis and positive epsilon.");
        const std::size_t width = input.Dim(input.Rank() - 1);
        const std::size_t groups = input.Size() / width;
        Tensor<T> output(input.GetShape());
        for (std::size_t group = 0; group < groups; ++group)
        {
            float mean = 0.0f;
            for (std::size_t column = 0; column < width; ++column)
                mean += static_cast<float>(input[group * width + column]);
            mean /= static_cast<float>(width);
            float variance = 0.0f;
            for (std::size_t column = 0; column < width; ++column)
            {
                const float centered =
                    static_cast<float>(input[group * width + column]) - mean;
                variance += centered * centered;
            }
            const float inverse = 1.0f
                / std::sqrt(variance / static_cast<float>(width) + epsilon);
            for (std::size_t column = 0; column < width; ++column)
                output[group * width + column] = static_cast<T>(
                    (static_cast<float>(input[group * width + column]) - mean) * inverse);
        }
        return output;
    }

    /// Input: floating tensor and epsilon; normalization is over the final axis.
    /// Output: unit root-mean-square groups without mean subtraction.
    template<TensorFloating T>
    [[nodiscard]]
    Tensor<T> RMSNormLastDim(const Tensor<T>& input, float epsilon = 1e-5f)
    {
        if (input.Rank() == 0 || input.Dim(input.Rank() - 1) == 0 || !(epsilon > 0.0f))
            throw std::invalid_argument("RMSNormLastDim requires a final axis and positive epsilon.");
        const std::size_t width = input.Dim(input.Rank() - 1);
        const std::size_t groups = input.Size() / width;
        Tensor<T> output(input.GetShape());
        for (std::size_t group = 0; group < groups; ++group)
        {
            float squares = 0.0f;
            for (std::size_t column = 0; column < width; ++column)
            {
                const float value = static_cast<float>(input[group * width + column]);
                squares += value * value;
            }
            const float inverse = 1.0f
                / std::sqrt(squares / static_cast<float>(width) + epsilon);
            for (std::size_t column = 0; column < width; ++column)
                output[group * width + column] =
                    static_cast<T>(static_cast<float>(input[group * width + column]) * inverse);
        }
        return output;
    }

    /// Input: tensor.
    /// Output: same-shaped tensor with max(x, 0).
    template<TensorFloating T>
    [[nodiscard]]
    Tensor<T> ReLU(const Tensor<T>& input)
    {
        return input.Map([](T value) { return std::max(T(0), value); });
    }

    /// Input: tensor.
    /// Output: same-shaped tensor with logistic activation.
    template<TensorFloating T>
    [[nodiscard]]
    Tensor<T> Sigmoid(const Tensor<T>& input)
    {
        return input.Map([](T value) { return T(1) / (T(1) + std::exp(-value)); });
    }

    /// Input: tensor.
    /// Output: same-shaped tensor with tanh activation.
    template<TensorFloating T>
    [[nodiscard]]
    Tensor<T> Tanh(const Tensor<T>& input)
    {
        return input.Map([](T value) { return std::tanh(value); });
    }

    /// Input: tensor with class logits in the final dimension.
    /// Output: probabilities normalized along the final dimension.
    /// Task: support classification heads and attention score normalization.
    template<TensorFloating T>
    [[nodiscard]]
    Tensor<T> SoftmaxLastDim(const Tensor<T>& logits)
    {
        if (logits.Rank() == 0)
        {
            throw std::invalid_argument("SoftmaxLastDim requires rank >= 1.");
        }

        Tensor<T> result(logits.GetShape());
        const std::size_t classes = logits.Dim(logits.Rank() - 1);

        if (classes == 0)
        {
            throw std::invalid_argument("SoftmaxLastDim requires non-zero final dimension.");
        }

        const std::size_t groups = logits.Size() / classes;

        for (std::size_t group = 0; group < groups; ++group)
        {
            const std::size_t base = group * classes;
            T maxValue = logits[base];
            for (std::size_t c = 1; c < classes; ++c)
            {
                maxValue = std::max(maxValue, logits[base + c]);
            }

            T sum = T(0);
            for (std::size_t c = 0; c < classes; ++c)
            {
                result[base + c] = std::exp(logits[base + c] - maxValue);
                sum += result[base + c];
            }

            const T invSum = T(1) / sum;
            for (std::size_t c = 0; c < classes; ++c)
            {
                result[base + c] *= invSum;
            }
        }

        return result;
    }

    /// Input: one-hot labels and predicted probabilities with the same shape.
    /// Output: mean cross-entropy loss.
    template<TensorFloating T>
    [[nodiscard]]
    T CrossEntropyMean(const Tensor<T>& labels, const Tensor<T>& probabilities, T epsilon = T(1e-7))
    {
        if (labels.GetShape() != probabilities.GetShape())
        {
            throw std::invalid_argument("CrossEntropyMean shape mismatch.");
        }

        T sum = T(0);
        for (std::size_t i = 0; i < labels.Size(); ++i)
        {
            if (labels[i] != T(0))
            {
                sum += labels[i] * -std::log(probabilities[i] + epsilon);
            }
        }

        if (labels.Empty())
        {
            return T(0);
        }

        const std::size_t sampleCount =
            labels.Rank() >= 2
                ? labels.Dim(0)
                : labels.Size();

        return sampleCount == 0
            ? T(0)
            : sum / static_cast<T>(sampleCount);
    }

    /// Input: predictions and labels with the same shape.
    /// Output: mean squared error.
    template<TensorFloating T>
    [[nodiscard]]
    T MeanSquaredError(const Tensor<T>& predictions, const Tensor<T>& labels)
    {
        if (predictions.GetShape() != labels.GetShape())
        {
            throw std::invalid_argument("MeanSquaredError shape mismatch.");
        }

        T sum = T(0);
        for (std::size_t i = 0; i < predictions.Size(); ++i)
        {
            const T diff = predictions[i] - labels[i];
            sum += diff * diff;
        }
        return predictions.Empty() ? T(0) : sum / static_cast<T>(predictions.Size());
    }

    /// Input: class labels and class count.
    /// Output: rank-2 one-hot tensor [labels.size(), classCount].
    template<TensorFloating T>
    [[nodiscard]]
    Tensor<T> OneHot(std::span<const std::size_t> labels, std::size_t classCount)
    {
        Tensor<T> result({ labels.size(), classCount }, T(0));
        for (std::size_t row = 0; row < labels.size(); ++row)
        {
            if (labels[row] < classCount)
            {
                result(row, labels[row]) = T(1);
            }
        }
        return result;
    }

    /// Input: tensor values and absolute clip bound.
    /// Output: values clipped in place to [-clipValue, +clipValue].
    /// Task: support gradient clipping for stable training.
    template<TensorFloating T>
    void ClipInPlace(Tensor<T>& tensor, T clipValue)
    {
        const T bound = std::abs(clipValue);
        tensor.ApplyInPlace([bound](T value)
        {
            return std::clamp(value, -bound, bound);
        });
    }

    /// Input: parameter tensor, gradient tensor, and learning rate.
    /// Output: parameter tensor updated by SGD.
    /// Task: provide the smallest optimizer primitive; momentum/Adam build on
    /// this same tensor mutation pattern.
    template<TensorFloating T>
    void SGDUpdate(Tensor<T>& parameters, const Tensor<T>& gradients, T learningRate)
    {
        if (parameters.GetShape() != gradients.GetShape())
        {
            throw std::invalid_argument("SGDUpdate shape mismatch.");
        }
        for (std::size_t i = 0; i < parameters.Size(); ++i)
        {
            parameters[i] -= learningRate * gradients[i];
        }
    }

    /// Input: matrix.
    /// Output: tensor copy with shape [rows, columns].
    /// Task: bridge existing linear-algebra routines into tensor pipelines.
    template<Arithmetic T>
    [[nodiscard]]
    Tensor<T> ToTensor(const DynamicMatrix<T>& matrix)
    {
        Tensor<T> result({ matrix.Rows(), matrix.Columns() });
        for (std::size_t i = 0; i < matrix.Size(); ++i)
        {
            result[i] = matrix[i];
        }
        return result;
    }

    /// Input: rank-2 tensor.
    /// Output: dynamic matrix copy.
    /// Task: allow tensor values to use KairoMath linear solvers and
    /// decompositions when needed.
    template<Arithmetic T>
    [[nodiscard]]
    DynamicMatrix<T> ToDynamicMatrix(const Tensor<T>& tensor)
    {
        if (tensor.Rank() != 2)
        {
            throw std::invalid_argument("ToDynamicMatrix requires a rank-2 tensor.");
        }
        DynamicMatrix<T> result(tensor.Dim(0), tensor.Dim(1));
        for (std::size_t i = 0; i < tensor.Size(); ++i)
        {
            result[i] = tensor[i];
        }
        return result;
    }

    /// Input: NHWC activations [batch,height,width,inputChannels] and OHWI
    /// filters [outputChannels,kernelHeight,kernelWidth,inputChannels].
    /// Output: valid-convolution NHWC activations. This is the CPU reference
    /// kernel; scheduler/SIMD/GPU backends must preserve these conventions.
    template<TensorFloating T>
    [[nodiscard]]
    Tensor<T> Conv2DValidNHWC(const Tensor<T>& input, const Tensor<T>& filters,
        std::size_t strideHeight = 1, std::size_t strideWidth = 1)
    {
        if (input.Rank() != 4 || filters.Rank() != 4 || strideHeight == 0 || strideWidth == 0
            || input.Dim(3) != filters.Dim(3) || filters.Dim(1) == 0 || filters.Dim(2) == 0
            || input.Dim(1) < filters.Dim(1) || input.Dim(2) < filters.Dim(2))
        {
            throw std::invalid_argument("Conv2DValidNHWC expects valid NHWC input, OHWI filters, and non-zero stride.");
        }
        const std::size_t outputHeight = (input.Dim(1) - filters.Dim(1)) / strideHeight + 1;
        const std::size_t outputWidth = (input.Dim(2) - filters.Dim(2)) / strideWidth + 1;
        Tensor<T> output({ input.Dim(0), outputHeight, outputWidth, filters.Dim(0) }, T(0));
        for (std::size_t batch = 0; batch < input.Dim(0); ++batch)
            for (std::size_t outputY = 0; outputY < outputHeight; ++outputY)
                for (std::size_t outputX = 0; outputX < outputWidth; ++outputX)
                    for (std::size_t outputChannel = 0; outputChannel < filters.Dim(0); ++outputChannel)
                    {
                        T sum = T(0);
                        for (std::size_t kernelY = 0; kernelY < filters.Dim(1); ++kernelY)
                            for (std::size_t kernelX = 0; kernelX < filters.Dim(2); ++kernelX)
                                for (std::size_t channel = 0; channel < input.Dim(3); ++channel)
                                {
                                    const std::size_t inY = outputY * strideHeight + kernelY;
                                    const std::size_t inX = outputX * strideWidth + kernelX;
                                    sum += input.At({ batch, inY, inX, channel })
                                        * filters.At({ outputChannel, kernelY, kernelX, channel });
                                }
                        output.At({ batch, outputY, outputX, outputChannel }) = sum;
                    }
        return output;
    }

    /// Input: NHWC activation and pooling window/stride dimensions.
    /// Output: valid max-pooled NHWC activation tensor.
    template<TensorFloating T>
    [[nodiscard]]
    Tensor<T> MaxPool2DValidNHWC(const Tensor<T>& input,
        std::size_t windowHeight, std::size_t windowWidth,
        std::size_t strideHeight = 1, std::size_t strideWidth = 1)
    {
        if (input.Rank() != 4 || windowHeight == 0 || windowWidth == 0 || strideHeight == 0 || strideWidth == 0
            || input.Dim(1) < windowHeight || input.Dim(2) < windowWidth)
        {
            throw std::invalid_argument("MaxPool2DValidNHWC expects valid NHWC input and non-zero window/stride.");
        }
        const std::size_t outputHeight = (input.Dim(1) - windowHeight) / strideHeight + 1;
        const std::size_t outputWidth = (input.Dim(2) - windowWidth) / strideWidth + 1;
        Tensor<T> output({ input.Dim(0), outputHeight, outputWidth, input.Dim(3) });
        for (std::size_t batch = 0; batch < input.Dim(0); ++batch)
            for (std::size_t outputY = 0; outputY < outputHeight; ++outputY)
                for (std::size_t outputX = 0; outputX < outputWidth; ++outputX)
                    for (std::size_t channel = 0; channel < input.Dim(3); ++channel)
                    {
                        T maximum = input.At({ batch, outputY * strideHeight, outputX * strideWidth, channel });
                        for (std::size_t windowY = 0; windowY < windowHeight; ++windowY)
                            for (std::size_t windowX = 0; windowX < windowWidth; ++windowX)
                                maximum = std::max(maximum, input.At({ batch, outputY * strideHeight + windowY, outputX * strideWidth + windowX, channel }));
                        output.At({ batch, outputY, outputX, channel }) = maximum;
                    }
        return output;
    }

    using Tensorf = Tensor<float>;
    using Tensord = Tensor<double>;
    using Tensori = Tensor<int>;
    using TensorFloat16 = Tensor<Float16>;
    using TensorBFloat16 = Tensor<BFloat16>;
    using TensorInt8 = Tensor<std::int8_t>;
    using TensorIndex = Tensor<std::int64_t>;

} // namespace kairo::foundation::math
