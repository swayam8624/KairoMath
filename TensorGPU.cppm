module;

#include <cstddef>
#include <span>
#include <stdexcept>

export module Kairo.Foundation.Math.TensorGPU;

import Kairo.Foundation.Math.Tensor;
import Kairo.GPU;

export namespace kairo::foundation::math
{
    /// Executes bounded Float32 Tensor kernels through a KairoGPU device.
    ///
    /// Input tensors remain host-owned and must be contiguous. Each operation
    /// performs explicit upload, dispatch, and readback so ownership and
    /// synchronization are deterministic. Persistent device residency belongs
    /// to the later Tensor storage/backend layer, not this integration bridge.
    class TensorGPUExecutor final
    {
    public:
        explicit TensorGPUExecutor(kairo::gpu::Device& device)
            : m_device(device)
        {
            if (!m_device.IsAvailable())
            {
                throw std::invalid_argument("TensorGPUExecutor requires an available GPU device.");
            }
        }

        /// Input: equal-shape contiguous Float32 tensors.
        /// Output: contiguous host Tensor tagged as GPU-produced.
        [[nodiscard]]
        Tensor<float> Add(const Tensor<float>& lhs, const Tensor<float>& rhs)
        {
            RequireEqualContiguousShapes(lhs, rhs);
            if (lhs.Empty())
            {
                throw std::invalid_argument("GPU tensor addition requires non-empty tensors.");
            }

            Tensor<float> output(lhs.GetShape());
            const std::size_t byteSize = lhs.Size() * sizeof(float);
            const auto lhsBuffer = CreateStorage(byteSize, "tensor-add-lhs");
            const auto rhsBuffer = CreateStorage(byteSize, "tensor-add-rhs");
            const auto outputBuffer = CreateStorage(byteSize, "tensor-add-output");
            m_device.Upload(lhsBuffer, std::as_bytes(std::span(lhs.Data(), lhs.Size())));
            m_device.Upload(rhsBuffer, std::as_bytes(std::span(rhs.Data(), rhs.Size())));
            m_device.VectorAddFloat(lhsBuffer, rhsBuffer, outputBuffer, lhs.Size());
            m_device.Download(outputBuffer, std::as_writable_bytes(std::span(output.Data(), output.Size())));
            output.SetBackend(TensorBackend::GPU);
            return output;
        }

        /// Input: contiguous row-major matrices [M,K] and [K,N].
        /// Output: contiguous host matrix [M,N] tagged as GPU-produced.
        [[nodiscard]]
        Tensor<float> MatMul(const Tensor<float>& lhs, const Tensor<float>& rhs)
        {
            if (lhs.Rank() != 2 || rhs.Rank() != 2 || lhs.Dim(1) != rhs.Dim(0)
                || lhs.Dim(0) == 0 || lhs.Dim(1) == 0 || rhs.Dim(1) == 0)
            {
                throw std::invalid_argument("GPU tensor matmul expects non-empty [M,K] x [K,N] tensors.");
            }
            RequireContiguous(lhs);
            RequireContiguous(rhs);

            Tensor<float> output({ lhs.Dim(0), rhs.Dim(1) });
            const auto lhsBuffer = CreateStorage(lhs.Size() * sizeof(float), "tensor-matmul-lhs");
            const auto rhsBuffer = CreateStorage(rhs.Size() * sizeof(float), "tensor-matmul-rhs");
            const auto outputBuffer = CreateStorage(output.Size() * sizeof(float), "tensor-matmul-output");
            m_device.Upload(lhsBuffer, std::as_bytes(std::span(lhs.Data(), lhs.Size())));
            m_device.Upload(rhsBuffer, std::as_bytes(std::span(rhs.Data(), rhs.Size())));
            m_device.MatMulFloat(
                lhsBuffer,
                rhsBuffer,
                outputBuffer,
                lhs.Dim(0),
                lhs.Dim(1),
                rhs.Dim(1));
            m_device.Download(outputBuffer, std::as_writable_bytes(std::span(output.Data(), output.Size())));
            output.SetBackend(TensorBackend::GPU);
            return output;
        }

    private:
        kairo::gpu::Device& m_device;

        [[nodiscard]]
        kairo::gpu::BufferHandle CreateStorage(std::size_t byteSize, const char* debugName)
        {
            return m_device.CreateBuffer({
                .byteSize = byteSize,
                .usage = kairo::gpu::BufferUsage::Storage,
                .debugName = debugName
            });
        }

        static void RequireContiguous(const Tensor<float>& tensor)
        {
            if (!tensor.IsContiguous())
            {
                throw std::invalid_argument("GPU Tensor operations require contiguous host storage.");
            }
        }

        static void RequireEqualContiguousShapes(const Tensor<float>& lhs, const Tensor<float>& rhs)
        {
            if (lhs.Rank() != rhs.Rank())
            {
                throw std::invalid_argument("GPU tensor addition requires equal shapes.");
            }
            for (std::size_t axis = 0; axis < lhs.Rank(); ++axis)
            {
                if (lhs.Dim(axis) != rhs.Dim(axis))
                {
                    throw std::invalid_argument("GPU tensor addition requires equal shapes.");
                }
            }
            RequireContiguous(lhs);
            RequireContiguous(rhs);
        }
    };
}
