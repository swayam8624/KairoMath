#include <cmath>
#include <cstddef>

import Kairo.Foundation.Math.Tensor;
import Kairo.Foundation.Math.TensorGPU;
import Kairo.GPU;

int main()
{
    using kairo::foundation::math::Tensor;
    using kairo::foundation::math::TensorBackend;
    kairo::gpu::Device device({ .backend = kairo::gpu::Backend::Metal, .debugName = "tensor-gpu-smoke" });
    kairo::foundation::math::TensorGPUExecutor gpu(device);

    const Tensor<float> lhs({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });
    const Tensor<float> rhs({ 2, 3 }, { 6, 5, 4, 3, 2, 1 });
    const auto added = gpu.Add(lhs, rhs);
    if (added.Backend() != TensorBackend::GPU || added(0, 0) != 7.0f || added(1, 2) != 7.0f) return 1;

    const Tensor<float> right({ 3, 2 }, { 1, 0, 0, 1, 1, 1 });
    const auto gpuProduct = gpu.MatMul(lhs, right);
    const auto cpuProduct = kairo::foundation::math::MatMul(lhs, right);
    if (gpuProduct.Backend() != TensorBackend::GPU
        || gpuProduct.Rank() != 2
        || gpuProduct.Dim(0) != cpuProduct.Dim(0)
        || gpuProduct.Dim(1) != cpuProduct.Dim(1)) return 2;
    for (std::size_t index = 0; index < gpuProduct.Size(); ++index)
    {
        if (std::abs(gpuProduct[index] - cpuProduct[index]) > 1e-5f) return 3;
    }
    return 0;
}
