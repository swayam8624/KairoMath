import Kairo.Foundation.Math.Tensor;
import Kairo.Foundation.Math.TensorExecution;
import Kairo.Scheduler;

int main()
{
    kairo::scheduler::Scheduler scheduler({ .workerCount = 2, .defaultMinChunkSize = 1 });
    using kairo::foundation::math::Tensor;
    const Tensor<float> lhs({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });
    const Tensor<float> rhs({ 2, 3 }, { 6, 5, 4, 3, 2, 1 });
    const auto added = kairo::foundation::math::ParallelAdd(scheduler, lhs, rhs);
    if (added(0, 0) != 7.0f || added(1, 2) != 7.0f) return 1;

    const Tensor<float> right({ 3, 2 }, { 1, 0, 0, 1, 1, 1 });
    const auto product = kairo::foundation::math::ParallelMatMul(scheduler, lhs, right);
    return product(0, 0) == 4.0f && product(0, 1) == 5.0f
        && product(1, 0) == 10.0f && product(1, 1) == 11.0f ? 0 : 1;
}
