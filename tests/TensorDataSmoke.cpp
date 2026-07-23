#include <algorithm>
#include <cstddef>
#include <vector>

import Kairo.Foundation.Math.Tensor;
import Kairo.Foundation.Math.TensorData;

int main()
{
    using namespace kairo::foundation::math;
    TensorDataset dataset(
        Tensor<float>({ 7, 2 }, {
            0, 100, 1, 101, 2, 102, 3, 103, 4, 104, 5, 105, 6, 106
        }),
        Tensor<float>({ 7, 1 }, { 10, 11, 12, 13, 14, 15, 16 }));

    const TensorDataset first = dataset.Shuffled(1234);
    const TensorDataset second = dataset.Shuffled(1234);
    if (first.Indices() != second.Indices() || first.Indices() == dataset.Indices()) return 1;

    auto [training, validation] = dataset.Split(2.0f / 7.0f, 77);
    if (training.Size() != 5 || validation.Size() != 2) return 2;
    std::vector<std::size_t> unionIndices = training.Indices();
    unionIndices.insert(
        unionIndices.end(), validation.Indices().begin(), validation.Indices().end());
    std::sort(unionIndices.begin(), unionIndices.end());
    if (unionIndices != dataset.Indices()) return 3;

    TensorBatchLoader loader(first, 3);
    std::vector<std::size_t> observed;
    std::size_t batchCount = 0;
    while (auto batch = loader.Next())
    {
        ++batchCount;
        observed.insert(
            observed.end(), batch->sourceIndices.begin(), batch->sourceIndices.end());
        if (batch->samples.Dim(0) != batch->sourceIndices.size()
            || batch->labels.Dim(0) != batch->sourceIndices.size()) return 4;
        for (std::size_t row = 0; row < batch->sourceIndices.size(); ++row)
        {
            const float source = static_cast<float>(batch->sourceIndices[row]);
            if (batch->samples(row, 0) != source || batch->labels(row, 0) != source + 10.0f)
                return 5;
        }
    }
    if (batchCount != 3 || observed != first.Indices()) return 6;

    TensorBatchLoader dropping(dataset, 3, true);
    std::size_t droppedCount = 0;
    while (auto batch = dropping.Next()) droppedCount += batch->samples.Dim(0);
    if (droppedCount != 6) return 7;

    TensorPrefetchLoader prefetch(first, 2, 2);
    observed.clear();
    while (auto batch = prefetch.Next())
        observed.insert(
            observed.end(), batch->sourceIndices.begin(), batch->sourceIndices.end());
    return observed == first.Indices() ? 0 : 8;
}
