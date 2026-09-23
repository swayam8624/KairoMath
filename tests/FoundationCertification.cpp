#include <catch2/catch_test_macros.hpp>

#include <cmath>

import Kairo.Foundation.Math;

using namespace kairo::foundation::math;

TEST_CASE("Certification: dynamic inversion remains stable across deterministic conditioned matrices", "[KairoMath][Certification]")
{
    for (int index = 1; index <= 256; ++index)
    {
        const double t = static_cast<double>(index) / 257.0;
        DynamicMatrix<double> matrix(3, 3, {
            3.0 + t, 0.15 + 0.03 * t, 0.07,
            0.15 + 0.03 * t, 2.5 + 0.5 * t, 0.11,
            0.07, 0.11, 2.0 + 0.25 * t
        });

        const DynamicMatrix<double> inverse = Inverse(matrix);
        REQUIRE(IsIdentity(matrix * inverse, 1.0e-9));

        const double condition = ConditionNumber(matrix);
        REQUIRE(std::isfinite(condition));
        REQUIRE(condition >= 1.0);
    }
}

TEST_CASE("Certification: transform world/local round trips remain bounded", "[KairoMath][Certification]")
{
    for (int index = 0; index < 512; ++index)
    {
        const float t = static_cast<float>(index) / 511.0f;
        const Transformf transform(
            Vec3f(10.0f * t - 5.0f, 3.0f - 6.0f * t, 2.0f * t),
            RotationAroundX(0.5f * t) *
                RotationAroundY(-0.75f * t) *
                RotationAroundZ(0.25f * t),
            Vec3f(0.5f + t, 1.25f - 0.5f * t, 2.0f + 0.25f * t));

        const Vec3f local(
            4.0f * t - 2.0f,
            std::sin(t * 3.0f),
            std::cos(t * 5.0f));

        const Vec3f world = TransformPoint(transform, local);
        const Vec3f reconstructed = WorldToLocal(transform, world);
        REQUIRE(NearlyEqual(reconstructed, local, 2.0e-4f));
    }
}
