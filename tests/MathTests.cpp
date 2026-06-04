#include <catch2/catch_test_macros.hpp>
#include <cmath>

import Kairo.Foundation.Math.Vector;
import Kairo.Foundation.Math.Matrix;
import Kairo.Foundation.Math.Quaternion;
import Kairo.Foundation.Math.Transform;

using namespace kairo::foundation::math;

TEST_CASE("Vector Project with non-unit axes", "[Vector]")
{
    SECTION("2D Projection on non-unit axis")
    {
        Vec2f v(2.0f, 3.0f);
        Vec2f onto(4.0f, 0.0f); // Non-unit axis along X
        Vec2f proj = Project(v, onto);
        
        REQUIRE(NearlyEqual(proj.x, 2.0f));
        REQUIRE(NearlyEqual(proj.y, 0.0f));
    }

    SECTION("3D Projection on non-unit axis")
    {
        Vec3f v(1.0f, 2.0f, 3.0f);
        Vec3f onto(0.0f, 2.0f, 0.0f); // Non-unit axis along Y
        Vec3f proj = Project(v, onto);
        
        REQUIRE(NearlyEqual(proj.x, 0.0f));
        REQUIRE(NearlyEqual(proj.y, 2.0f));
        REQUIRE(NearlyEqual(proj.z, 0.0f));
    }

    SECTION("Zero-length projection axis returns zero")
    {
        Vec3f v(1.0f, 2.0f, 3.0f);
        Vec3f onto(0.0f, 0.0f, 0.0f);
        Vec3f proj = Project(v, onto);
        
        REQUIRE(proj.x == 0.0f);
        REQUIRE(proj.y == 0.0f);
        REQUIRE(proj.z == 0.0f);
    }
}

TEST_CASE("Vector Refract edge cases", "[Vector]")
{
    SECTION("Normal incidence no refraction deflection")
    {
        Vec3f incident(0.0f, -1.0f, 0.0f);
        Vec3f normal(0.0f, 1.0f, 0.0f);
        float eta = 1.0f; // air to air
        Vec3f refracted = Refract(incident, normal, eta);
        
        REQUIRE(NearlyEqual(refracted, incident));
    }

    SECTION("Angled refraction")
    {
        // Incident ray at 45 degrees
        Vec3f incident = Normalize(Vec3f(1.0f, -1.0f, 0.0f));
        Vec3f normal(0.0f, 1.0f, 0.0f);
        float eta = 1.0f / 1.5f; // air to glass (refraction index ~1.5)
        Vec3f refracted = Refract(incident, normal, eta);
        
        // Output direction should be bent closer to normal
        REQUIRE(refracted.x > 0.0f);
        REQUIRE(refracted.y < 0.0f);
        REQUIRE(std::abs(refracted.x) < std::abs(incident.x));
    }

    SECTION("Total Internal Reflection returns zero vector")
    {
        // Moving from glass (1.5) to air (1.0) at steep angle (e.g. 60 degrees)
        // Critical angle is asin(1/1.5) = ~41.8 degrees
        // 60 degrees is greater than critical angle, should TIR
        float angle = 60.0f * (3.14159265f / 180.0f);
        Vec3f incident(std::sin(angle), -std::cos(angle), 0.0f);
        Vec3f normal(0.0f, 1.0f, 0.0f);
        float eta = 1.5f / 1.0f; // glass to air
        
        Vec3f refracted = Refract(incident, normal, eta);
        
        REQUIRE(refracted.x == 0.0f);
        REQUIRE(refracted.y == 0.0f);
        REQUIRE(refracted.z == 0.0f);
    }
}

TEST_CASE("Matrix4 Inverse correctness", "[Matrix]")
{
    SECTION("Translation-only matrix inverse")
    {
        Mat4f translation = MakeTranslation(Vec3f(10.0f, -5.0f, 2.5f));
        Mat4f inv = Inverse(translation);
        Mat4f identity = translation * inv;
        
        REQUIRE(IsIdentity(identity));
        REQUIRE(NearlyEqual(inv(0, 3), -10.0f));
        REQUIRE(NearlyEqual(inv(1, 3), 5.0f));
        REQUIRE(NearlyEqual(inv(2, 3), -2.5f));
    }

    SECTION("Rotation-only matrix inverse")
    {
        // Rotate 90 degrees around Y
        Mat4f rotation = MakeRotationY(3.14159265f / 2.0f);
        Mat4f inv = Inverse(rotation);
        Mat4f identity = rotation * inv;
        
        REQUIRE(IsIdentity(identity));
        // Inverse of rotation is transpose
        REQUIRE(NearlyEqual(inv(0, 2), -1.0f));
        REQUIRE(NearlyEqual(inv(2, 0), 1.0f));
    }

    SECTION("Scale-only matrix inverse")
    {
        Mat4f scale = MakeScale(Vec3f(2.0f, 0.5f, 4.0f));
        Mat4f inv = Inverse(scale);
        Mat4f identity = scale * inv;
        
        REQUIRE(IsIdentity(identity));
        REQUIRE(NearlyEqual(inv(0, 0), 0.5f));
        REQUIRE(NearlyEqual(inv(1, 1), 2.0f));
        REQUIRE(NearlyEqual(inv(2, 2), 0.25f));
    }

    SECTION("Combined TRS matrix inverse")
    {
        Mat4f translation = MakeTranslation(Vec3f(3.0f, 4.0f, 5.0f));
        Mat4f rotation = MakeRotationZ(0.5f);
        Mat4f scale = MakeScale(Vec3f(2.0f, 2.0f, 2.0f));
        Mat4f trs = translation * rotation * scale;
        
        Mat4f inv = Inverse(trs);
        Mat4f identity = trs * inv;
        
        REQUIRE(IsIdentity(identity));
    }

    SECTION("LookAt view matrix inverse")
    {
        Vec3f eye(0.0f, 5.0f, 10.0f);
        Vec3f target(0.0f, 0.0f, 0.0f);
        Vec3f up(0.0f, 1.0f, 0.0f);
        
        Mat4f view = LookAt(eye, target, up);
        Mat4f inv = Inverse(view);
        Mat4f identity = view * inv;
        
        REQUIRE(IsIdentity(identity));
    }
}

TEST_CASE("Transform Inverse and WorldToLocal", "[Transform]")
{
    SECTION("Uniform scale Transform Inverse")
    {
        Transformf t(
            Vec3f(2.0f, 3.0f, 4.0f),
            RotationAroundY(1.0f),
            Vec3f(2.0f, 2.0f, 2.0f) // Uniform scale
        );
        
        Transformf inv = Inverse(t);
        Transformf identity = t * inv;
        Transformf inv_inv = Inverse(inv);
        
        REQUIRE(identity.IsIdentity());
        REQUIRE(NearlyEqual(inv_inv.Translation, t.Translation));
        REQUIRE(NearlyEqual(inv_inv.Rotation, t.Rotation));
        REQUIRE(NearlyEqual(inv_inv.Scale, t.Scale));
    }

    SECTION("Non-uniform scale WorldToLocal point transformation")
    {
        Transformf t(
            Vec3f(5.0f, -2.0f, 1.0f),
            RotationAroundX(0.7f) * RotationAroundZ(0.3f),
            Vec3f(1.5f, 0.5f, 3.0f) // Non-uniform scale
        );
        
        Vec3f localPoint(1.0f, 2.0f, 3.0f);
        Vec3f worldPoint = TransformPoint(t, localPoint);
        
        // Convert back using WorldToLocal
        Vec3f reconstructedPoint = WorldToLocal(t, worldPoint);
        
        REQUIRE(NearlyEqual(reconstructedPoint, localPoint));
    }
}
