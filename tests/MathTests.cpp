#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

import Kairo.Foundation.Math.Vector;
import Kairo.Foundation.Math.Matrix;
import Kairo.Foundation.Math.Quaternion;
import Kairo.Foundation.Math.Transform;
import Kairo.Foundation.Math.DynamicMatrix;
import Kairo.Foundation.Math.Tensor;
import Kairo.Foundation.Math.LinearAlgebra.LinearSolve;
import Kairo.Foundation.Math.LinearAlgebra.Decomposition;
import Kairo.Foundation.Math.LinearAlgebra.Eigen;
import Kairo.Foundation.Math.LinearAlgebra.SVD;
import Kairo.Foundation.Math.LinearAlgebra.Statistics;
import Kairo.Foundation.Math.LinearAlgebra.MatrixFunctions;
import Kairo.Foundation.Math.Optimization;
import Kairo.Foundation.Math.Probability;

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

TEST_CASE("Matrix2 and New Matrix/Vector methods", "[Matrix][Vector]")
{
    SECTION("Matrix2 Basics")
    {
        Mat2f m(1.0f, 2.0f, 3.0f, 4.0f);
        REQUIRE(NearlyEqual(m(0, 0), 1.0f));
        REQUIRE(NearlyEqual(m(0, 1), 2.0f));
        REQUIRE(NearlyEqual(m(1, 0), 3.0f));
        REQUIRE(NearlyEqual(m(1, 1), 4.0f));

        Mat2f identity = Mat2f::Identity();
        REQUIRE(NearlyEqual(identity(0, 0), 1.0f));
        REQUIRE(NearlyEqual(identity(1, 1), 1.0f));
        REQUIRE(NearlyEqual(identity(0, 1), 0.0f));

        Mat2f prod = m * identity;
        REQUIRE(NearlyEqual(prod, m));
        
        Vec2f v(1.0f, 2.0f);
        Vec2f res = m * v; // {1*1 + 2*2, 3*1 + 4*2} = {5, 11}
        REQUIRE(NearlyEqual(res.x, 5.0f));
        REQUIRE(NearlyEqual(res.y, 11.0f));
    }

    SECTION("Trace and Diagonal")
    {
        Mat2f m2(1.0f, 2.0f, 3.0f, 4.0f);
        REQUIRE(NearlyEqual(Trace(m2), 5.0f));
        REQUIRE(NearlyEqual(Diagonal(m2), Vec2f(1.0f, 4.0f)));

        Mat3f m3(1.0f, 0.0f, 0.0f, 0.0f, 5.0f, 0.0f, 0.0f, 0.0f, 9.0f);
        REQUIRE(NearlyEqual(Trace(m3), 15.0f));
        REQUIRE(NearlyEqual(Diagonal(m3), Vec3f(1.0f, 5.0f, 9.0f)));

        Mat4f m4(2.0f); // diagonal 2.0f
        REQUIRE(NearlyEqual(Trace(m4), 8.0f));
        REQUIRE(NearlyEqual(Diagonal(m4), Vec4f(2.0f, 2.0f, 2.0f, 2.0f)));
    }

    SECTION("OuterProduct")
    {
        Vec2f u2(1.0f, 2.0f);
        Vec2f v2(3.0f, 4.0f);
        Mat2f op2 = OuterProduct(u2, v2);
        REQUIRE(NearlyEqual(op2, Mat2f(3.0f, 4.0f, 6.0f, 8.0f)));

        Vec3f u3(1.0f, 2.0f, 3.0f);
        Vec3f v3(4.0f, 5.0f, 6.0f);
        Mat3f op3 = OuterProduct(u3, v3);
        REQUIRE(NearlyEqual(op3(0, 0), 4.0f));
        REQUIRE(NearlyEqual(op3(1, 1), 10.0f));
        REQUIRE(NearlyEqual(op3(2, 2), 18.0f));

        Vec4f u4(1.0f, 2.0f, 3.0f, 4.0f);
        Vec4f v4(5.0f, 6.0f, 7.0f, 8.0f);
        Mat4f op4 = OuterProduct(u4, v4);
        REQUIRE(NearlyEqual(op4(0, 0), 5.0f));
        REQUIRE(NearlyEqual(op4(3, 3), 32.0f));
    }

    SECTION("Cofactor, Adjugate, Determinant, Inverse")
    {
        Mat2f m2(1.0f, 2.0f, 3.0f, 4.0f);
        REQUIRE(NearlyEqual(Determinant(m2), -2.0f));
        
        Mat2f cof2 = Cofactor(m2);
        REQUIRE(NearlyEqual(cof2, Mat2f(4.0f, -3.0f, -2.0f, 1.0f)));

        Mat2f adj2 = Adjugate(m2);
        REQUIRE(NearlyEqual(adj2, Mat2f(4.0f, -2.0f, -3.0f, 1.0f)));

        Mat2f inv2 = Inverse(m2);
        Mat2f expectedInv2 = adj2 / -2.0f;
        REQUIRE(NearlyEqual(inv2, expectedInv2));
        REQUIRE(NearlyEqual(m2 * inv2, Mat2f::Identity()));

        Mat3f m3(1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f, 4.0f, 0.0f, 5.0f);
        REQUIRE(NearlyEqual(Cofactor(m3)(0, 0), 15.0f));
        REQUIRE(NearlyEqual(Adjugate(m3)(0, 0), 15.0f));
        Mat3f inv3 = Inverse(m3);
        REQUIRE(NearlyEqual(m3 * inv3, Mat3f::Identity()));
    }

    SECTION("Symmetric and Orthogonal")
    {
        Mat2f sym2(1.0f, 2.0f, 2.0f, 4.0f);
        REQUIRE(IsSymmetric(sym2));
        REQUIRE(!IsOrthogonal(sym2));

        Mat2f rot2(0.0f, 1.0f, -1.0f, 0.0f); // 90 degree rotation
        REQUIRE(IsOrthogonal(rot2));
        REQUIRE(!IsSymmetric(rot2));
    }

    SECTION("Reject, FaceForward, Orthogonal")
    {
        Vec3f v(1.0f, 2.0f, 3.0f);
        Vec3f onto(0.0f, 1.0f, 0.0f);
        Vec3f proj = Project(v, onto);
        Vec3f rej = Reject(v, onto);
        REQUIRE(NearlyEqual(proj, Vec3f(0.0f, 2.0f, 0.0f)));
        REQUIRE(NearlyEqual(rej, Vec3f(1.0f, 0.0f, 3.0f)));
        REQUIRE(NearlyEqual(Dot(rej, onto), 0.0f));

        Vec3f n(0.0f, 1.0f, 0.0f);
        Vec3f i(1.0f, -1.0f, 0.0f);
        Vec3f r(0.0f, 1.0f, 0.0f);
        Vec3f ff = FaceForward(n, i, r);
        REQUIRE(NearlyEqual(ff, n));

        Vec2f v2(2.0f, 3.0f);
        Vec2f orth2 = Orthogonal(v2);
        REQUIRE(NearlyEqual(Dot(v2, orth2), 0.0f));

        Vec3f v3(2.0f, 3.0f, 4.0f);
        Vec3f orth3 = Orthogonal(v3);
        REQUIRE(NearlyEqual(Dot(v3, orth3), 0.0f));
    }
}

TEST_CASE("DynamicMatrix Core Features", "[DynamicMatrix]")
{
    SECTION("Constructors and Access")
    {
        DynamicMatrix<float> m1;
        REQUIRE(m1.Empty());

        DynamicMatrix<float> m2(2, 3);
        REQUIRE(m2.Rows() == 2);
        REQUIRE(m2.Columns() == 3);
        REQUIRE(m2.Size() == 6);
        REQUIRE(!m2.Empty());
        REQUIRE(m2(0, 0) == 0.0f);

        DynamicMatrix<float> m3(2, 2, 5.0f);
        REQUIRE(m3(0, 0) == 5.0f);
        REQUIRE(m3(1, 1) == 5.0f);

        DynamicMatrix<float> m4(2, 2, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
        REQUIRE(m4(0, 0) == 1.0f);
        REQUIRE(m4(0, 1) == 2.0f);
        REQUIRE(m4(1, 0) == 3.0f);
        REQUIRE(m4(1, 1) == 4.0f);

        DynamicMatrix<float> m5(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
        REQUIRE(m5(0, 0) == 1.0f);
        REQUIRE(m5(1, 1) == 4.0f);
        REQUIRE_THROWS_AS(
            DynamicMatrix<float>(2, 2, {1.0f, 2.0f, 3.0f}),
            std::invalid_argument);

        m5.At(1, 0) = 7.0f;
        REQUIRE(m5.At(1, 0) == 7.0f);
        REQUIRE_THROWS_AS(m5.At(2, 0), std::out_of_range);
        REQUIRE_THROWS_AS(m5.At(0, 2), std::out_of_range);

        float* rowData = m5.RowData(1);
        REQUIRE(rowData[0] == 7.0f);
        REQUIRE(rowData[1] == 4.0f);

        auto row = m5.RowSpan(0);
        REQUIRE(row.size() == 2);
        row[1] = 9.0f;
        REQUIRE(m5(0, 1) == 9.0f);
        REQUIRE_THROWS_AS(m5.RowSpan(2), std::out_of_range);
    }

    SECTION("Move operations leave source empty")
    {
        DynamicMatrix<float> source(3, 3, 2.0f);
        DynamicMatrix<float> moved(std::move(source));

        REQUIRE(moved.Rows() == 3);
        REQUIRE(moved.Columns() == 3);
        REQUIRE(moved.Size() == 9);
        REQUIRE(source.Rows() == 0);
        REQUIRE(source.Columns() == 0);
        REQUIRE(source.Size() == 0);
        REQUIRE(source.Empty());

        DynamicMatrix<float> assigned(1, 1, 1.0f);
        assigned = std::move(moved);

        REQUIRE(assigned.Rows() == 3);
        REQUIRE(assigned.Columns() == 3);
        REQUIRE(assigned.Size() == 9);
        REQUIRE(moved.Rows() == 0);
        REQUIRE(moved.Columns() == 0);
        REQUIRE(moved.Size() == 0);
        REQUIRE(moved.Empty());
    }

    SECTION("Factories and Operators")
    {
        auto zero = DynamicMatrix<float>::Zero(2, 3);
        REQUIRE(zero(0, 0) == 0.0f);

        auto identity = DynamicMatrix<float>::Identity(3);
        REQUIRE(identity(0, 0) == 1.0f);
        REQUIRE(identity(0, 1) == 0.0f);
        REQUIRE(identity(1, 1) == 1.0f);

        auto diag = DynamicMatrix<float>::Diagonal({2.0f, 3.0f});
        REQUIRE(diag(0, 0) == 2.0f);
        REQUIRE(diag(1, 1) == 3.0f);
        REQUIRE(diag(0, 1) == 0.0f);

        DynamicMatrix<float> a(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
        DynamicMatrix<float> b(2, 2, {2.0f, 0.0f, 1.0f, -1.0f});

        auto sum = a + b;
        REQUIRE(sum(0, 0) == 3.0f);
        REQUIRE(sum(1, 1) == 3.0f);

        auto diff = a - b;
        REQUIRE(diff(0, 0) == -1.0f);
        REQUIRE(diff(1, 1) == 5.0f);

        auto scaled = a * 2.0f;
        REQUIRE(scaled(0, 0) == 2.0f);

        auto prod = a * b; // {1*2+2*1, 1*0-2*1, 3*2+4*1, 3*0-4*1} = {4, -2, 10, -4}
        REQUIRE(prod(0, 0) == 4.0f);
        REQUIRE(prod(0, 1) == -2.0f);
        REQUIRE(prod(1, 0) == 10.0f);
        REQUIRE(prod(1, 1) == -4.0f);
    }

    SECTION("Matrix operations")
    {
        DynamicMatrix<float> a(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
        auto t = a.Transpose();
        REQUIRE(t.Rows() == 3);
        REQUIRE(t.Columns() == 2);
        REQUIRE(t(0, 0) == 1.0f);
        REQUIRE(t(1, 0) == 2.0f);
        REQUIRE(t(0, 1) == 4.0f);

        DynamicMatrix<float> square(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
        REQUIRE(square.Trace() == 5.0f);
        REQUIRE(NearlyEqual(square.FrobeniusNorm(), std::sqrt(1.0f + 4.0f + 9.0f + 16.0f)));

        square.SwapRows(0, 1);
        REQUIRE(square(0, 0) == 3.0f);
        REQUIRE(square(1, 0) == 1.0f);

        square.SwapColumns(0, 1);
        REQUIRE(square(0, 0) == 4.0f);
        REQUIRE(square(1, 0) == 2.0f);
    }

    SECTION("NearlyEqual uses relative tolerance")
    {
        DynamicMatrix<double> largeA(1, 1, {1.0e9});
        DynamicMatrix<double> largeB(1, 1, {1.0e9 + 8.0});
        REQUIRE(NearlyEqual(largeA, largeB, 1.0e-8));

        DynamicMatrix<double> smallA(1, 1, {1.0e-12});
        DynamicMatrix<double> smallB(1, 1, {2.0e-12});
        REQUIRE(NearlyEqual(smallA, smallB, 1.0e-11));

        DynamicMatrix<double> farA(1, 1, {1.0e9});
        DynamicMatrix<double> farB(1, 1, {1.0e9 + 1000.0});
        REQUIRE_FALSE(NearlyEqual(farA, farB, 1.0e-8));
    }
}

TEST_CASE("LinearSolve Phase 1", "[LinearSolve]")
{
    SECTION("Forward and Backward Substitution")
    {
        // Lower triangular L
        DynamicMatrix<float> L(3, 3, {
            2.0f, 0.0f, 0.0f,
            1.0f, 3.0f, 0.0f,
            1.0f, -1.0f, 2.0f
        });
        std::vector<float> b_low = {4.0f, 5.0f, 6.0f};
        // 2*x0 = 4 => x0 = 2
        // 1*2 + 3*x1 = 5 => 3*x1 = 3 => x1 = 1
        // 1*2 - 1*1 + 2*x2 = 6 => 1 + 2*x2 = 6 => 2*x2 = 5 => x2 = 2.5
        auto x_low = ForwardSubstitution(L, b_low);
        REQUIRE(std::abs(x_low[0] - 2.0f) < 1e-5f);
        REQUIRE(std::abs(x_low[1] - 1.0f) < 1e-5f);
        REQUIRE(std::abs(x_low[2] - 2.5f) < 1e-5f);

        // Upper triangular U
        DynamicMatrix<float> U(3, 3, {
            2.0f, 1.0f, -1.0f,
            0.0f, 3.0f, 2.0f,
            0.0f, 0.0f, 4.0f
        });
        std::vector<float> b_up = {4.0f, 5.0f, 8.0f};
        // 4*x2 = 8 => x2 = 2
        // 3*x1 + 2*2 = 5 => 3*x1 = 1 => x1 = 1/3
        // 2*x0 + 1/3 - 2 = 4 => 2*x0 = 17/3 => x0 = 17/6 = 2.83333
        auto x_up = BackwardSubstitution(U, b_up);
        REQUIRE(std::abs(x_up[2] - 2.0f) < 1e-5f);
        REQUIRE(std::abs(x_up[1] - 0.33333f) < 1e-4f);
        REQUIRE(std::abs(x_up[0] - 2.83333f) < 1e-4f);
    }

    SECTION("Gaussian and Gauss-Jordan Elimination")
    {
        DynamicMatrix<float> A(3, 3, {
            2.0f, 1.0f, -1.0f,
            -3.0f, -1.0f, 2.0f,
            -2.0f, 1.0f, 2.0f
        });
        std::vector<float> b = {8.0f, -11.0f, -3.0f};

        auto x1 = GaussianElimination(A, b);
        auto x2 = GaussJordanElimination(A, b);
        auto x3 = LinearSolve(A, b);

        REQUIRE(std::abs(x1[0] - 2.0f) < 1e-4f);
        REQUIRE(std::abs(x1[1] - 3.0f) < 1e-4f);
        REQUIRE(std::abs(x1[2] - -1.0f) < 1e-4f);

        for (std::size_t i = 0; i < 3; ++i)
        {
            REQUIRE(std::abs(x2[i] - x1[i]) < 1e-4f);
            REQUIRE(std::abs(x3[i] - x1[i]) < 1e-4f);
        }
    }

    SECTION("REF and RREF")
    {
        DynamicMatrix<float> A(3, 3, {
            1.0f, 2.0f, 3.0f,
            2.0f, 4.0f, 6.0f,
            3.0f, 6.0f, 10.0f
        });
        auto ref = RowEchelonForm(A);
        auto rref = ReducedRowEchelonForm(A);

        REQUIRE(ref(1, 0) == 0.0f);
        REQUIRE(ref(1, 1) == 0.0f);
        REQUIRE(rref(0, 0) == 1.0f);
        REQUIRE(rref(0, 1) == 2.0f);
    }
}

TEST_CASE("Decomposition Phase 2", "[Decomposition]")
{
    SECTION("LU and LUP")
    {
        DynamicMatrix<float> A(3, 3, {
            1.0f, 2.0f, 4.0f,
            3.0f, 8.0f, 14.0f,
            2.0f, 6.0f, 13.0f
        });

        auto lu = LU(A);
        auto recon = lu.L * lu.U;
        REQUIRE(NearlyEqual(recon, A));

        // LUP with a matrix that requires pivoting (zero on diagonal)
        DynamicMatrix<float> B(3, 3, {
            0.0f, 1.0f, 2.0f,
            1.0f, 2.0f, 3.0f,
            3.0f, 1.0f, 1.0f
        });
        auto lup = LUP(B);
        auto reconP = lup.L * lup.U;

        // Apply permutation to original matrix B
        DynamicMatrix<float> PB(3, 3);
        for (std::size_t r = 0; r < 3; ++r)
        {
            for (std::size_t c = 0; c < 3; ++c)
            {
                PB(r, c) = B(lup.P[r], c);
            }
        }
        REQUIRE(NearlyEqual(reconP, PB));
    }

    SECTION("QR Decomposition")
    {
        DynamicMatrix<float> A(3, 3, {
            12.0f, -51.0f, 4.0f,
            6.0f, 167.0f, -68.0f,
            -4.0f, 24.0f, -41.0f
        });

        auto qr = QR(A);
        auto recon = qr.Q * qr.R;
        REQUIRE(NearlyEqual(recon, A, 1e-2f));

        // Verify Q is orthogonal
        auto QTQ = qr.Q.Transpose() * qr.Q;
        REQUIRE(IsIdentity(QTQ, 1e-4f));
    }

    SECTION("Cholesky and LDLT")
    {
        // Symmetric positive-definite matrix
        DynamicMatrix<float> A(3, 3, {
            4.0f, 12.0f, -16.0f,
            12.0f, 37.0f, -43.0f,
            -16.0f, -43.0f, 98.0f
        });

        auto L = Cholesky(A);
        auto recon = L * L.Transpose();
        REQUIRE(NearlyEqual(recon, A));

        auto ldlt = LDLT(A);
        auto D = DynamicMatrix<float>::Diagonal(ldlt.D);
        auto reconLDLT = ldlt.L * D * ldlt.L.Transpose();
        REQUIRE(NearlyEqual(reconLDLT, A));
    }
}

TEST_CASE("Eigen Phase 3", "[Eigen]")
{
    SECTION("Power Iterations")
    {
        DynamicMatrix<float> A(2, 2, {
            2.0f, 1.0f,
            1.0f, 2.0f
        });

        auto pi = PowerIteration(A);
        REQUIRE(std::abs(pi.eigenvalue - 3.0f) < 1e-3f); // Dominant eigenvalue is 3

        auto ipi = InversePowerIteration(A);
        REQUIRE(std::abs(ipi.eigenvalue - 1.0f) < 1e-3f); // Smallest eigenvalue is 1
    }

    SECTION("QR Eigenvalues & Eigenvectors")
    {
        DynamicMatrix<float> A(3, 3, {
            3.0f, 2.0f, 4.0f,
            2.0f, 0.0f, 2.0f,
            4.0f, 2.0f, 3.0f
        });

        auto evals = QREigenValues(A);
        std::sort(evals.begin(), evals.end());

        // Eigenvalues should be approximately -1, -1, 8
        REQUIRE(std::abs(evals[0] - -1.0f) < 1e-2f);
        REQUIRE(std::abs(evals[1] - -1.0f) < 1e-2f);
        REQUIRE(std::abs(evals[2] - 8.0f) < 1e-2f);

        auto qrResult = QREigenVectors(A);
        // Verify V * D * V^T = A
        auto D = DynamicMatrix<float>::Diagonal(qrResult.eigenvalues);
        auto recon = qrResult.eigenvectors * D * qrResult.eigenvectors.Transpose();
        REQUIRE(NearlyEqual(recon, A, 1e-2f));
    }
}

TEST_CASE("SVD Phase 4", "[SVD]")
{
    SECTION("Singular Value Decomposition")
    {
        DynamicMatrix<float> A(4, 2, {
            2.0f, 4.0f,
            1.0f, 3.0f,
            0.0f, 0.0f,
            0.0f, 0.0f
        });

        auto svd = SingularValueDecomposition(A);
        
        // Reconstruction: U * Sigma * V^T
        DynamicMatrix<float> Sigma = DynamicMatrix<float>::Zero(2, 2);
        Sigma(0, 0) = svd.Sigma[0];
        Sigma(1, 1) = svd.Sigma[1];
        auto recon = svd.U * Sigma * svd.V.Transpose();
        REQUIRE(NearlyEqual(recon, A, 1e-2f));
    }

    SECTION("Pseudo-Inverse and Low Rank")
    {
        DynamicMatrix<float> A(3, 2, {
            1.0f, 2.0f,
            3.0f, 4.0f,
            5.0f, 6.0f
        });

        auto pinv = PseudoInverse(A);
        auto check = A * pinv * A;
        REQUIRE(NearlyEqual(check, A, 1e-2f));

        auto lowRank = LowRankApproximation(A, 1);
        REQUIRE(lowRank.Rows() == 3);
        REQUIRE(lowRank.Columns() == 2);
    }
}

TEST_CASE("Statistics Phase 5", "[Statistics]")
{
    SECTION("Covariance and Correlation")
    {
        DynamicMatrix<float> X(3, 2, {
            1.0f, 2.0f,
            2.0f, 4.0f,
            3.0f, 6.0f
        });

        auto cov = CovarianceMatrix(X);
        REQUIRE(cov.Rows() == 2);
        REQUIRE(cov(0, 0) == 1.0f);
        REQUIRE(cov(0, 1) == 2.0f);
        REQUIRE(NearlyEqual(cov, cov.Transpose()));

        auto corr = CorrelationMatrix(X);
        REQUIRE(std::abs(corr(0, 1) - 1.0f) < 1e-4f); // Perfectly correlated
    }

    SECTION("PCA and Linear Regression")
    {
        DynamicMatrix<float> X(5, 2, {
            2.5f, 2.4f,
            0.5f, 0.7f,
            2.2f, 2.9f,
            1.9f, 2.2f,
            3.1f, 3.0f
        });

        auto pca = PCA(X);
        REQUIRE(pca.explainedVariance[0] > pca.explainedVariance[1]);
        REQUIRE(pca.components.Rows() == 2);
        REQUIRE(IsIdentity(pca.components.Transpose() * pca.components, 1.0e-4f));

        // Linear Regression
        DynamicMatrix<float> RegX(3, 2, {
            1.0f, 1.0f,
            1.0f, 2.0f,
            1.0f, 3.0f
        });
        std::vector<float> y = {6.0f, 5.0f, 7.0f};
        auto beta = LinearRegression(RegX, y);
        // beta should fit y = beta0 + beta1 * x
        REQUIRE(beta.size() == 2);
    }
}

TEST_CASE("Linear Algebra Bugfixes", "[LinearAlgebraBugfixes]")
{
    SECTION("Vector LengthInverse Zero Guard")
    {
        Vec2f v2 = Vec2f::Zero();
        REQUIRE(v2.LengthInverse() == 0.0f);

        Vec3f v3 = Vec3f::Zero();
        REQUIRE(v3.LengthInverse() == 0.0f);

        Vec4f v4 = Vec4f::Zero();
        REQUIRE(v4.LengthInverse() == 0.0f);
    }

    SECTION("Matrix constexpr Determinant and Inverse")
    {
        constexpr Mat2f m2 = Mat2f::Identity();
        constexpr Mat2f inv2 = Inverse(m2);
        static_assert(inv2(0, 0) == 1.0f);

        constexpr Mat3f m3 = Mat3f::Identity();
        constexpr Mat3f inv3 = Inverse(m3);
        static_assert(inv3(1, 1) == 1.0f);

        constexpr Mat4f m4 = Mat4f::Identity();
        constexpr Mat4f inv4 = Inverse(m4);
        static_assert(inv4(2, 2) == 1.0f);

        constexpr float det4 = Determinant(m4);
        static_assert(det4 == 1.0f);
    }

    SECTION("Quaternion Rotate Fast Path")
    {
        Quatf q = RotationAroundY(0.5f);
        Vec3f v(1.0f, 0.0f, 0.0f);
        Vec3f rotated = Rotate(q, v);
        REQUIRE(std::abs(rotated.Length() - 1.0f) < 1e-4f);
    }

    SECTION("Decomposition Non-Square Matrix Exceptions")
    {
        DynamicMatrix<float> A(3, 2);
        REQUIRE_THROWS_AS(LU(A), std::invalid_argument);
        REQUIRE_THROWS_AS(LUP(A), std::invalid_argument);
        REQUIRE_THROWS_AS(Cholesky(A), std::invalid_argument);
        REQUIRE_THROWS_AS(LDLT(A), std::invalid_argument);
    }

    SECTION("SVD Rank-Deficient Basis Completion")
    {
        DynamicMatrix<float> A(3, 2, {
            1.0f, 2.0f,
            2.0f, 4.0f,
            3.0f, 6.0f
        });
        auto svd = SingularValueDecomposition(A);
        REQUIRE(svd.U.Rows() == 3);
        REQUIRE(svd.U.Columns() == 2);
        
        auto UT = svd.U.Transpose();
        auto UTU = UT * svd.U;
        REQUIRE(NearlyEqual(UTU, DynamicMatrix<float>::Identity(2)));
    }
}

TEST_CASE("Dynamic API Validation Policy", "[DynamicMatrix][Tensor][Validation]")
{
    SECTION("DynamicMatrix rejects partial-empty and mismatched dimensions")
    {
        REQUIRE_THROWS_AS(
            DynamicMatrix<float>(0, 3),
            std::invalid_argument);

        DynamicMatrix<float> a(2, 3);
        DynamicMatrix<float> b(3, 2);
        DynamicMatrix<float> c(2, 2);

        REQUIRE_THROWS_AS(a + c, std::invalid_argument);
        REQUIRE_THROWS_AS(a - c, std::invalid_argument);
        REQUIRE_THROWS_AS(a * c, std::invalid_argument);
        REQUIRE_THROWS_AS(a / 0.0f, std::domain_error);
        REQUIRE_THROWS_AS(a.Trace(), std::invalid_argument);
        REQUIRE_NOTHROW(a * b);
    }

    SECTION("Linear solvers reject invalid runtime dimensions")
    {
        DynamicMatrix<float> nonsquare(2, 3);
        DynamicMatrix<float> square(2, 2, {
            2.0f, 0.0f,
            0.0f, 4.0f
        });

        std::vector<float> shortRhs = { 1.0f };
        DynamicMatrix<float> matrixRhs(3, 1);

        REQUIRE_THROWS_AS(LinearSolve(square, shortRhs), std::invalid_argument);
        REQUIRE_THROWS_AS(LinearSolve(nonsquare, std::vector<float>{ 1.0f, 2.0f }), std::invalid_argument);
        REQUIRE_THROWS_AS(LinearSolve(square, matrixRhs), std::invalid_argument);

        DynamicMatrix<float> rhs(2, 2, {
            2.0f, 4.0f,
            8.0f, 12.0f
        });

        DynamicMatrix<float> solved =
            LinearSolve(square, rhs);

        REQUIRE(NearlyEqual(square * solved, rhs));
    }

    SECTION("Decomposition and statistics APIs reject invalid runtime inputs")
    {
        DynamicMatrix<float> wide(2, 3);
        DynamicMatrix<float> tall(3, 2);

        REQUIRE_THROWS_AS(QR(wide), std::invalid_argument);
        REQUIRE_THROWS_AS(GolubKahanBidiagonalization(wide), std::invalid_argument);
        REQUIRE_THROWS_AS(SingularValueDecomposition(DynamicMatrix<float>()), std::invalid_argument);
        REQUIRE_THROWS_AS(PowerIteration(wide), std::invalid_argument);
        REQUIRE_THROWS_AS(InversePowerIteration(wide), std::invalid_argument);
        REQUIRE_THROWS_AS(ShiftedInversePowerIteration(wide, 1.0f), std::invalid_argument);
        REQUIRE_THROWS_AS(QREigenVectors(wide), std::invalid_argument);
        REQUIRE_THROWS_AS(LinearRegression(tall, std::vector<float>{ 1.0f, 2.0f }), std::invalid_argument);
    }

    SECTION("Tensor public kernels reject invalid data-driven shapes")
    {
        Tensor<float> logits({ 2, 0 });

        REQUIRE_THROWS_AS(
            SoftmaxLastDim(logits),
            std::invalid_argument);

        Tensor<float> labels({ 2, 3 }, {
            1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f
        });

        Tensor<float> probabilities({ 2, 3 }, {
            0.5f, 0.25f, 0.25f,
            0.1f, 0.8f, 0.1f
        });

        const float loss =
            CrossEntropyMean(labels, probabilities);

        const float expected =
            (-std::log(0.5f + 1.0e-7f) -
             std::log(0.8f + 1.0e-7f)) /
            2.0f;

        REQUIRE(std::abs(loss - expected) < 1.0e-5f);
    }
}

TEST_CASE("DynamicMatrix Upgrades - Rank, Determinant, Inverse, ConditionNumber", "[DynamicMatrix][Upgrades]")
{
    SECTION("Determinant")
    {
        // 2x2 Determinant
        DynamicMatrix<double> A(2, 2, {3.0, 8.0, 4.0, 6.0});
        REQUIRE(std::abs(Determinant(A) - -14.0) < 1e-9);

        // 3x3 Determinant
        DynamicMatrix<double> B(3, 3, {
            1.0, 2.0, 3.0,
            0.0, 1.0, 4.0,
            5.0, 6.0, 0.0
        });
        // det = 1*(0-24) - 2*(0-20) + 3*(0-5) = -24 + 40 - 15 = 1
        REQUIRE(std::abs(Determinant(B) - 1.0) < 1e-9);

        // Singular matrix determinant is 0
        DynamicMatrix<double> C(3, 3, {
            1.0, 2.0, 3.0,
            2.0, 4.0, 6.0,
            3.0, 5.0, 7.0
        });
        REQUIRE(std::abs(Determinant(C) - 0.0) < 1e-9);
    }

    SECTION("Inverse")
    {
        DynamicMatrix<double> A(3, 3, {
            2.0, 1.0, 1.0,
            1.0, 3.0, 1.0,
            1.0, 1.0, 4.0
        });
        DynamicMatrix<double> invA = Inverse(A);
        DynamicMatrix<double> identity = A * invA;
        REQUIRE(IsIdentity(identity, 1e-9));
    }

    SECTION("Rank")
    {
        // Full rank 3x3
        DynamicMatrix<double> A(3, 3, {
            2.0, 1.0, 1.0,
            1.0, 3.0, 1.0,
            1.0, 1.0, 4.0
        });
        REQUIRE(Rank(A) == 3);

        // Rank-deficient 3x3 (rank 1)
        DynamicMatrix<double> B(3, 3, {
            1.0, 2.0, 3.0,
            2.0, 4.0, 6.0,
            3.0, 6.0, 9.0
        });
        REQUIRE(Rank(B) == 1);
    }

    SECTION("ConditionNumber")
    {
        DynamicMatrix<double> I = DynamicMatrix<double>::Identity(3);
        REQUIRE(std::abs(ConditionNumber(I) - 1.0) < 1e-9);

        // Singular matrix has infinite condition number
        DynamicMatrix<double> B(3, 3, {
            1.0, 2.0, 3.0,
            2.0, 4.0, 6.0,
            3.0, 6.0, 9.0
        });
        REQUIRE(std::isinf(ConditionNumber(B)));
    }
}

TEST_CASE("Eigen and SVD Advanced Decompositions", "[LinearAlgebra][Decompositions]")
{
    SECTION("Householder Tridiagonalization")
    {
        DynamicMatrix<double> A(3, 3, {
            4.0, 1.0, -2.0,
            1.0, 3.0, 5.0,
            -2.0, 5.0, 6.0
        });
        auto tri = HouseholderTridiagonalization(A);
        
        // Check T is tridiagonal: T(0, 2) and T(2, 0) must be 0
        REQUIRE(std::abs(tri.T_mat(0, 2)) < 1e-9);
        REQUIRE(std::abs(tri.T_mat(2, 0)) < 1e-9);

        // Check Q is orthogonal
        DynamicMatrix<double> QTQ = tri.Q.Transpose() * tri.Q;
        REQUIRE(IsIdentity(QTQ, 1e-9));

        // Check Q * T * Q^T = A
        DynamicMatrix<double> recon = tri.Q * tri.T_mat * tri.Q.Transpose();
        REQUIRE(NearlyEqual(recon, A, 1e-9));
    }

    SECTION("Golub-Kahan Bidiagonalization")
    {
        DynamicMatrix<double> A(4, 3, {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0
        });
        auto bid = GolubKahanBidiagonalization(A);

        // B must be upper bidiagonal of size 4x3. Non-zero only on diagonal and superdiagonal.
        for (std::size_t r = 0; r < 4; ++r)
        {
            for (std::size_t c = 0; c < 3; ++c)
            {
                if (r != c && r + 1 != c)
                {
                    REQUIRE(std::abs(bid.B(r, c)) < 1e-9);
                }
            }
        }

        // U must be 4x4 orthogonal
        DynamicMatrix<double> UTU = bid.U.Transpose() * bid.U;
        REQUIRE(IsIdentity(UTU, 1e-9));

        // V must be 3x3 orthogonal
        DynamicMatrix<double> VTV = bid.V.Transpose() * bid.V;
        REQUIRE(IsIdentity(VTV, 1e-9));

        // U * B * V^T = A
        DynamicMatrix<double> recon = bid.U * bid.B * bid.V.Transpose();
        REQUIRE(NearlyEqual(recon, A, 1e-9));
    }
}

TEST_CASE("Matrix Functions Phase 6", "[LinearAlgebra][MatrixFunctions]")
{
    SECTION("Matrix exponential of zero is identity")
    {
        DynamicMatrix<double> zero =
            DynamicMatrix<double>::Zero(3, 3);

        DynamicMatrix<double> expZero =
            MatrixExponential(zero);

        REQUIRE(IsIdentity(expZero, 1e-12));
    }

    SECTION("Matrix exponential of diagonal matrix exponentiates diagonal entries")
    {
        DynamicMatrix<double> diagonal =
            DynamicMatrix<double>::Diagonal(
                std::vector<double>
                {
                    0.0,
                    1.0,
                    -1.0
                });

        DynamicMatrix<double> expDiagonal =
            MatrixExponential(diagonal);

        REQUIRE(std::abs(expDiagonal(0, 0) - 1.0) < 1e-11);
        REQUIRE(std::abs(expDiagonal(1, 1) - std::exp(1.0)) < 1e-11);
        REQUIRE(std::abs(expDiagonal(2, 2) - std::exp(-1.0)) < 1e-11);
        REQUIRE(std::abs(expDiagonal(0, 1)) < 1e-12);
        REQUIRE(std::abs(expDiagonal(1, 2)) < 1e-12);
    }

    SECTION("Matrix exponential of nilpotent matrix terminates as I + A")
    {
        DynamicMatrix<double> nilpotent(2, 2, {
            0.0, 2.0,
            0.0, 0.0
        });

        DynamicMatrix<double> expected(2, 2, {
            1.0, 2.0,
            0.0, 1.0
        });

        REQUIRE(NearlyEqual(
            MatrixExponential(nilpotent),
            expected,
            1e-11));
    }

    SECTION("Matrix exponential of 2D skew generator produces rotation")
    {
        const double angle =
            0.75;

        DynamicMatrix<double> generator(2, 2, {
            0.0, -angle,
            angle, 0.0
        });

        DynamicMatrix<double> expected(2, 2, {
            std::cos(angle), -std::sin(angle),
            std::sin(angle),  std::cos(angle)
        });

        REQUIRE(NearlyEqual(
            MatrixExponential(generator),
            expected,
            1e-11));
    }

    SECTION("Matrix exponential rejects non-square matrices")
    {
        DynamicMatrix<double> nonsquare(2, 3);

        REQUIRE_THROWS_AS(
            MatrixExponential(nonsquare),
            std::invalid_argument);
    }
}

TEST_CASE("Optimization Phase C", "[Optimization]")
{
    auto column = [](std::initializer_list<double> values)
    {
        DynamicMatrix<double> result(values.size(), 1);
        std::size_t row = 0;

        for (double value : values)
        {
            result(row, 0) = value;
            ++row;
        }

        return result;
    };

    SECTION("GradientDescent minimizes f(x,y)=x^2+y^2")
    {
        auto objective = [](const DynamicMatrix<double>& x)
        {
            return (x(0, 0) * x(0, 0)) + (x(1, 0) * x(1, 0));
        };

        auto gradient = [](const DynamicMatrix<double>& x)
        {
            return DynamicMatrix<double>(2, 1, {
                2.0 * x(0, 0),
                2.0 * x(1, 0)
            });
        };

        OptimizationSettings<double> settings;
        settings.MaxIterations = 200;
        settings.LearningRate = 1.0;
        settings.GradientTolerance = 1.0e-10;

        const auto result =
            GradientDescent(
                objective,
                gradient,
                column({ 3.0, -4.0 }),
                settings);

        REQUIRE(result.Converged);
        REQUIRE(result.Value < 1.0e-18);
        REQUIRE(std::abs(result.X(0, 0)) < 1.0e-9);
        REQUIRE(std::abs(result.X(1, 0)) < 1.0e-9);
    }

    SECTION("Momentum minimizes shifted parabola")
    {
        auto objective = [](const DynamicMatrix<double>& x)
        {
            const double dx = x(0, 0) - 2.0;
            const double dy = x(1, 0) + 3.0;
            return (dx * dx) + (dy * dy);
        };

        auto gradient = [](const DynamicMatrix<double>& x)
        {
            return DynamicMatrix<double>(2, 1, {
                2.0 * (x(0, 0) - 2.0),
                2.0 * (x(1, 0) + 3.0)
            });
        };

        OptimizationSettings<double> settings;
        settings.MaxIterations = 1000;
        settings.LearningRate = 0.03;
        settings.GradientTolerance = 1.0e-8;
        settings.ValueTolerance = 0.0;

        const auto result =
            Momentum(
                objective,
                gradient,
                column({ -5.0, 4.0 }),
                0.9,
                settings);

        REQUIRE(result.Converged);
        REQUIRE(result.Value < 1.0e-10);
        REQUIRE(std::abs(result.X(0, 0) - 2.0) < 1.0e-5);
        REQUIRE(std::abs(result.X(1, 0) + 3.0) < 1.0e-5);
    }

    SECTION("Nesterov minimizes shifted parabola")
    {
        auto objective = [](const DynamicMatrix<double>& x)
        {
            const double dx = x(0, 0) + 1.0;
            const double dy = x(1, 0) - 5.0;
            return (dx * dx) + (dy * dy);
        };

        auto gradient = [](const DynamicMatrix<double>& x)
        {
            return DynamicMatrix<double>(2, 1, {
                2.0 * (x(0, 0) + 1.0),
                2.0 * (x(1, 0) - 5.0)
            });
        };

        OptimizationSettings<double> settings;
        settings.MaxIterations = 1000;
        settings.LearningRate = 0.03;
        settings.GradientTolerance = 1.0e-8;
        settings.ValueTolerance = 0.0;

        const auto result =
            Nesterov(
                objective,
                gradient,
                column({ 4.0, -2.0 }),
                0.9,
                settings);

        REQUIRE(result.Converged);
        REQUIRE(result.Value < 1.0e-10);
        REQUIRE(std::abs(result.X(0, 0) + 1.0) < 1.0e-5);
        REQUIRE(std::abs(result.X(1, 0) - 5.0) < 1.0e-5);
    }

    SECTION("Adam minimizes Rosenbrock approximately")
    {
        auto objective = [](const DynamicMatrix<double>& x)
        {
            const double a = 1.0 - x(0, 0);
            const double b = x(1, 0) - (x(0, 0) * x(0, 0));
            return (a * a) + (100.0 * b * b);
        };

        auto gradient = [](const DynamicMatrix<double>& x)
        {
            const double x0 = x(0, 0);
            const double x1 = x(1, 0);
            return DynamicMatrix<double>(2, 1, {
                -2.0 * (1.0 - x0) - (400.0 * x0 * (x1 - (x0 * x0))),
                200.0 * (x1 - (x0 * x0))
            });
        };

        OptimizationSettings<double> settings;
        settings.MaxIterations = 20000;
        settings.LearningRate = 0.002;
        settings.GradientTolerance = 1.0e-5;
        settings.StepTolerance = 1.0e-14;
        settings.ValueTolerance = 0.0;

        const auto result =
            Adam(
                objective,
                gradient,
                column({ -1.2, 1.0 }),
                0.9,
                0.999,
                1.0e-8,
                settings);

        REQUIRE(result.Value < 1.0e-3);
        REQUIRE(std::abs(result.X(0, 0) - 1.0) < 4.0e-2);
        REQUIRE(std::abs(result.X(1, 0) - 1.0) < 8.0e-2);
    }

    SECTION("ConjugateGradientSolve solves small SPD system")
    {
        DynamicMatrix<double> A(2, 2, {
            4.0, 1.0,
            1.0, 3.0
        });

        OptimizationSettings<double> settings;
        settings.MaxIterations = 20;
        settings.GradientTolerance = 1.0e-12;

        const auto result =
            ConjugateGradientSolve(
                A,
                column({ 1.0, 2.0 }),
                settings);

        REQUIRE(result.Converged);
        REQUIRE(result.ResidualNorm < 1.0e-12);
        REQUIRE(std::abs(result.X(0, 0) - (1.0 / 11.0)) < 1.0e-10);
        REQUIRE(std::abs(result.X(1, 0) - (7.0 / 11.0)) < 1.0e-10);
    }

    SECTION("Newton solves quadratic in one/few steps")
    {
        auto objective = [](const DynamicMatrix<double>& x)
        {
            const double dx = x(0, 0) - 3.0;
            const double dy = x(1, 0) + 2.0;
            return (dx * dx) + (dy * dy);
        };

        auto gradient = [](const DynamicMatrix<double>& x)
        {
            return DynamicMatrix<double>(2, 1, {
                2.0 * (x(0, 0) - 3.0),
                2.0 * (x(1, 0) + 2.0)
            });
        };

        auto hessian = [](const DynamicMatrix<double>&)
        {
            return DynamicMatrix<double>(2, 2, {
                2.0, 0.0,
                0.0, 2.0
            });
        };

        OptimizationSettings<double> settings;
        settings.MaxIterations = 20;
        settings.GradientTolerance = 1.0e-12;

        const auto result =
            Newton(
                objective,
                gradient,
                hessian,
                column({ 0.0, 0.0 }),
                settings);

        REQUIRE(result.Converged);
        REQUIRE(result.Value < 1.0e-20);
        REQUIRE(std::abs(result.X(0, 0) - 3.0) < 1.0e-10);
        REQUIRE(std::abs(result.X(1, 0) + 2.0) < 1.0e-10);
    }

    SECTION("GaussNewton fits y = ax + b")
    {
        const std::vector<double> xs = { 0.0, 1.0, 2.0, 3.0 };
        const std::vector<double> ys = { 1.0, 3.0, 5.0, 7.0 };

        auto residual = [&](const DynamicMatrix<double>& p)
        {
            DynamicMatrix<double> r(xs.size(), 1);

            for (std::size_t i = 0; i < xs.size(); ++i)
            {
                r(i, 0) = (p(0, 0) * xs[i]) + p(1, 0) - ys[i];
            }

            return r;
        };

        auto jacobian = [&](const DynamicMatrix<double>&)
        {
            DynamicMatrix<double> J(xs.size(), 2);

            for (std::size_t i = 0; i < xs.size(); ++i)
            {
                J(i, 0) = xs[i];
                J(i, 1) = 1.0;
            }

            return J;
        };

        OptimizationSettings<double> settings;
        settings.MaxIterations = 20;
        settings.GradientTolerance = 1.0e-12;

        const auto result =
            GaussNewton(
                residual,
                jacobian,
                column({ 0.0, 0.0 }),
                settings);

        REQUIRE(result.Converged);
        REQUIRE(result.Value < 1.0e-20);
        REQUIRE(std::abs(result.X(0, 0) - 2.0) < 1.0e-10);
        REQUIRE(std::abs(result.X(1, 0) - 1.0) < 1.0e-10);
    }

    SECTION("LevenbergMarquardt fits y = a * exp(bx)")
    {
        const std::vector<double> xs = { 0.0, 0.5, 1.0, 1.5 };
        const double trueA = 2.0;
        const double trueB = 0.4;

        auto residual = [&](const DynamicMatrix<double>& p)
        {
            DynamicMatrix<double> r(xs.size(), 1);

            for (std::size_t i = 0; i < xs.size(); ++i)
            {
                const double predicted =
                    p(0, 0) * std::exp(p(1, 0) * xs[i]);

                const double observed =
                    trueA * std::exp(trueB * xs[i]);

                r(i, 0) = predicted - observed;
            }

            return r;
        };

        auto jacobian = [&](const DynamicMatrix<double>& p)
        {
            DynamicMatrix<double> J(xs.size(), 2);

            for (std::size_t i = 0; i < xs.size(); ++i)
            {
                const double expTerm =
                    std::exp(p(1, 0) * xs[i]);

                J(i, 0) = expTerm;
                J(i, 1) = p(0, 0) * xs[i] * expTerm;
            }

            return J;
        };

        OptimizationSettings<double> settings;
        settings.MaxIterations = 100;
        settings.GradientTolerance = 1.0e-12;
        settings.StepTolerance = 1.0e-12;
        settings.ValueTolerance = 1.0e-14;

        const auto result =
            LevenbergMarquardt(
                residual,
                jacobian,
                column({ 1.5, 0.1 }),
                1.0e-3,
                10.0,
                0.1,
                settings);

        REQUIRE(result.Converged);
        REQUIRE(result.Value < 1.0e-18);
        REQUIRE(std::abs(result.X(0, 0) - trueA) < 1.0e-8);
        REQUIRE(std::abs(result.X(1, 0) - trueB) < 1.0e-8);
    }
}

TEST_CASE("Probability Phase D", "[Probability][Statistics]")
{
    SECTION("Distributions expose analytic density, CDF, moments, and validation")
    {
        UniformDistribution<double> uniform(-2.0, 2.0);
        REQUIRE(std::abs(uniform.Pdf(0.0) - 0.25) < 1.0e-12);
        REQUIRE(std::abs(uniform.Cdf(0.0) - 0.5) < 1.0e-12);
        REQUIRE(std::abs(uniform.MeanValue()) < 1.0e-12);
        REQUIRE(std::abs(uniform.Variance() - (16.0 / 12.0)) < 1.0e-12);

        NormalDistribution<double> normal(0.0, 1.0);
        REQUIRE(std::abs(normal.Cdf(0.0) - 0.5) < 1.0e-12);
        REQUIRE(std::abs(normal.Pdf(0.0) - 0.3989422804014327) < 1.0e-12);

        BernoulliDistribution<double> bernoulli(0.25);
        REQUIRE(std::abs(bernoulli.Pmf(true) - 0.25) < 1.0e-12);
        REQUIRE(std::abs(bernoulli.Variance() - 0.1875) < 1.0e-12);

        ExponentialDistribution<double> exponential(2.0);
        REQUIRE(std::abs(exponential.MeanValue() - 0.5) < 1.0e-12);
        REQUIRE(std::abs(exponential.Cdf(0.0)) < 1.0e-12);

        REQUIRE_THROWS_AS(UniformDistribution<double>(1.0, 1.0), std::invalid_argument);
        REQUIRE_THROWS_AS(NormalDistribution<double>(0.0, 0.0), std::invalid_argument);
        REQUIRE_THROWS_AS(BernoulliDistribution<double>(1.5), std::invalid_argument);
        REQUIRE_THROWS_AS(ExponentialDistribution<double>(0.0), std::invalid_argument);
    }

    SECTION("Sampling uses explicit deterministic random generator state")
    {
        RandomGenerator first(1234);
        RandomGenerator second(1234);

        UniformDistribution<double> uniform(10.0, 20.0);

        const double firstSample =
            uniform.Sample(first);

        const double secondSample =
            uniform.Sample(second);

        REQUIRE(firstSample >= 10.0);
        REQUIRE(firstSample < 20.0);
        REQUIRE(firstSample == secondSample);

        NormalDistribution<double> normal(3.0, 2.0);
        REQUIRE(normal.Sample(first) == normal.Sample(second));

        std::size_t index =
            SampleWeightedIndex(
                std::vector<double>{ 0.0, 0.0, 1.0 },
                first);

        REQUIRE(index == 2);
        REQUIRE_THROWS_AS(SampleWeightedIndex(std::vector<double>{ 0.0, 0.0 }, first), std::invalid_argument);
    }

    SECTION("Sample analysis helpers compute mean and variance")
    {
        std::vector<double> samples =
        {
            1.0,
            2.0,
            3.0,
            4.0
        };

        REQUIRE(std::abs(Mean(samples) - 2.5) < 1.0e-12);
        REQUIRE(std::abs(Variance(samples, false) - 1.25) < 1.0e-12);
        REQUIRE(std::abs(Variance(samples, true) - (5.0 / 3.0)) < 1.0e-12);
        REQUIRE(std::abs(StandardDeviation(samples, false) - std::sqrt(1.25)) < 1.0e-12);
    }
}
