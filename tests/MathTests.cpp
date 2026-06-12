#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <vector>

import Kairo.Foundation.Math.Vector;
import Kairo.Foundation.Math.Matrix;
import Kairo.Foundation.Math.Quaternion;
import Kairo.Foundation.Math.Transform;
import Kairo.Foundation.Math.DynamicMatrix;
import Kairo.Foundation.Math.LinearAlgebra.LinearSolve;
import Kairo.Foundation.Math.LinearAlgebra.Decomposition;
import Kairo.Foundation.Math.LinearAlgebra.Eigen;
import Kairo.Foundation.Math.LinearAlgebra.SVD;
import Kairo.Foundation.Math.LinearAlgebra.Statistics;
import Kairo.Foundation.Math.LinearAlgebra.MatrixFunctions;

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

    SECTION("Transform Inverse Non-Uniform Scale Exceptions")
    {
        Transformf t;
        t.Scale = Vec3f(1.0f, 2.0f, 1.0f);
        REQUIRE_THROWS_AS(Inverse(t), std::invalid_argument);

        t.Scale = Vec3f(0.0f, 1.0f, 1.0f);
        REQUIRE_THROWS_AS(Inverse(t), std::invalid_argument);
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

