module;

#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <numbers>
#include <type_traits>
#include <utility>

export module Foundation.Math.Quaternion;

import Foundation.Math.Vector;
import Foundation.Math.Matrix;


export namespace foundation::math
{
    template<FloatingPoint T>
    struct Quaternion;

    template<FloatingPoint T>
    [[nodiscard]]
    bool NearlyEqual(
        const Quaternion<T>& lhs,
        const Quaternion<T>& rhs,
        T epsilon = std::numeric_limits<T>::epsilon() * T(10)) noexcept;

    /// A quaternion representing 3D orientation and rotation.
    ///
    /// Stored as (x, y, z, w), where xyz is the vector/imaginary part and w is
    /// the scalar/real part. Unit quaternions represent rotations. This type is
    /// allocation-free, standard-layout, trivially copyable, and intentionally
    /// transparent like the vector and matrix foundation types.
    template<FloatingPoint T>
    struct Quaternion final
    {
        using ValueType = T;

        static constexpr std::size_t Size = 4;

        /// Input: none.
        /// Output: identity rotation (0, 0, 0, 1).
        /// Task: provide the neutral rotation for composition and defaults.
        [[nodiscard]]
        static constexpr Quaternion Identity() noexcept
        {
            return { T(0), T(0), T(0), T(1) };
        }

        /// Input: none.
        /// Output: all-zero quaternion.
        /// Task: provide a sentinel/math value. This is not a valid rotation.
        [[nodiscard]]
        static constexpr Quaternion Zero() noexcept
        {
            return { T(0), T(0), T(0), T(0) };
        }

        T x;
        T y;
        T z;
        T w;

        /// Input: none.
        /// Output: identity quaternion.
        /// Task: safe default construction for transforms and cameras.
        constexpr Quaternion() noexcept
            : x(T(0))
            , y(T(0))
            , z(T(0))
            , w(T(1))
        {
        }

        /// Input: x, y, z, and w components.
        /// Output: quaternion containing the supplied components.
        /// Task: explicit low-level construction with no hidden normalization.
        constexpr Quaternion(T xValue, T yValue, T zValue, T wValue) noexcept
            : x(xValue)
            , y(yValue)
            , z(zValue)
            , w(wValue)
        {
        }

        /// Input: vector part and scalar part.
        /// Output: quaternion assembled as (vectorPart.x, y, z, scalarPart).
        /// Task: useful for pure-vector quaternions and algebra helpers.
        constexpr Quaternion(const Vector3<T>& vectorPart, T scalarPart) noexcept
            : x(vectorPart.x)
            , y(vectorPart.y)
            , z(vectorPart.z)
            , w(scalarPart)
        {
        }

        /// Input: component index in [0, Size).
        /// Output: mutable reference to x/y/z/w in that order.
        /// Task: generic indexed access for tests, serialization, and tooling.
        [[nodiscard]]
        constexpr T& operator[](std::size_t index) noexcept
        {
            assert(index < Size);
            return Data()[index];
        }

        /// Input: component index in [0, Size).
        /// Output: const reference to x/y/z/w in that order.
        /// Task: read-only generic indexed access.
        [[nodiscard]]
        constexpr const T& operator[](std::size_t index) const noexcept
        {
            assert(index < Size);
            return Data()[index];
        }

        /// Input: none.
        /// Output: pointer to x, the first contiguous component.
        /// Task: expose storage for serialization, editor tooling, and upload paths.
        [[nodiscard]]
        constexpr T* Data() noexcept
        {
            return &x;
        }

        /// Input: none.
        /// Output: const pointer to x.
        /// Task: read-only component access for upload and serialization.
        [[nodiscard]]
        constexpr const T* Data() const noexcept
        {
            return &x;
        }

        /// Input: this quaternion.
        /// Output: unchanged copy.
        /// Task: expression symmetry with unary minus.
        [[nodiscard]]
        constexpr Quaternion operator+() const noexcept
        {
            return *this;
        }

        /// Input: this quaternion.
        /// Output: component-wise negation.
        /// Task: flip quaternion sign. q and -q represent the same rotation, but
        /// sign flipping is useful for shortest-path interpolation.
        [[nodiscard]]
        constexpr Quaternion operator-() const noexcept
        {
            return { -x, -y, -z, -w };
        }

        /// Input: another quaternion.
        /// Output: exact component-wise equality.
        /// Task: deterministic exact comparison. Use NearlyEqual() for rotation
        /// equivalence because q and -q represent the same orientation.
        [[nodiscard]]
        constexpr bool operator==(const Quaternion&) const noexcept = default;

        /// Input: this quaternion.
        /// Output: squared magnitude.
        /// Task: avoid sqrt when checking normalization or invertibility.
        [[nodiscard]]
        constexpr T LengthSquared() const noexcept
        {
            return (x * x) + (y * y) + (z * z) + (w * w);
        }

        /// Input: this quaternion.
        /// Output: magnitude.
        /// Task: measure quaternion length. Unit quaternions should be near 1.
        [[nodiscard]]
        T Length() const noexcept
        {
            return std::sqrt(LengthSquared());
        }

        /// Input: this quaternion.
        /// Output: reciprocal length, or zero for near-zero input.
        /// Task: safe building block for normalization-heavy code.
        [[nodiscard]]
        T LengthInverse() const noexcept
        {
            const T length = Length();
            return length <= std::numeric_limits<T>::epsilon()
                ? T(0)
                : T(1) / length;
        }

        /// Input: this quaternion.
        /// Output: normalized quaternion, or identity for near-zero input.
        /// Task: produce a unit quaternion suitable for rotation APIs.
        [[nodiscard]]
        Quaternion Normalized() const noexcept
        {
            const T inverseLength = LengthInverse();
            if (inverseLength == T(0))
            {
                return Identity();
            }

            return { x * inverseLength, y * inverseLength, z * inverseLength, w * inverseLength };
        }

        /// Input: this quaternion.
        /// Output: none; mutates this quaternion to unit length when possible.
        /// Task: in-place normalization for transform update paths.
        void Normalize() noexcept
        {
            const T inverseLength = LengthInverse();
            if (inverseLength == T(0))
            {
                *this = Identity();
                return;
            }

            x *= inverseLength;
            y *= inverseLength;
            z *= inverseLength;
            w *= inverseLength;
        }

        /// Input: this quaternion.
        /// Output: conjugate (-x, -y, -z, w).
        /// Task: invert a unit quaternion and support general inverse math.
        [[nodiscard]]
        constexpr Quaternion Conjugate() const noexcept
        {
            return { -x, -y, -z, w };
        }

        /// Input: optional tolerance for LengthSquared() comparison to 1.
        /// Output: true when this quaternion is close to unit length.
        /// Task: validate values before using them as rotations in debug/tests.
        [[nodiscard]]
        bool IsNormalized(
            T epsilon = std::numeric_limits<T>::epsilon() * T(10)) const noexcept
        {
            return NearlyEqual(LengthSquared(), T(1), epsilon);
        }

        /// Input: optional tolerance.
        /// Output: true when this quaternion represents identity rotation.
        /// Task: tolerant identity check using quaternion rotation equivalence.
        [[nodiscard]]
        bool IsIdentity(
            T epsilon = std::numeric_limits<T>::epsilon() * T(10)) const noexcept
        {
            return NearlyEqual(*this, Identity(), epsilon);
        }

        /// Input: this quaternion.
        /// Output: true when every component is exactly zero.
        /// Task: detect the zero sentinel/math value. It is not a valid rotation.
        [[nodiscard]]
        constexpr bool IsZero() const noexcept
        {
            return x == T(0) && y == T(0) && z == T(0) && w == T(0);
        }

        /// Input: this quaternion.
        /// Output: xyz vector part.
        /// Task: expose the imaginary/vector part for algebra and diagnostics.
        [[nodiscard]]
        constexpr Vector3<T> VectorPart() const noexcept
        {
            return { x, y, z };
        }

        /// Input: this quaternion.
        /// Output: w scalar part.
        /// Task: expose the real/scalar part for algebra and diagnostics.
        [[nodiscard]]
        constexpr T ScalarPart() const noexcept
        {
            return w;
        }
    };

    using Quatf = Quaternion<float>;
    using Quatd = Quaternion<double>;

    template<FloatingPoint T>
    using AxisAnglePair = std::pair<Vector3<T>, T>;

    /// Input: two quaternions.
    /// Output: scalar dot product.
    /// Task: measure angular closeness and choose interpolation direction.
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr T Dot(const Quaternion<T>& lhs, const Quaternion<T>& rhs) noexcept
    {
        return (lhs.x * rhs.x) + (lhs.y * rhs.y) + (lhs.z * rhs.z) + (lhs.w * rhs.w);
    }

    /// Input: quaternion and scalar.
    /// Output: quaternion after component-wise scaling.
    /// Task: in-place scalar multiplication for interpolation formulas.
    template<FloatingPoint T>
    constexpr Quaternion<T>& operator*=(Quaternion<T>& quaternion, T scalar) noexcept
    {
        quaternion.x *= scalar;
        quaternion.y *= scalar;
        quaternion.z *= scalar;
        quaternion.w *= scalar;
        return quaternion;
    }

    /// Input: quaternion and non-zero scalar.
    /// Output: quaternion after component-wise division.
    /// Task: in-place scalar division for inverse/normalization formulas.
    template<FloatingPoint T>
    constexpr Quaternion<T>& operator/=(Quaternion<T>& quaternion, T scalar) noexcept
    {
        assert(scalar != T(0));
        const T inverseScalar = T(1) / scalar;
        quaternion.x *= inverseScalar;
        quaternion.y *= inverseScalar;
        quaternion.z *= inverseScalar;
        quaternion.w *= inverseScalar;
        return quaternion;
    }

    /// Input: quaternion and scalar. Output: scaled quaternion. Task: value scaling.
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr Quaternion<T> operator*(Quaternion<T> quaternion, T scalar) noexcept
    {
        return quaternion *= scalar;
    }

    /// Input: scalar and quaternion. Output: scaled quaternion. Task: symmetric scaling.
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr Quaternion<T> operator*(T scalar, Quaternion<T> quaternion) noexcept
    {
        return quaternion *= scalar;
    }

    /// Input: quaternion and non-zero scalar. Output: divided quaternion.
    /// Task: value scalar division.
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr Quaternion<T> operator/(Quaternion<T> quaternion, T scalar) noexcept
    {
        return quaternion /= scalar;
    }

    /// Input: two quaternions. Output: component-wise sum.
    /// Task: support interpolation formulas; this is not rotation composition.
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr Quaternion<T> operator+(const Quaternion<T>& lhs, const Quaternion<T>& rhs) noexcept
    {
        return { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w };
    }

    /// Input: two quaternions. Output: component-wise difference.
    /// Task: support interpolation/debug math; this is not relative rotation.
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr Quaternion<T> operator-(const Quaternion<T>& lhs, const Quaternion<T>& rhs) noexcept
    {
        return { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w };
    }

    /// Input: two quaternions.
    /// Output: Hamilton product lhs * rhs.
    /// Task: compose rotations. With column-vector convention, `a * b` applies
    /// b first, then a, matching the matrix composition style in Matrix.cppm.
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr Quaternion<T> operator*(const Quaternion<T>& lhs, const Quaternion<T>& rhs) noexcept
    {
        return
        {
            (lhs.w * rhs.x) + (lhs.x * rhs.w) + (lhs.y * rhs.z) - (lhs.z * rhs.y),
            (lhs.w * rhs.y) - (lhs.x * rhs.z) + (lhs.y * rhs.w) + (lhs.z * rhs.x),
            (lhs.w * rhs.z) + (lhs.x * rhs.y) - (lhs.y * rhs.x) + (lhs.z * rhs.w),
            (lhs.w * rhs.w) - (lhs.x * rhs.x) - (lhs.y * rhs.y) - (lhs.z * rhs.z)
        };
    }

    /// Input: quaternion.
    /// Output: inverse quaternion, or identity for near-zero length.
    /// Task: undo a rotation/algebraic quaternion. For unit quaternions this is
    /// just the conjugate; for non-unit quaternions the length correction matters.
    template<FloatingPoint T>
    [[nodiscard]]
    Quaternion<T> Inverse(const Quaternion<T>& quaternion) noexcept
    {
        const T lengthSquared = quaternion.LengthSquared();
        if (lengthSquared <= std::numeric_limits<T>::epsilon())
        {
            return Quaternion<T>::Identity();
        }

        return quaternion.Conjugate() / lengthSquared;
    }

    /// Input: rotation axis and angle in radians.
    /// Output: unit quaternion representing that axis-angle rotation.
    /// Task: construct rotations from a geometric axis. The axis is normalized
    /// internally so callers cannot accidentally create garbage rotations.
    template<FloatingPoint T>
    [[nodiscard]]
    Quaternion<T> AxisAngle(const Vector3<T>& axis, T angleRadians) noexcept
    {
        const Vector3<T> normalizedAxis = SafeNormalize(axis);
        if (normalizedAxis.IsZero())
        {
            return Quaternion<T>::Identity();
        }

        const T halfAngle = angleRadians * T(0.5);
        const T sine = std::sin(halfAngle);
        const T cosine = std::cos(halfAngle);

        return
        {
            normalizedAxis.x * sine,
            normalizedAxis.y * sine,
            normalizedAxis.z * sine,
            cosine
        };
    }

    /// Input: angle in radians. Output: quaternion rotation around +X.
    /// Task: axis helper. The name avoids colliding with Matrix MakeRotationX().
    template<FloatingPoint T>
    [[nodiscard]]
    Quaternion<T> RotationAroundX(T radians) noexcept
    {
        return AxisAngle(Vector3<T>::UnitX(), radians);
    }

    /// Input: angle in radians. Output: quaternion rotation around +Y.
    /// Task: axis helper. The name avoids colliding with Matrix MakeRotationY().
    template<FloatingPoint T>
    [[nodiscard]]
    Quaternion<T> RotationAroundY(T radians) noexcept
    {
        return AxisAngle(Vector3<T>::UnitY(), radians);
    }

    /// Input: angle in radians. Output: quaternion rotation around +Z.
    /// Task: axis helper. The name avoids colliding with Matrix MakeRotationZ().
    template<FloatingPoint T>
    [[nodiscard]]
    Quaternion<T> RotationAroundZ(T radians) noexcept
    {
        return AxisAngle(Vector3<T>::UnitZ(), radians);
    }

    template<FloatingPoint T>
    [[nodiscard]]
    Quaternion<T> FromMatrix3(const Matrix3<T>& matrix) noexcept;

    /// Input: source and destination directions.
    /// Output: shortest rotation from source to destination.
    /// Task: build aiming/facing rotations for cameras, turrets, and alignment.
    template<FloatingPoint T>
    [[nodiscard]]
    Quaternion<T> FromToRotation(const Vector3<T>& from, const Vector3<T>& to) noexcept
    {
        const Vector3<T> start = SafeNormalize(from);
        const Vector3<T> end = SafeNormalize(to);

        if (start.IsZero() || end.IsZero())
        {
            return Quaternion<T>::Identity();
        }

        const T cosine = Clamp(Dot(start, end), T(-1), T(1));

        if (cosine >= T(1) - std::numeric_limits<T>::epsilon())
        {
            return Quaternion<T>::Identity();
        }

        if (cosine <= T(-1) + std::numeric_limits<T>::epsilon())
        {
            Vector3<T> axis = Orthogonal(start);
            axis.Normalize();
            return AxisAngle(axis, T(std::numbers::pi));
        }

        const Vector3<T> axis = Cross(start, end);
        const T s = std::sqrt((T(1) + cosine) * T(2));
        const T inverseS = T(1) / s;

        return Quaternion<T>
        {
            axis.x * inverseS,
            axis.y * inverseS,
            axis.z * inverseS,
            s * T(0.5)
        }.Normalized();
    }

    /// Input: forward direction and approximate up direction.
    /// Output: orientation whose local forward points along forward.
    /// Task: construct camera/NPC/projectile facing rotations. This uses the
    /// engine convention +X right, +Y up, and -Z forward.
    template<FloatingPoint T>
    [[nodiscard]]
    Quaternion<T> LookRotation(
        const Vector3<T>& forward,
        const Vector3<T>& up = Vector3<T>::Up()) noexcept
    {
        const Vector3<T> f = SafeNormalize(forward);
        if (f.IsZero())
        {
            return Quaternion<T>::Identity();
        }

        Vector3<T> r = SafeNormalize(Cross(up, -f));
        if (r.IsZero())
        {
            r = SafeNormalize(Orthogonal(f));
        }

        const Vector3<T> u = Cross(-f, r);

        Matrix3<T> rotation = Matrix3<T>::Identity();
        rotation.SetColumn(0, r);
        rotation.SetColumn(1, u);
        rotation.SetColumn(2, -f);

        return FromMatrix3(rotation);
    }

    /// Input: quaternion and vector.
    /// Output: vector rotated by quaternion.
    /// Task: rotate vectors robustly. This uses Inverse(quaternion), not merely
    /// Conjugate(), so non-unit input still behaves correctly.
    template<FloatingPoint T>
    [[nodiscard]]
    Vector3<T> Rotate(const Quaternion<T>& quaternion, const Vector3<T>& vector) noexcept
    {
        const Quaternion<T> pure { vector, T(0) };
        const Quaternion<T> result = quaternion * pure * Inverse(quaternion);
        return { result.x, result.y, result.z };
    }

    /// Input: orientation quaternion. Output: local +X axis in world space.
    /// Task: get the right direction for cameras, transforms, and movement.
    template<FloatingPoint T>
    [[nodiscard]]
    Vector3<T> Right(const Quaternion<T>& quaternion) noexcept
    {
        return Rotate(quaternion, Vector3<T>::Right());
    }

    /// Input: orientation quaternion. Output: local +Y axis in world space.
    /// Task: get the up direction for cameras, transforms, and billboards.
    template<FloatingPoint T>
    [[nodiscard]]
    Vector3<T> Up(const Quaternion<T>& quaternion) noexcept
    {
        return Rotate(quaternion, Vector3<T>::Up());
    }

    /// Input: orientation quaternion. Output: local -Z axis in world space.
    /// Task: get the engine forward direction.
    template<FloatingPoint T>
    [[nodiscard]]
    Vector3<T> Forward(const Quaternion<T>& quaternion) noexcept
    {
        return Rotate(quaternion, Vector3<T>::Forward());
    }

    /// Input: orientation quaternion. Output: local +Z axis in world space.
    /// Task: get the backward direction as the opposite of forward.
    template<FloatingPoint T>
    [[nodiscard]]
    Vector3<T> Backward(const Quaternion<T>& quaternion) noexcept
    {
        return -Forward(quaternion);
    }

    /// Input: orientation quaternion. Output: local -X axis in world space.
    /// Task: get the left direction as the opposite of right.
    template<FloatingPoint T>
    [[nodiscard]]
    Vector3<T> Left(const Quaternion<T>& quaternion) noexcept
    {
        return -Right(quaternion);
    }

    /// Input: orientation quaternion. Output: local -Y axis in world space.
    /// Task: get the down direction as the opposite of up.
    template<FloatingPoint T>
    [[nodiscard]]
    Vector3<T> Down(const Quaternion<T>& quaternion) noexcept
    {
        return -Up(quaternion);
    }

    /// Input: quaternion.
    /// Output: 3x3 rotation matrix.
    /// Task: convert orientation to a linear rotation matrix for transform math.
    template<FloatingPoint T>
    [[nodiscard]]
    Matrix3<T> ToMatrix3(const Quaternion<T>& quaternion) noexcept
    {
        const Quaternion<T> q = quaternion.Normalized();

        const T xx = q.x * q.x;
        const T yy = q.y * q.y;
        const T zz = q.z * q.z;
        const T xy = q.x * q.y;
        const T xz = q.x * q.z;
        const T yz = q.y * q.z;
        const T wx = q.w * q.x;
        const T wy = q.w * q.y;
        const T wz = q.w * q.z;

        return
        {
            T(1) - T(2) * (yy + zz),
            T(2) * (xy - wz),
            T(2) * (xz + wy),

            T(2) * (xy + wz),
            T(1) - T(2) * (xx + zz),
            T(2) * (yz - wx),

            T(2) * (xz - wy),
            T(2) * (yz + wx),
            T(1) - T(2) * (xx + yy)
        };
    }

    /// Input: quaternion.
    /// Output: 4x4 rotation matrix.
    /// Task: convert orientation to a homogeneous transform matrix.
    template<FloatingPoint T>
    [[nodiscard]]
    Matrix4<T> ToMatrix4(const Quaternion<T>& quaternion) noexcept
    {
        Matrix4<T> result = Matrix4<T>::Identity();
        const Matrix3<T> rotation = ToMatrix3(quaternion);

        for (std::size_t row = 0; row < Matrix3<T>::Rows; ++row)
        {
            for (std::size_t column = 0; column < Matrix3<T>::Columns; ++column)
            {
                result(row, column) = rotation(row, column);
            }
        }

        return result;
    }

    /// Input: 3x3 rotation matrix.
    /// Output: normalized quaternion.
    /// Task: convert matrix basis orientation into quaternion form.
    template<FloatingPoint T>
    [[nodiscard]]
    Quaternion<T> FromMatrix3(const Matrix3<T>& matrix) noexcept
    {
        const T epsilon = std::numeric_limits<T>::epsilon() * T(10);
        const T trace = Trace(matrix);
        Quaternion<T> result;

        if (trace > T(0))
        {
            const T s = std::sqrt(trace + T(1)) * T(2);
            if (std::abs(s) <= epsilon)
            {
                return Quaternion<T>::Identity();
            }

            result.w = T(0.25) * s;
            result.x = (matrix(2, 1) - matrix(1, 2)) / s;
            result.y = (matrix(0, 2) - matrix(2, 0)) / s;
            result.z = (matrix(1, 0) - matrix(0, 1)) / s;
        }
        else if (matrix(0, 0) > matrix(1, 1) && matrix(0, 0) > matrix(2, 2))
        {
            const T s = std::sqrt(T(1) + matrix(0, 0) - matrix(1, 1) - matrix(2, 2)) * T(2);
            if (std::abs(s) <= epsilon)
            {
                return Quaternion<T>::Identity();
            }

            result.w = (matrix(2, 1) - matrix(1, 2)) / s;
            result.x = T(0.25) * s;
            result.y = (matrix(0, 1) + matrix(1, 0)) / s;
            result.z = (matrix(0, 2) + matrix(2, 0)) / s;
        }
        else if (matrix(1, 1) > matrix(2, 2))
        {
            const T s = std::sqrt(T(1) + matrix(1, 1) - matrix(0, 0) - matrix(2, 2)) * T(2);
            if (std::abs(s) <= epsilon)
            {
                return Quaternion<T>::Identity();
            }

            result.w = (matrix(0, 2) - matrix(2, 0)) / s;
            result.x = (matrix(0, 1) + matrix(1, 0)) / s;
            result.y = T(0.25) * s;
            result.z = (matrix(1, 2) + matrix(2, 1)) / s;
        }
        else
        {
            const T s = std::sqrt(T(1) + matrix(2, 2) - matrix(0, 0) - matrix(1, 1)) * T(2);
            if (std::abs(s) <= epsilon)
            {
                return Quaternion<T>::Identity();
            }

            result.w = (matrix(1, 0) - matrix(0, 1)) / s;
            result.x = (matrix(0, 2) + matrix(2, 0)) / s;
            result.y = (matrix(1, 2) + matrix(2, 1)) / s;
            result.z = T(0.25) * s;
        }

        return result.Normalized();
    }

    /// Input: 4x4 matrix.
    /// Output: normalized quaternion from the upper-left 3x3 rotation portion.
    /// Task: convert transform rotation to quaternion. ExtractRotationMatrix()
    /// removes scale before conversion.
    template<FloatingPoint T>
    [[nodiscard]]
    Quaternion<T> FromMatrix4(const Matrix4<T>& matrix) noexcept
    {
        return FromMatrix3(ExtractRotationMatrix(matrix));
    }

    /// Input: quaternion.
    /// Output: Euler angles in radians: x = pitch, y = yaw, z = roll.
    /// Task: expose human-editable angles for tools. Euler angles are not ideal
    /// for internal rotation storage because they can suffer from gimbal lock.
    /// Convention: intrinsic XYZ/Tait-Bryan style as paired with FromEuler();
    /// keep both functions together if this convention changes.
    template<FloatingPoint T>
    [[nodiscard]]
    Vector3<T> ToEuler(const Quaternion<T>& quaternion) noexcept
    {
        const Quaternion<T> q = quaternion.Normalized();

        const T sinPitch = T(2) * ((q.w * q.x) + (q.y * q.z));
        const T cosPitch = T(1) - (T(2) * ((q.x * q.x) + (q.y * q.y)));
        const T pitch = std::atan2(sinPitch, cosPitch);

        const T sinYaw = T(2) * ((q.w * q.y) - (q.z * q.x));
        const T yaw = std::abs(sinYaw) >= T(1)
            ? std::copysign(T(std::numbers::pi / 2), sinYaw)
            : std::asin(sinYaw);

        const T sinRoll = T(2) * ((q.w * q.z) + (q.x * q.y));
        const T cosRoll = T(1) - (T(2) * ((q.y * q.y) + (q.z * q.z)));
        const T roll = std::atan2(sinRoll, cosRoll);

        return { pitch, yaw, roll };
    }

    /// Input: Euler angles in radians: x = pitch, y = yaw, z = roll.
    /// Output: normalized quaternion.
    /// Task: convert editor/tool angles into stable quaternion storage.
    /// Convention: paired with ToEuler(); x is pitch, y is yaw, z is roll.
    template<FloatingPoint T>
    [[nodiscard]]
    Quaternion<T> FromEuler(const Vector3<T>& euler) noexcept
    {
        const T cx = std::cos(euler.x * T(0.5));
        const T sx = std::sin(euler.x * T(0.5));
        const T cy = std::cos(euler.y * T(0.5));
        const T sy = std::sin(euler.y * T(0.5));
        const T cz = std::cos(euler.z * T(0.5));
        const T sz = std::sin(euler.z * T(0.5));

        return Quaternion<T>
        {
            (sx * cy * cz) - (cx * sy * sz),
            (cx * sy * cz) + (sx * cy * sz),
            (cx * cy * sz) - (sx * sy * cz),
            (cx * cy * cz) + (sx * sy * sz)
        }.Normalized();
    }

    /// Input: quaternion.
    /// Output: pair of normalized axis and angle in radians.
    /// Task: convert orientation to axis-angle representation for tools/debug UI.
    template<FloatingPoint T>
    [[nodiscard]]
    AxisAnglePair<T> ToAxisAngle(const Quaternion<T>& quaternion) noexcept
    {
        Quaternion<T> q = quaternion.Normalized();
        q.w = Clamp(q.w, T(-1), T(1));
        const T angle = T(2) * std::acos(q.w);
        const T denominator = std::sqrt(T(1) - (q.w * q.w));

        if (denominator < std::numeric_limits<T>::epsilon())
        {
            return { Vector3<T>::UnitX(), T(0) };
        }

        return { Vector3<T>{ q.x / denominator, q.y / denominator, q.z / denominator }, angle };
    }

    /// Input: endpoints and alpha.
    /// Output: component-wise linear interpolation.
    /// Task: cheap interpolation primitive. For rotations, prefer NLerp or SLerp.
    template<FloatingPoint T>
    [[nodiscard]]
    Quaternion<T> Lerp(const Quaternion<T>& lhs, const Quaternion<T>& rhs, T alpha) noexcept
    {
        return
        {
            lhs.x + ((rhs.x - lhs.x) * alpha),
            lhs.y + ((rhs.y - lhs.y) * alpha),
            lhs.z + ((rhs.z - lhs.z) * alpha),
            lhs.w + ((rhs.w - lhs.w) * alpha)
        };
    }

    /// Input: endpoints and alpha.
    /// Output: normalized linear interpolation along the shortest sign path.
    /// Task: fast rotation interpolation for animation/camera smoothing.
    template<FloatingPoint T>
    [[nodiscard]]
    Quaternion<T> NLerp(const Quaternion<T>& lhs, const Quaternion<T>& rhs, T alpha) noexcept
    {
        Quaternion<T> end = rhs;
        if (Dot(lhs, rhs) < T(0))
        {
            end = -rhs;
        }

        return Lerp(lhs, end, alpha).Normalized();
    }

    /// Input: endpoints and alpha.
    /// Output: spherical interpolation along the shortest sign path.
    /// Task: constant-angular-velocity interpolation for rotations.
    template<FloatingPoint T>
    [[nodiscard]]
    Quaternion<T> SLerp(const Quaternion<T>& lhs, const Quaternion<T>& rhs, T alpha) noexcept
    {
        Quaternion<T> end = rhs;
        T cosine = Dot(lhs, rhs);

        if (cosine < T(0))
        {
            end = -rhs;
            cosine = -cosine;
        }

        if (cosine > T(0.9995))
        {
            return NLerp(lhs, end, alpha);
        }

        cosine = Clamp(cosine, T(-1), T(1));
        const T angle = std::acos(cosine);
        const T sinAngle = std::sin(angle);

        const T lhsWeight = std::sin((T(1) - alpha) * angle) / sinAngle;
        const T rhsWeight = std::sin(alpha * angle) / sinAngle;

        return ((lhs * lhsWeight) + (end * rhsWeight)).Normalized();
    }

    /// Input: two quaternions.
    /// Output: angular distance in radians.
    /// Task: measure rotational difference. q and -q are treated as equivalent
    /// by taking the absolute dot product.
    template<FloatingPoint T>
    [[nodiscard]]
    T AngleBetween(const Quaternion<T>& lhs, const Quaternion<T>& rhs) noexcept
    {
        const T dot = Clamp(Dot(lhs.Normalized(), rhs.Normalized()), T(-1), T(1));
        return T(2) * std::acos(std::abs(dot));
    }

    /// Input: two quaternions and absolute epsilon.
    /// Output: true when they represent nearly the same rotation.
    /// Task: compare quaternions while handling the famous q == -q rotation
    /// equivalence. Component-wise equality alone is not enough for rotations.
    template<FloatingPoint T>
    [[nodiscard]]
    bool NearlyEqual(const Quaternion<T>& lhs, const Quaternion<T>& rhs, T epsilon) noexcept
    {
        const bool sameSign =
            NearlyEqual(lhs.x, rhs.x, epsilon) &&
            NearlyEqual(lhs.y, rhs.y, epsilon) &&
            NearlyEqual(lhs.z, rhs.z, epsilon) &&
            NearlyEqual(lhs.w, rhs.w, epsilon);

        const bool oppositeSign =
            NearlyEqual(lhs.x, -rhs.x, epsilon) &&
            NearlyEqual(lhs.y, -rhs.y, epsilon) &&
            NearlyEqual(lhs.z, -rhs.z, epsilon) &&
            NearlyEqual(lhs.w, -rhs.w, epsilon);

        return sameSign || oppositeSign;
    }

    namespace constants
    {
        inline constexpr Quatf IdentityQuatf = Quatf::Identity();
        inline constexpr Quatf ZeroQuatf = Quatf::Zero();
    }

    static_assert(Quaternion<float>::Size == 4);
    static_assert(std::is_trivially_copyable_v<Quatf>);
    static_assert(std::is_standard_layout_v<Quatf>);
    static_assert(sizeof(Quatf) == sizeof(float) * 4);
    static_assert(alignof(Quatf) == alignof(float));

} // namespace foundation::math
