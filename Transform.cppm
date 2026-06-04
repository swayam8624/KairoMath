module;

#include <cassert>
#include <cmath>
#include <concepts>
#include <limits>
#include <type_traits>

export module Kairo.Foundation.Math.Transform;

import Kairo.Foundation.Math.Vector;
import Kairo.Foundation.Math.Matrix;
import Kairo.Foundation.Math.Quaternion;


export namespace kairo::foundation::math
{
    template<FloatingPoint T>
    struct Transform;

    template<FloatingPoint T>
    [[nodiscard]]
    Matrix4<T> ToMatrix4(const Transform<T>& transform) noexcept;

    //=========================================================
    // Transform
    //=========================================================

    /// Position, orientation, and scale stored as TRS components.
    ///
    /// Design notes:
    /// - This is intentionally not just a Matrix4. Keeping Translation, Rotation,
    ///   and Scale separate makes editor tooling, animation blending, hierarchy
    ///   composition, and serialization easier and more numerically stable.
    /// - Rotation is a Quaternion because it avoids gimbal lock and interpolates
    ///   cleanly with SLerp/NLerp.
    /// - Scale is explicit and non-zero scale is required for inverse transforms.
    template<FloatingPoint T>
    struct Transform final
    {
        Vector3<T> Translation = Vector3<T>::Zero();
        Quaternion<T> Rotation = Quaternion<T>::Identity();
        Vector3<T> Scale = Vector3<T>::One();

        /// Input: none.
        /// Output: identity transform.
        /// Task: neutral transform for scene graph nodes and defaults.
        [[nodiscard]]
        static constexpr Transform Identity() noexcept
        {
            return {};
        }

        /// Input: translation.
        /// Output: transform with translation and identity rotation/scale.
        /// Task: concise construction for positioned objects.
        [[nodiscard]]
        static constexpr Transform FromTranslation(const Vector3<T>& translation) noexcept
        {
            return Transform(translation);
        }

        /// Input: rotation.
        /// Output: transform with rotation and zero translation/unit scale.
        /// Task: concise construction for orientation-only transforms.
        [[nodiscard]]
        static Transform FromRotation(const Quaternion<T>& rotation) noexcept
        {
            return Transform(rotation);
        }

        /// Input: scale.
        /// Output: transform with scale and identity translation/rotation.
        /// Task: concise construction for scale-only transforms.
        [[nodiscard]]
        static constexpr Transform FromScale(const Vector3<T>& scale) noexcept
        {
            Transform result;
            result.Scale = scale;
            return result;
        }

        /// Input: none.
        /// Output: identity transform.
        /// Task: deterministic default construction.
        constexpr Transform() noexcept = default;

        /// Input: translation.
        /// Output: translation-only transform.
        /// Task: common entity/camera construction path.
        constexpr explicit Transform(const Vector3<T>& translation) noexcept
            : Translation(translation)
        {
        }

        /// Input: rotation.
        /// Output: rotation-only transform.
        /// Task: common orientation construction path.
        explicit Transform(const Quaternion<T>& rotation) noexcept
            : Rotation(rotation.Normalized())
        {
        }

        /// Input: translation and rotation.
        /// Output: transform with unit scale.
        /// Task: common camera/entity construction path.
        Transform(const Vector3<T>& translation, const Quaternion<T>& rotation) noexcept
            : Translation(translation)
            , Rotation(rotation.Normalized())
        {
        }

        /// Input: translation, rotation, and scale.
        /// Output: transform containing supplied TRS components.
        /// Task: full explicit construction. Rotation is normalized because TRS
        /// transforms assume unit quaternion orientation.
        Transform(
            const Vector3<T>& translation,
            const Quaternion<T>& rotation,
            const Vector3<T>& scale) noexcept
            : Translation(translation)
            , Rotation(rotation.Normalized())
            , Scale(scale)
        {
        }

        /// Input: another transform.
        /// Output: exact component-wise equality.
        /// Task: deterministic exact comparison. Use NearlyEqual() for tests and
        /// runtime tolerance checks.
        [[nodiscard]]
        constexpr bool operator==(const Transform&) const noexcept = default;

        /// Input: new translation.
        /// Output: none; mutates this transform.
        /// Task: explicit setter for editor/property code.
        constexpr void SetTranslation(const Vector3<T>& translation) noexcept
        {
            Translation = translation;
        }

        /// Input: new rotation.
        /// Output: none; mutates this transform.
        /// Task: setter that preserves the unit-quaternion invariant.
        void SetRotation(const Quaternion<T>& rotation) noexcept
        {
            Rotation = rotation.Normalized();
        }

        /// Input: new scale.
        /// Output: none; mutates this transform.
        /// Task: explicit setter for editor/property code.
        constexpr void SetScale(const Vector3<T>& scale) noexcept
        {
            Scale = scale;
        }

        /// Input: local-space delta translation.
        /// Output: none; mutates Translation.
        /// Task: add an offset in the current parent/world space.
        constexpr void Translate(const Vector3<T>& offset) noexcept
        {
            Translation += offset;
        }

        /// Input: component-wise scale multiplier.
        /// Output: none; mutates Scale.
        /// Task: accumulate non-uniform scale.
        constexpr void ScaleBy(const Vector3<T>& scaleFactor) noexcept
        {
            Scale *= scaleFactor;
        }

        /// Input: local-space rotation delta.
        /// Output: none; mutates Rotation.
        /// Task: append a local rotation. `Rotation * delta` is the order wanted
        /// for cameras, scene nodes, and entities using local-space controls.
        void Rotate(const Quaternion<T>& delta) noexcept
        {
            Rotation = (Rotation * delta).Normalized();
        }

        /// Input: target point and approximate up direction.
        /// Output: none; mutates Rotation to face the target.
        /// Task: common camera, NPC, turret, and projectile facing helper.
        void LookAt(
            const Vector3<T>& target,
            const Vector3<T>& up = Vector3<T>::Up()) noexcept
        {
            Rotation = LookRotation(target - Translation, up);
        }

        /// Input: none.
        /// Output: local -Z direction transformed by Rotation.
        /// Task: get forward direction under the engine right-handed convention.
        [[nodiscard]]
        Vector3<T> Forward() const noexcept
        {
            return kairo::foundation::math::Forward(Rotation);
        }

        /// Input: none.
        /// Output: local +X direction transformed by Rotation.
        /// Task: get right direction for movement/camera code.
        [[nodiscard]]
        Vector3<T> Right() const noexcept
        {
            return kairo::foundation::math::Right(Rotation);
        }

        /// Input: none.
        /// Output: local +Y direction transformed by Rotation.
        /// Task: get up direction for movement/camera code.
        [[nodiscard]]
        Vector3<T> Up() const noexcept
        {
            return kairo::foundation::math::Up(Rotation);
        }

        /// Input: none. Output: opposite of Forward(). Task: convenience direction.
        [[nodiscard]]
        Vector3<T> Backward() const noexcept
        {
            return kairo::foundation::math::Backward(Rotation);
        }

        /// Input: none. Output: opposite of Right(). Task: convenience direction.
        [[nodiscard]]
        Vector3<T> Left() const noexcept
        {
            return kairo::foundation::math::Left(Rotation);
        }

        /// Input: none. Output: opposite of Up(). Task: convenience direction.
        [[nodiscard]]
        Vector3<T> Down() const noexcept
        {
            return kairo::foundation::math::Down(Rotation);
        }

        /// Input: none.
        /// Output: homogeneous world matrix for this transform.
        /// Task: high-frequency helper for render submission and scene traversal.
        [[nodiscard]]
        Matrix4<T> WorldMatrix() const noexcept
        {
            return ToMatrix4(*this);
        }

        /// Input: none.
        /// Output: 3x3 rotation matrix from Rotation.
        /// Task: expose basis matrix for physics, cameras, and debug drawing.
        [[nodiscard]]
        Matrix3<T> RotationMatrix() const noexcept
        {
            return ToMatrix3(Rotation);
        }

        /// Input: optional tolerance.
        /// Output: true when this transform is nearly identity.
        /// Task: tolerate floating-point drift in tests and runtime checks.
        [[nodiscard]]
        bool IsIdentity(
            T epsilon = std::numeric_limits<T>::epsilon() * T(10)) const noexcept
        {
            return
                NearlyEqual(Translation, Vector3<T>::Zero(), epsilon) &&
                NearlyEqual(Rotation, Quaternion<T>::Identity(), epsilon) &&
                NearlyEqual(Scale, Vector3<T>::One(), epsilon);
        }
    };

    using Transformf = Transform<float>;
    using Transformd = Transform<double>;

    //=========================================================
    // Conversion
    //=========================================================

    /// Input: transform.
    /// Output: row-major matrix equivalent to Translation * Rotation * Scale.
    /// Task: convert TRS into the matrix form used by renderers and shaders.
    template<FloatingPoint T>
    [[nodiscard]]
    Matrix4<T> ToMatrix4(const Transform<T>& transform) noexcept
    {
        const Matrix4<T> translation = MakeTranslation(transform.Translation);
        const Matrix4<T> rotation = ToMatrix4(transform.Rotation);
        const Matrix4<T> scale = MakeScale(transform.Scale);

        return translation * rotation * scale;
    }

    //=========================================================
    // Transform Operations
    //=========================================================

    /// Input: transform and local point.
    /// Output: world point.
    /// Task: apply scale, then rotation, then translation.
    template<FloatingPoint T>
    [[nodiscard]]
    Vector3<T> TransformPoint(const Transform<T>& transform, const Vector3<T>& point) noexcept
    {
        Vector3<T> result = point;
        result *= transform.Scale;
        result = Rotate(transform.Rotation, result);
        result += transform.Translation;
        return result;
    }

    /// Input: transform and local direction.
    /// Output: world direction.
    /// Task: apply rotation only; translation and scale are intentionally ignored.
    template<FloatingPoint T>
    [[nodiscard]]
    Vector3<T> TransformDirection(const Transform<T>& transform, const Vector3<T>& direction) noexcept
    {
        return Rotate(transform.Rotation, direction);
    }

    /// Input: transform and local vector.
    /// Output: world vector.
    /// Task: apply scale and rotation but ignore translation.
    template<FloatingPoint T>
    [[nodiscard]]
    Vector3<T> TransformVector(const Transform<T>& transform, const Vector3<T>& vector) noexcept
    {
        Vector3<T> result = vector;
        result *= transform.Scale;
        return Rotate(transform.Rotation, result);
    }

    /// Input: transform and local normal.
    /// Output: world normal.
    /// Task: transform normals correctly under non-uniform scale by reusing the
    /// matrix inverse-transpose implementation. Later this can be optimized with
    /// a dedicated InverseTranspose3x3().
    template<FloatingPoint T>
    [[nodiscard]]
    Vector3<T> TransformNormal(const Transform<T>& transform, const Vector3<T>& normal) noexcept
    {
        return kairo::foundation::math::TransformNormal(ToMatrix4(transform), normal);
    }

    /// Input: transform and local point.
    /// Output: world point.
    /// Task: semantic alias for TransformPoint().
    template<FloatingPoint T>
    [[nodiscard]]
    Vector3<T> LocalToWorld(const Transform<T>& transform, const Vector3<T>& point) noexcept
    {
        return TransformPoint(transform, point);
    }

    /// Input: transform and world point.
    /// Output: local point.
    /// Task: undo translation, rotation, and scale.
    template<FloatingPoint T>
    [[nodiscard]]
    Vector3<T> WorldToLocal(const Transform<T>& transform, const Vector3<T>& point) noexcept
    {
        assert(std::abs(transform.Scale.x) > std::numeric_limits<T>::epsilon());
        assert(std::abs(transform.Scale.y) > std::numeric_limits<T>::epsilon());
        assert(std::abs(transform.Scale.z) > std::numeric_limits<T>::epsilon());

        const Vector3<T> translated = point - transform.Translation;
        const Vector3<T> rotated = Rotate(Inverse(transform.Rotation), translated);

        return
        {
            rotated.x / transform.Scale.x,
            rotated.y / transform.Scale.y,
            rotated.z / transform.Scale.z
        };
    }

    /// Input: transform and local direction.
    /// Output: world direction.
    /// Task: semantic alias for TransformDirection().
    template<FloatingPoint T>
    [[nodiscard]]
    Vector3<T> LocalToWorldDirection(const Transform<T>& transform, const Vector3<T>& direction) noexcept
    {
        return TransformDirection(transform, direction);
    }

    /// Input: transform and world direction.
    /// Output: local direction.
    /// Task: undo rotation for direction-only values.
    template<FloatingPoint T>
    [[nodiscard]]
    Vector3<T> WorldToLocalDirection(const Transform<T>& transform, const Vector3<T>& direction) noexcept
    {
        return Rotate(Inverse(transform.Rotation), direction);
    }

    //=========================================================
    // Composition
    //=========================================================

    /// Input: parent and child transforms.
    /// Output: child transformed into parent space.
    /// Task: compose scene graph transforms. Parent * Child means child local
    /// transform is applied first, then parent.
    template<FloatingPoint T>
    [[nodiscard]]
    Transform<T> operator*(const Transform<T>& parent, const Transform<T>& child) noexcept
    {
        Transform<T> result;
        result.Scale = parent.Scale * child.Scale;
        result.Rotation = (parent.Rotation * child.Rotation).Normalized();
        result.Translation = TransformPoint(parent, child.Translation);
        return result;
    }

    /// Input: right-hand transform.
    /// Output: lhs after composition.
    /// Task: in-place scene graph composition.
    template<FloatingPoint T>
    Transform<T>& operator*=(Transform<T>& lhs, const Transform<T>& rhs) noexcept
    {
        lhs = lhs * rhs;
        return lhs;
    }

    /// Input: invertible transform with uniform scale.
    /// Output: inverse transform.
    /// Task: build a TRS inverse for common scene graph transforms.
    ///
    /// Important: a general rotated non-uniform scale inverse contains shear and
    /// cannot always be represented exactly as another TRS transform. Use
    /// WorldToLocal() for point conversion or Inverse(ToMatrix4(transform)) when
    /// an exact non-uniform-scale inverse matrix is required.
    template<FloatingPoint T>
    [[nodiscard]]
    Transform<T> Inverse(const Transform<T>& transform) noexcept
    {
        assert(std::abs(transform.Scale.x) > std::numeric_limits<T>::epsilon());
        assert(std::abs(transform.Scale.y) > std::numeric_limits<T>::epsilon());
        assert(std::abs(transform.Scale.z) > std::numeric_limits<T>::epsilon());
        assert(HasUniformScale(transform));

        Transform<T> result;
        result.Scale =
        {
            T(1) / transform.Scale.x,
            T(1) / transform.Scale.y,
            T(1) / transform.Scale.z
        };

        result.Rotation = Inverse(transform.Rotation);
        result.Translation = Rotate(result.Rotation, -transform.Translation);
        result.Translation *= result.Scale;
        return result;
    }

    /// Input: two transforms and interpolation alpha.
    /// Output: blended transform.
    /// Task: interpolate translation/scale linearly and rotation spherically.
    /// This may later move to Foundation.Math.Interpolation, but it is useful
    /// immediately for animation, cameras, and editor gizmos.
    template<FloatingPoint T>
    [[nodiscard]]
    Transform<T> Interpolate(const Transform<T>& a, const Transform<T>& b, T alpha) noexcept
    {
        Transform<T> result;
        result.Translation = Lerp(a.Translation, b.Translation, alpha);
        result.Scale = Lerp(a.Scale, b.Scale, alpha);
        result.Rotation = SLerp(a.Rotation, b.Rotation, alpha);
        return result;
    }

    //=========================================================
    // Queries
    //=========================================================

    /// Input: transform.
    /// Output: translation reference.
    /// Task: consistent extraction API alongside Matrix extraction helpers.
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr const Vector3<T>& ExtractTranslation(const Transform<T>& transform) noexcept
    {
        return transform.Translation;
    }

    /// Input: transform.
    /// Output: rotation reference.
    /// Task: consistent extraction API alongside Matrix extraction helpers.
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr const Quaternion<T>& ExtractRotation(const Transform<T>& transform) noexcept
    {
        return transform.Rotation;
    }

    /// Input: transform.
    /// Output: scale reference.
    /// Task: consistent extraction API alongside Matrix extraction helpers.
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr const Vector3<T>& ExtractScale(const Transform<T>& transform) noexcept
    {
        return transform.Scale;
    }

    /// Input: transform and tolerance.
    /// Output: true when all scale axes are nearly equal.
    /// Task: detect when simpler uniform-scale math is valid.
    template<FloatingPoint T>
    [[nodiscard]]
    bool HasUniformScale(
        const Transform<T>& transform,
        T epsilon = std::numeric_limits<T>::epsilon() * T(10)) noexcept
    {
        return
            NearlyEqual(transform.Scale.x, transform.Scale.y, epsilon) &&
            NearlyEqual(transform.Scale.y, transform.Scale.z, epsilon);
    }

    /// Input: transform and tolerance.
    /// Output: true when rotation is normalized and scale is invertible.
    /// Task: validate transforms before physics/render submission.
    template<FloatingPoint T>
    [[nodiscard]]
    bool IsValid(
        const Transform<T>& transform,
        T epsilon = std::numeric_limits<T>::epsilon() * T(10)) noexcept
    {
        return
            transform.Rotation.IsNormalized(epsilon) &&
            std::abs(transform.Scale.x) > epsilon &&
            std::abs(transform.Scale.y) > epsilon &&
            std::abs(transform.Scale.z) > epsilon;
    }

    /// Input: two transforms and tolerance.
    /// Output: true when translation, rotation, and scale are nearly equal.
    /// Task: transform comparison for tests, animation validation, and scene diffing.
    template<FloatingPoint T>
    [[nodiscard]]
    bool NearlyEqual(
        const Transform<T>& lhs,
        const Transform<T>& rhs,
        T epsilon = std::numeric_limits<T>::epsilon() * T(10)) noexcept
    {
        return
            NearlyEqual(lhs.Translation, rhs.Translation, epsilon) &&
            NearlyEqual(lhs.Rotation, rhs.Rotation, epsilon) &&
            NearlyEqual(lhs.Scale, rhs.Scale, epsilon);
    }

    namespace constants
    {
        inline constexpr Transformf IdentityTransformf = Transformf::Identity();
        inline constexpr Transformd IdentityTransformd = Transformd::Identity();
    }

    static_assert(std::is_standard_layout_v<Transformf>);

} // namespace kairo::foundation::math
