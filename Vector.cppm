module;

#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

export module Kairo.Foundation.Math.Vector;


export namespace kairo::foundation::math
{
    //=========================================================
    // Concepts
    //=========================================================

    /// Input: a candidate scalar type.
    /// Output: true when the type is usable as a vector component.
    /// Task: keep this module restricted to integer and floating-point value
    /// types. Engine vectors should stay trivially copyable, allocation-free,
    /// and friendly to GPU upload/serialization paths.
    template<typename T>
    concept Arithmetic =
        std::integral<T> ||
        std::floating_point<T>;

    /// Input: a candidate scalar type.
    /// Output: true for floating-point vector component types.
    /// Task: gate operations such as Length(), Normalize(), Reflect(), and
    /// Refract(), where integer arithmetic would silently produce bad math.
    template<typename T>
    concept FloatingPoint =
        std::floating_point<T>;

    //=========================================================
    // Forward Declarations
    //=========================================================

    template<Arithmetic T>
    struct Vector2;

    template<Arithmetic T>
    struct Vector3;

    template<Arithmetic T>
    struct Vector4;

    //=========================================================
    // Vector2
    //=========================================================

    /// A 2D mathematical vector.
    ///
    /// Design notes:
    /// - This is a struct because a vector is a transparent value type, not an
    ///   object with hidden invariants. Direct field access is useful in engine
    ///   math, serialization, debugging, editor tooling, and GPU upload code.
    /// - The type owns no memory and performs no allocation.
    /// - The public API intentionally avoids SIMD-specific alignment. A future
    ///   SIMDVector4f can be introduced as an implementation detail once
    ///   vectors, matrices, quaternions, and transforms are stable.
    template<Arithmetic T>
    struct Vector2 final
    {
        using ValueType = T;

        static constexpr std::size_t Size = 2;

        /// Input: none.
        /// Output: (0, 0).
        /// Task: provide a named zero vector for positions, offsets, UVs, and
        /// fallback values. Prefer this over spelling magic zeros repeatedly.
        [[nodiscard]]
        static constexpr Vector2 Zero() noexcept
        {
            return { T(0), T(0) };
        }

        /// Input: none.
        /// Output: (1, 1).
        /// Task: provide a named all-ones vector for scale values, masks, and
        /// component-wise arithmetic.
        [[nodiscard]]
        static constexpr Vector2 One() noexcept
        {
            return { T(1), T(1) };
        }

        /// Input: none.
        /// Output: +X unit vector.
        /// Task: provide a canonical basis vector for coordinate construction.
        [[nodiscard]]
        static constexpr Vector2 UnitX() noexcept
        {
            return { T(1), T(0) };
        }

        /// Input: none.
        /// Output: +Y unit vector.
        /// Task: provide a canonical basis vector for coordinate construction.
        [[nodiscard]]
        static constexpr Vector2 UnitY() noexcept
        {
            return { T(0), T(1) };
        }

        T x;
        T y;

        //-----------------------------------------------------
        // Constructors
        //-----------------------------------------------------

        /// Input: none.
        /// Output: zero-initialized vector.
        /// Task: default construction should be safe for beginner code and
        /// deterministic for engine code.
        constexpr Vector2() noexcept
            : x(T(0))
            , y(T(0))
        {
        }

        /// Input: one scalar.
        /// Output: vector with every component set to scalar.
        /// Task: support concise construction of uniform scale/mask values.
        constexpr explicit Vector2(T scalar) noexcept
            : x(scalar)
            , y(scalar)
        {
        }

        /// Input: x and y components.
        /// Output: vector containing the supplied components.
        /// Task: construct a transparent 2D value without hidden conversions.
        constexpr Vector2(T xValue, T yValue) noexcept
            : x(xValue)
            , y(yValue)
        {
        }

        //-----------------------------------------------------
        // Element Access
        //-----------------------------------------------------

        /// Input: component index in [0, Size).
        /// Output: mutable reference to the requested component.
        /// Task: provide array-style access for generic algorithms. The assert
        /// catches programmer errors in debug builds; it is not a recoverable
        /// runtime failure, so std::expected would be the wrong tool here.
        [[nodiscard]]
        constexpr T& operator[](std::size_t index) noexcept
        {
            assert(index < Size);
            return Data()[index];
        }

        /// Input: component index in [0, Size).
        /// Output: const reference to the requested component.
        /// Task: provide read-only array-style access for generic algorithms.
        [[nodiscard]]
        constexpr const T& operator[](std::size_t index) const noexcept
        {
            assert(index < Size);
            return Data()[index];
        }

        /// Input: none.
        /// Output: pointer to the first component.
        /// Task: expose contiguous component storage for graphics APIs,
        /// serialization, and span construction. This type intentionally stores
        /// only adjacent scalar members; static assertions at the end verify the
        /// expected size for the supported engine aliases.
        [[nodiscard]]
        constexpr T* Data() noexcept
        {
            return &x;
        }

        /// Input: none.
        /// Output: const pointer to the first component.
        /// Task: const overload for upload/read-only serialization paths.
        [[nodiscard]]
        constexpr const T* Data() const noexcept
        {
            return &x;
        }

        //-----------------------------------------------------
        // Unary Operators
        //-----------------------------------------------------

        /// Input: this vector.
        /// Output: unchanged copy.
        /// Task: keep arithmetic expressions symmetric with unary minus.
        [[nodiscard]]
        constexpr Vector2 operator+() const noexcept
        {
            return *this;
        }

        /// Input: this vector.
        /// Output: component-wise negation.
        /// Task: reverse direction without modifying the source vector.
        [[nodiscard]]
        constexpr Vector2 operator-() const noexcept
        {
            return { -x, -y };
        }

        //-----------------------------------------------------
        // Compound Assignment
        //-----------------------------------------------------

        /// Input: right-hand vector.
        /// Output: this vector after component-wise addition.
        /// Task: mutate in place to avoid temporary-heavy code in hot loops.
        constexpr Vector2& operator+=(const Vector2& rhs) noexcept
        {
            x += rhs.x;
            y += rhs.y;
            return *this;
        }

        /// Input: right-hand vector.
        /// Output: this vector after component-wise subtraction.
        /// Task: mutate in place for displacement and delta calculations.
        constexpr Vector2& operator-=(const Vector2& rhs) noexcept
        {
            x -= rhs.x;
            y -= rhs.y;
            return *this;
        }

        /// Input: scalar multiplier.
        /// Output: this vector after uniform scaling.
        /// Task: support fast in-place scaling with no allocation.
        constexpr Vector2& operator*=(T scalar) noexcept
        {
            x *= scalar;
            y *= scalar;
            return *this;
        }

        /// Input: non-zero scalar divisor.
        /// Output: this vector after uniform division.
        /// Task: support in-place division. Division by zero is a programmer
        /// error and is asserted in debug builds.
        constexpr Vector2& operator/=(T scalar) noexcept
        {
            assert(scalar != T(0));
            if constexpr (FloatingPoint<T>)
            {
                const T inverseScalar = T(1) / scalar;
                x *= inverseScalar;
                y *= inverseScalar;
            }
            else
            {
                x /= scalar;
                y /= scalar;
            }
            return *this;
        }

        /// Input: right-hand vector.
        /// Output: this vector after component-wise multiplication.
        /// Task: support Hadamard-style masks/scales used by shading and color
        /// math without requiring a separate temporary.
        constexpr Vector2& operator*=(const Vector2& rhs) noexcept
        {
            x *= rhs.x;
            y *= rhs.y;
            return *this;
        }

        //-----------------------------------------------------
        // Comparison
        //-----------------------------------------------------

        /// Input: another vector.
        /// Output: true when every component compares exactly equal.
        /// Task: exact comparison for integers and deterministic values. Use
        /// NearlyEqual() for floating-point tolerance comparisons.
        [[nodiscard]]
        constexpr bool operator==(const Vector2&) const noexcept = default;

        //-----------------------------------------------------
        // Math
        //-----------------------------------------------------

        /// Input: this vector.
        /// Output: squared Euclidean length.
        /// Task: avoid sqrt when only relative distance/magnitude is needed.
        [[nodiscard]]
        constexpr T LengthSquared() const noexcept
        {
            return (x * x) + (y * y);
        }

        /// Input: this floating-point vector.
        /// Output: Euclidean length.
        /// Task: compute magnitude. Floating-point only because integer sqrt
        /// would either truncate or require a different API.
        [[nodiscard]]
        T Length() const noexcept
            requires FloatingPoint<T>
        {
            return std::sqrt(LengthSquared());
        }

        /// Input: this non-zero floating-point vector.
        /// Output: reciprocal of Length().
        /// Task: expose a useful building block for optimized math code. Callers
        /// are responsible for avoiding zero-length input.
        [[nodiscard]]
        T LengthInverse() const noexcept
            requires FloatingPoint<T>
        {
            const T length = Length();
            return length <= std::numeric_limits<T>::epsilon()
                ? T(0)
                : T(1) / length;
        }

        /// Input: this floating-point vector.
        /// Output: unit-length vector, or zero when the input is too small.
        /// Task: return a normalized copy while avoiding division by values near
        /// zero. Use SafeNormalize() when a custom fallback is needed.
        [[nodiscard]]
        Vector2 Normalized() const noexcept
            requires FloatingPoint<T>
        {
            const T length = Length();
            if (length <= std::numeric_limits<T>::epsilon())
            {
                return Zero();
            }
            return *this / length;
        }

        /// Input: this floating-point vector.
        /// Output: none; this vector becomes unit length when possible.
        /// Task: normalize in place. Zero and near-zero vectors are left as-is to
        /// avoid creating infinities or NaNs in downstream engine systems.
        void Normalize() noexcept
            requires FloatingPoint<T>
        {
            const T length = Length();
            if (length <= std::numeric_limits<T>::epsilon())
            {
                return;
            }
            *this /= length;
        }

        /// Input: this vector.
        /// Output: true when every component is exactly zero.
        /// Task: cheap exact zero check. For floating-point tolerance checks,
        /// compare LengthSquared() or use NearlyEqual().
        [[nodiscard]]
        constexpr bool IsZero() const noexcept
        {
            return x == T(0) && y == T(0);
        }
    };

    //=========================================================
    // Vector3
    //=========================================================

    /// A 3D mathematical vector for positions, directions, normals, velocities,
    /// forces, RGB colors, and future geometry/ray-tracing systems.
    template<Arithmetic T>
    struct Vector3 final
    {
        using ValueType = T;

        static constexpr std::size_t Size = 3;

        /// Input: none. Output: (0, 0, 0). Task: named zero value.
        [[nodiscard]]
        static constexpr Vector3 Zero() noexcept
        {
            return { T(0), T(0), T(0) };
        }

        /// Input: none. Output: (1, 1, 1). Task: named all-ones value.
        [[nodiscard]]
        static constexpr Vector3 One() noexcept
        {
            return { T(1), T(1), T(1) };
        }

        /// Input: none. Output: +X basis vector. Task: basis construction.
        [[nodiscard]]
        static constexpr Vector3 UnitX() noexcept
        {
            return { T(1), T(0), T(0) };
        }

        /// Input: none. Output: +Y basis vector. Task: basis construction.
        [[nodiscard]]
        static constexpr Vector3 UnitY() noexcept
        {
            return { T(0), T(1), T(0) };
        }

        /// Input: none. Output: +Z basis vector. Task: basis construction.
        [[nodiscard]]
        static constexpr Vector3 UnitZ() noexcept
        {
            return { T(0), T(0), T(1) };
        }

        /// Input: none. Output: engine right direction (+X).
        /// Task: semantic alias for gameplay/camera code.
        [[nodiscard]]
        static constexpr Vector3 Right() noexcept
        {
            return UnitX();
        }

        /// Input: none. Output: engine left direction (-X).
        /// Task: semantic alias for gameplay/camera code.
        [[nodiscard]]
        static constexpr Vector3 Left() noexcept
        {
            return { T(-1), T(0), T(0) };
        }

        /// Input: none. Output: engine up direction (+Y).
        /// Task: semantic alias for gameplay/camera code.
        [[nodiscard]]
        static constexpr Vector3 Up() noexcept
        {
            return UnitY();
        }

        /// Input: none. Output: engine down direction (-Y).
        /// Task: semantic alias for gameplay/camera code.
        [[nodiscard]]
        static constexpr Vector3 Down() noexcept
        {
            return { T(0), T(-1), T(0) };
        }

        /// Input: none. Output: engine forward direction (-Z).
        /// Task: semantic alias for right-handed camera/gameplay code.
        [[nodiscard]]
        static constexpr Vector3 Forward() noexcept
        {
            return { T(0), T(0), T(-1) };
        }

        /// Input: none. Output: engine backward direction (+Z).
        /// Task: semantic alias for right-handed camera/gameplay code.
        [[nodiscard]]
        static constexpr Vector3 Backward() noexcept
        {
            return UnitZ();
        }

        T x;
        T y;
        T z;

        //-----------------------------------------------------
        // Constructors
        //-----------------------------------------------------

        /// Input: none.
        /// Output: zero-initialized vector.
        /// Task: deterministic default construction.
        constexpr Vector3() noexcept
            : x(T(0))
            , y(T(0))
            , z(T(0))
        {
        }

        /// Input: one scalar.
        /// Output: vector with every component set to scalar.
        /// Task: concise construction for uniform scale/mask values.
        constexpr explicit Vector3(T scalar) noexcept
            : x(scalar)
            , y(scalar)
            , z(scalar)
        {
        }

        /// Input: x, y, and z components.
        /// Output: vector containing the supplied components.
        /// Task: construct a transparent 3D value.
        constexpr Vector3(T xValue, T yValue, T zValue) noexcept
            : x(xValue)
            , y(yValue)
            , z(zValue)
        {
        }

        /// Input: xy vector and z component.
        /// Output: vector composed from both inputs.
        /// Task: support dimensional promotion for geometry and texture code.
        constexpr Vector3(const Vector2<T>& xy, T zValue) noexcept
            : x(xy.x)
            , y(xy.y)
            , z(zValue)
        {
        }

        //-----------------------------------------------------
        // Element Access
        //-----------------------------------------------------


        /// Input: index in [0, Size). Output: mutable component reference.
        /// Task: array-style access for generic math code.
        [[nodiscard]]
        constexpr T& operator[](std::size_t index) noexcept
        {
            assert(index < Size);
            return Data()[index];
        }

        /// Input: index in [0, Size). Output: const component reference.
        /// Task: read-only array-style access for generic math code.
        [[nodiscard]]
        constexpr const T& operator[](std::size_t index) const noexcept
        {
            assert(index < Size);
            return Data()[index];
        }

        /// Input: none.
        /// Output: pointer to x, the first component.
        /// Task: provide direct component upload access for Vulkan, OpenGL,
        /// Metal argument buffers, binary serialization, and tooling.
        [[nodiscard]]
        constexpr T* Data() noexcept
        {
            return &x;
        }

        /// Input: none. Output: const pointer to x. Task: read-only data access.
        [[nodiscard]]
        constexpr const T* Data() const noexcept
        {
            return &x;
        }

        //-----------------------------------------------------
        // Operations
        //-----------------------------------------------------


        /// Input: this vector. Output: unchanged copy. Task: expression symmetry.
        [[nodiscard]]
        constexpr Vector3 operator+() const noexcept
        {
            return *this;
        }

        /// Input: this vector. Output: component-wise negation.
        /// Task: reverse a direction or displacement.
        [[nodiscard]]
        constexpr Vector3 operator-() const noexcept
        {
            return { -x, -y, -z };
        }

        /// Input: rhs. Output: this += rhs. Task: in-place vector addition.
        constexpr Vector3& operator+=(const Vector3& rhs) noexcept
        {
            x += rhs.x;
            y += rhs.y;
            z += rhs.z;
            return *this;
        }

        /// Input: rhs. Output: this -= rhs. Task: in-place vector subtraction.
        constexpr Vector3& operator-=(const Vector3& rhs) noexcept
        {
            x -= rhs.x;
            y -= rhs.y;
            z -= rhs.z;
            return *this;
        }

        /// Input: scalar. Output: this *= scalar. Task: in-place uniform scale.
        constexpr Vector3& operator*=(T scalar) noexcept
        {
            x *= scalar;
            y *= scalar;
            z *= scalar;
            return *this;
        }

        /// Input: non-zero scalar. Output: this /= scalar.
        /// Task: in-place uniform division; asserts on programmer error.
        constexpr Vector3& operator/=(T scalar) noexcept
        {
            assert(scalar != T(0));
            if constexpr (FloatingPoint<T>)
            {
                const T inverseScalar = T(1) / scalar;
                x *= inverseScalar;
                y *= inverseScalar;
                z *= inverseScalar;
            }
            else
            {
                x /= scalar;
                y /= scalar;
                z /= scalar;
            }
            return *this;
        }

        /// Input: rhs. Output: component-wise product stored in this vector.
        /// Task: in-place Hadamard multiplication.
        constexpr Vector3& operator*=(const Vector3& rhs) noexcept
        {
            x *= rhs.x;
            y *= rhs.y;
            z *= rhs.z;
            return *this;
        }

        //-----------------------------------------------------
        // Comparisons
        //-----------------------------------------------------


        /// Input: another vector. Output: exact component equality.
        /// Task: deterministic exact comparison; use NearlyEqual() for floats.
        [[nodiscard]]
        constexpr bool operator==(const Vector3&) const noexcept = default;

        /// Input: this vector. Output: squared Euclidean length.
        /// Task: magnitude comparison without sqrt.
        [[nodiscard]]
        constexpr T LengthSquared() const noexcept
        {
            return (x * x) + (y * y) + (z * z);
        }

        /// Input: this floating-point vector. Output: Euclidean length.
        /// Task: compute magnitude for directions, normals, and velocities.
        [[nodiscard]]
        T Length() const noexcept
            requires FloatingPoint<T>
        {
            return std::sqrt(LengthSquared());
        }

        /// Input: non-zero floating-point vector.
        /// Output: reciprocal length.
        /// Task: useful optimization hook for normalization-heavy code.
        [[nodiscard]]
        T LengthInverse() const noexcept
            requires FloatingPoint<T>
        {
            const T length = Length();
            return length <= std::numeric_limits<T>::epsilon()
                ? T(0)
                : T(1) / length;
        }

        /// Input: this floating-point vector.
        /// Output: normalized copy, or zero for near-zero input.
        /// Task: safe value-returning normalization.
        [[nodiscard]]
        Vector3 Normalized() const noexcept
            requires FloatingPoint<T>
        {
            const T length = Length();
            if (length <= std::numeric_limits<T>::epsilon())
            {
                return Zero();
            }
            return *this / length;
        }

        /// Input: this floating-point vector.
        /// Output: none; mutates this vector when length is usable.
        /// Task: in-place normalization for hot loops.
        void Normalize() noexcept
            requires FloatingPoint<T>
        {
            const T length = Length();
            if (length <= std::numeric_limits<T>::epsilon())
            {
                return;
            }
            *this /= length;
        }

        /// Input: this vector. Output: true when all components are exactly zero.
        /// Task: cheap exact zero check.
        [[nodiscard]]
        constexpr bool IsZero() const noexcept
        {
            return x == T(0) && y == T(0) && z == T(0);
        }

        /// Input: this vector. Output: (x, y).
        /// Task: dimensional projection for UV/screen-space code.
        [[nodiscard]]
        constexpr Vector2<T> XY() const noexcept
        {
            return { x, y };
        }
    };

    //=========================================================
    // Vector4
    //=========================================================

    /// A 4D mathematical vector for homogeneous coordinates, RGBA values, and
    /// future matrix/quaternion support.
    ///
    /// This public type is deliberately not alignas(16). For example,
    /// Vector4<double> would become awkwardly aligned for little benefit. SIMD
    /// should be introduced later through dedicated internal SIMDVector4f /
    /// SIMDMatrix4f types instead of forcing every public vector into one ABI.
    template<Arithmetic T>
    struct Vector4 final
    {
        using ValueType = T;

        static constexpr std::size_t Size = 4;

        /// Input: none. Output: (0, 0, 0, 0). Task: named zero value.
        [[nodiscard]]
        static constexpr Vector4 Zero() noexcept
        {
            return { T(0), T(0), T(0), T(0) };
        }

        /// Input: none. Output: (1, 1, 1, 1). Task: named all-ones value.
        [[nodiscard]]
        static constexpr Vector4 One() noexcept
        {
            return { T(1), T(1), T(1), T(1) };
        }

        /// Input: none. Output: +X basis vector. Task: basis construction.
        [[nodiscard]]
        static constexpr Vector4 UnitX() noexcept
        {
            return { T(1), T(0), T(0), T(0) };
        }

        /// Input: none. Output: +Y basis vector. Task: basis construction.
        [[nodiscard]]
        static constexpr Vector4 UnitY() noexcept
        {
            return { T(0), T(1), T(0), T(0) };
        }

        /// Input: none. Output: +Z basis vector. Task: basis construction.
        [[nodiscard]]
        static constexpr Vector4 UnitZ() noexcept
        {
            return { T(0), T(0), T(1), T(0) };
        }

        /// Input: none. Output: +W basis vector. Task: homogeneous coordinate
        /// and matrix-column construction.
        [[nodiscard]]
        static constexpr Vector4 UnitW() noexcept
        {
            return { T(0), T(0), T(0), T(1) };
        }

        T x;
        T y;
        T z;
        T w;

        //-----------------------------------------------------
        // Constructors
        //-----------------------------------------------------

        /// Input: none. Output: zero-initialized vector.
        /// Task: deterministic default construction.
        constexpr Vector4() noexcept
            : x(T(0))
            , y(T(0))
            , z(T(0))
            , w(T(0))
        {
        }

        /// Input: one scalar. Output: all components set to scalar.
        /// Task: concise uniform construction.
        constexpr explicit Vector4(T scalar) noexcept
            : x(scalar)
            , y(scalar)
            , z(scalar)
            , w(scalar)
        {
        }

        /// Input: x, y, z, and w components.
        /// Output: vector containing the supplied components.
        /// Task: construct a transparent 4D value.
        constexpr Vector4(T xValue, T yValue, T zValue, T wValue) noexcept
            : x(xValue)
            , y(yValue)
            , z(zValue)
            , w(wValue)
        {
        }

        /// Input: xyz vector and w component.
        /// Output: vector composed from both inputs.
        /// Task: dimensional promotion for homogeneous coordinates and colors.
        constexpr Vector4(const Vector3<T>& xyz, T wValue) noexcept
            : x(xyz.x)
            , y(xyz.y)
            , z(xyz.z)
            , w(wValue)
        {
        }

        //-----------------------------------------------------
        // Element Access
        //-----------------------------------------------------


        /// Input: index in [0, Size). Output: mutable component reference.
        /// Task: array-style generic access.
        [[nodiscard]]
        constexpr T& operator[](std::size_t index) noexcept
        {
            assert(index < Size);
            return Data()[index];
        }

        /// Input: index in [0, Size). Output: const component reference.
        /// Task: read-only array-style generic access.
        [[nodiscard]]
        constexpr const T& operator[](std::size_t index) const noexcept
        {
            assert(index < Size);
            return Data()[index];
        }

        /// Input: none. Output: pointer to x.
        /// Task: expose contiguous values for graphics APIs and serialization.
        [[nodiscard]]
        constexpr T* Data() noexcept
        {
            return &x;
        }

        /// Input: none. Output: const pointer to x. Task: read-only data access.
        [[nodiscard]]
        constexpr const T* Data() const noexcept
        {
            return &x;
        }

        //-----------------------------------------------------
        // Operations
        //-----------------------------------------------------


        /// Input: this vector. Output: unchanged copy. Task: expression symmetry.
        [[nodiscard]]
        constexpr Vector4 operator+() const noexcept
        {
            return *this;
        }

        /// Input: this vector. Output: component-wise negation.
        /// Task: reverse a 4D vector direction/value.
        [[nodiscard]]
        constexpr Vector4 operator-() const noexcept
        {
            return { -x, -y, -z, -w };
        }

        /// Input: rhs. Output: this += rhs. Task: in-place addition.
        constexpr Vector4& operator+=(const Vector4& rhs) noexcept
        {
            x += rhs.x;
            y += rhs.y;
            z += rhs.z;
            w += rhs.w;
            return *this;
        }

        /// Input: rhs. Output: this -= rhs. Task: in-place subtraction.
        constexpr Vector4& operator-=(const Vector4& rhs) noexcept
        {
            x -= rhs.x;
            y -= rhs.y;
            z -= rhs.z;
            w -= rhs.w;
            return *this;
        }

        /// Input: scalar. Output: this *= scalar. Task: in-place uniform scale.
        constexpr Vector4& operator*=(T scalar) noexcept
        {
            x *= scalar;
            y *= scalar;
            z *= scalar;
            w *= scalar;
            return *this;
        }

        /// Input: non-zero scalar. Output: this /= scalar.
        /// Task: in-place uniform division; asserts on invalid divisor.
        constexpr Vector4& operator/=(T scalar) noexcept
        {
            assert(scalar != T(0));
            if constexpr (FloatingPoint<T>)
            {
                const T inverseScalar = T(1) / scalar;
                x *= inverseScalar;
                y *= inverseScalar;
                z *= inverseScalar;
                w *= inverseScalar;
            }
            else
            {
                x /= scalar;
                y /= scalar;
                z /= scalar;
                w /= scalar;
            }
            return *this;
        }

        /// Input: rhs. Output: component-wise product stored in this vector.
        /// Task: in-place Hadamard multiplication.
        constexpr Vector4& operator*=(const Vector4& rhs) noexcept
        {
            x *= rhs.x;
            y *= rhs.y;
            z *= rhs.z;
            w *= rhs.w;
            return *this;
        }

        /// Input: another vector. Output: exact component equality.
        /// Task: deterministic exact comparison.
        [[nodiscard]]
        constexpr bool operator==(const Vector4&) const noexcept = default;

        /// Input: this vector. Output: squared Euclidean length.
        /// Task: magnitude comparison without sqrt.
        [[nodiscard]]
        constexpr T LengthSquared() const noexcept
        {
            return (x * x) + (y * y) + (z * z) + (w * w);
        }

        /// Input: this floating-point vector. Output: Euclidean length.
        /// Task: compute magnitude.
        [[nodiscard]]
        T Length() const noexcept
            requires FloatingPoint<T>
        {
            return std::sqrt(LengthSquared());
        }

        /// Input: non-zero floating-point vector.
        /// Output: reciprocal length.
        /// Task: optimization hook for normalization-heavy internals.
        [[nodiscard]]
        T LengthInverse() const noexcept
            requires FloatingPoint<T>
        {
            const T length = Length();
            return length <= std::numeric_limits<T>::epsilon()
                ? T(0)
                : T(1) / length;
        }

        /// Input: this floating-point vector.
        /// Output: normalized copy, or zero for near-zero input.
        /// Task: safe value-returning normalization.
        [[nodiscard]]
        Vector4 Normalized() const noexcept
            requires FloatingPoint<T>
        {
            const T length = Length();
            if (length <= std::numeric_limits<T>::epsilon())
            {
                return Zero();
            }
            return *this / length;
        }

        /// Input: this floating-point vector.
        /// Output: none; mutates this vector when length is usable.
        /// Task: in-place normalization.
        void Normalize() noexcept
            requires FloatingPoint<T>
        {
            const T length = Length();
            if (length <= std::numeric_limits<T>::epsilon())
            {
                return;
            }
            *this /= length;
        }

        /// Input: this vector. Output: true when all components are exactly zero.
        /// Task: cheap exact zero check.
        [[nodiscard]]
        constexpr bool IsZero() const noexcept
        {
            return x == T(0) && y == T(0) && z == T(0) && w == T(0);
        }

        /// Input: this vector. Output: (x, y, z).
        /// Task: dimensional projection for position/direction extraction.
        [[nodiscard]]
        constexpr Vector3<T> XYZ() const noexcept
        {
            return { x, y, z };
        }
    };

    //=========================================================
    // Binary Operators
    //=========================================================

    /// Input: two vectors. Output: component-wise sum. Task: value addition.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector2<T> operator+(Vector2<T> lhs, const Vector2<T>& rhs) noexcept
    {
        return lhs += rhs;
    }

    /// Input: two vectors. Output: component-wise difference. Task: value subtraction.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector2<T> operator-(Vector2<T> lhs, const Vector2<T>& rhs) noexcept
    {
        return lhs -= rhs;
    }

    /// Input: vector and scalar. Output: uniformly scaled vector.
    /// Task: support vector * scalar expressions.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector2<T> operator*(Vector2<T> lhs, T scalar) noexcept
    {
        return lhs *= scalar;
    }

    /// Input: scalar and vector. Output: uniformly scaled vector.
    /// Task: support scalar * vector expressions.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector2<T> operator*(T scalar, Vector2<T> rhs) noexcept
    {
        return rhs *= scalar;
    }

    /// Input: vector and non-zero scalar. Output: uniformly divided vector.
    /// Task: support vector / scalar expressions.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector2<T> operator/(Vector2<T> lhs, T scalar) noexcept
    {
        return lhs /= scalar;
    }

    /// Input: two vectors. Output: component-wise product.
    /// Task: Hadamard multiplication for masks/scales.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector2<T> operator*(Vector2<T> lhs, const Vector2<T>& rhs) noexcept
    {
        return lhs *= rhs;
    }

    /// Input: two vectors. Output: component-wise sum. Task: value addition.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector3<T> operator+(Vector3<T> lhs, const Vector3<T>& rhs) noexcept
    {
        return lhs += rhs;
    }

    /// Input: two vectors. Output: component-wise difference. Task: value subtraction.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector3<T> operator-(Vector3<T> lhs, const Vector3<T>& rhs) noexcept
    {
        return lhs -= rhs;
    }

    /// Input: vector and scalar. Output: uniformly scaled vector.
    /// Task: support vector * scalar expressions.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector3<T> operator*(Vector3<T> lhs, T scalar) noexcept
    {
        return lhs *= scalar;
    }

    /// Input: scalar and vector. Output: uniformly scaled vector.
    /// Task: support scalar * vector expressions.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector3<T> operator*(T scalar, Vector3<T> rhs) noexcept
    {
        return rhs *= scalar;
    }

    /// Input: vector and non-zero scalar. Output: uniformly divided vector.
    /// Task: support vector / scalar expressions.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector3<T> operator/(Vector3<T> lhs, T scalar) noexcept
    {
        return lhs /= scalar;
    }

    /// Input: two vectors. Output: component-wise product.
    /// Task: Hadamard multiplication for masks/scales.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector3<T> operator*(Vector3<T> lhs, const Vector3<T>& rhs) noexcept
    {
        return lhs *= rhs;
    }

    /// Input: two vectors. Output: component-wise sum. Task: value addition.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector4<T> operator+(Vector4<T> lhs, const Vector4<T>& rhs) noexcept
    {
        return lhs += rhs;
    }

    /// Input: two vectors. Output: component-wise difference. Task: value subtraction.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector4<T> operator-(Vector4<T> lhs, const Vector4<T>& rhs) noexcept
    {
        return lhs -= rhs;
    }

    /// Input: vector and scalar. Output: uniformly scaled vector.
    /// Task: support vector * scalar expressions.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector4<T> operator*(Vector4<T> lhs, T scalar) noexcept
    {
        return lhs *= scalar;
    }

    /// Input: scalar and vector. Output: uniformly scaled vector.
    /// Task: support scalar * vector expressions.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector4<T> operator*(T scalar, Vector4<T> rhs) noexcept
    {
        return rhs *= scalar;
    }

    /// Input: vector and non-zero scalar. Output: uniformly divided vector.
    /// Task: support vector / scalar expressions.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector4<T> operator/(Vector4<T> lhs, T scalar) noexcept
    {
        return lhs /= scalar;
    }

    /// Input: two vectors. Output: component-wise product.
    /// Task: Hadamard multiplication for masks/scales.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector4<T> operator*(Vector4<T> lhs, const Vector4<T>& rhs) noexcept
    {
        return lhs *= rhs;
    }

    //=========================================================
    // Core Products
    //=========================================================

    /// Input: two 2D vectors.
    /// Output: scalar dot product.
    /// Task: measure projection/alignment. Positive means broadly same
    /// direction, negative means opposite, zero means perpendicular.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr T Dot(const Vector2<T>& lhs, const Vector2<T>& rhs) noexcept
    {
        return (lhs.x * rhs.x) + (lhs.y * rhs.y);
    }

    /// Input: two 3D vectors. Output: scalar dot product.
    /// Task: alignment/projection test used by lighting, physics, and culling.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr T Dot(const Vector3<T>& lhs, const Vector3<T>& rhs) noexcept
    {
        return (lhs.x * rhs.x) + (lhs.y * rhs.y) + (lhs.z * rhs.z);
    }

    /// Input: two 4D vectors. Output: scalar dot product.
    /// Task: alignment/projection test for homogeneous/vectorized math.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr T Dot(const Vector4<T>& lhs, const Vector4<T>& rhs) noexcept
    {
        return (lhs.x * rhs.x) + (lhs.y * rhs.y) + (lhs.z * rhs.z) + (lhs.w * rhs.w);
    }

    /// Input: two 3D vectors.
    /// Output: vector perpendicular to both inputs, using the right-hand rule.
    /// Task: construct normals, tangent bases, camera frames, and geometric
    /// orientation tests.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector3<T> Cross(const Vector3<T>& lhs, const Vector3<T>& rhs) noexcept
    {
        return
        {
            (lhs.y * rhs.z) - (lhs.z * rhs.y),
            (lhs.z * rhs.x) - (lhs.x * rhs.z),
            (lhs.x * rhs.y) - (lhs.y * rhs.x)
        };
    }

    /// Input: normal, incident direction, and reference direction.
    /// Output: normal or -normal so it faces against the incident/reference side.
    /// Task: orient normals consistently for coordinate frames.
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr Vector2<T> FaceForward(
        const Vector2<T>& normal,
        const Vector2<T>& incident,
        const Vector2<T>& reference) noexcept
    {
        return Dot(reference, incident) < T(0)
            ? normal
            : -normal;
    }

    /// Input: normal, incident direction, and reference direction.
    /// Output: normal or -normal so it faces against the incident/reference side.
    /// Task: orient normals consistently for PBR, ray tracing, normal mapping,
    /// and shading coordinate frames.
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr Vector3<T> FaceForward(
        const Vector3<T>& normal,
        const Vector3<T>& incident,
        const Vector3<T>& reference) noexcept
    {
        return Dot(reference, incident) < T(0)
            ? normal
            : -normal;
    }

    /// Input: normal, incident direction, and reference direction.
    /// Output: normal or -normal so it faces against the incident/reference side.
    /// Task: orient normals consistently for coordinate frames.
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr Vector4<T> FaceForward(
        const Vector4<T>& normal,
        const Vector4<T>& incident,
        const Vector4<T>& reference) noexcept
    {
        return Dot(reference, incident) < T(0)
            ? normal
            : -normal;
    }

    /// Input: a 2D vector.
    /// Output: a non-normalized vector perpendicular to the input.
    /// Task: produce a perpendicular vector.
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr Vector2<T> Orthogonal(const Vector2<T>& vector) noexcept
    {
        return { -vector.y, vector.x };
    }

    /// Input: a 3D vector.
    /// Output: a non-normalized vector perpendicular to the input.
    /// Task: produce a stable seed vector for camera basis generation, tangent
    /// frame construction, and orthonormal frames. The zero vector returns zero.
    template<FloatingPoint T>
    [[nodiscard]]
    Vector3<T> Orthogonal(const Vector3<T>& vector) noexcept
    {
        if (std::abs(vector.x) > std::abs(vector.z))
        {
            return { -vector.y, vector.x, T(0) };
        }

        return { T(0), -vector.z, vector.y };
    }

    /// Input: a 4D vector.
    /// Output: a non-normalized vector perpendicular to the input.
    /// Task: produce a perpendicular vector.
    template<FloatingPoint T>
    [[nodiscard]]
    Vector4<T> Orthogonal(const Vector4<T>& vector) noexcept
    {
        if (std::abs(vector.x) > std::abs(vector.w))
        {
            return { -vector.y, vector.x, T(0), T(0) };
        }

        return { T(0), T(0), -vector.w, vector.z };
    }

    //=========================================================
    // Scalar Helpers
    //=========================================================

    /// Input: two scalar values. Output: smaller value. Task: constexpr min.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr T Min(T lhs, T rhs) noexcept
    {
        return lhs < rhs ? lhs : rhs;
    }

    /// Input: two scalar values. Output: larger value. Task: constexpr max.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr T Max(T lhs, T rhs) noexcept
    {
        return lhs > rhs ? lhs : rhs;
    }

    /// Input: value and inclusive bounds.
    /// Output: value clamped into [minimum, maximum].
    /// Task: keep scalar values inside numeric ranges. The caller must pass
    /// minimum <= maximum.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr T Clamp(T value, T minimum, T maximum) noexcept
    {
        return value < minimum ? minimum :
            value > maximum ? maximum :
            value;
    }

    //=========================================================
    // Comparison Helpers
    //=========================================================

    /// Input: two floating-point scalars and a tolerance factor.
    /// Output: true when values are close under absolute or relative tolerance.
    /// Task: tolerate floating-point rounding noise across small and large
    /// magnitudes. The absolute branch handles values near zero; the relative
    /// branch prevents large-scale comparisons from being dominated by a tiny
    /// fixed epsilon.
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr bool NearlyEqual(
        T lhs,
        T rhs,
        T epsilon = std::numeric_limits<T>::epsilon() * T(10)) noexcept
    {
        const T difference =
            std::abs(lhs - rhs);

        if (difference <= epsilon)
        {
            return true;
        }

        T scale =
            std::abs(lhs) > std::abs(rhs)
                ? std::abs(lhs)
                : std::abs(rhs);

        if (scale < T(1))
        {
            scale = T(1);
        }

        return difference <= epsilon * scale;
    }

    /// Input: two vectors and a scalar tolerance factor.
    /// Output: true when every component is nearly equal.
    /// Task: vector floating-point comparison for tests and geometric checks,
    /// using the scalar absolute/relative tolerance policy per component.
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr bool NearlyEqual(
        const Vector2<T>& lhs,
        const Vector2<T>& rhs,
        T epsilon = std::numeric_limits<T>::epsilon() * T(10)) noexcept
    {
        return
            NearlyEqual(lhs.x, rhs.x, epsilon) &&
            NearlyEqual(lhs.y, rhs.y, epsilon);
    }

    /// Input: two vectors and a scalar tolerance factor.
    /// Output: true when every component is nearly equal.
    /// Task: vector floating-point comparison for tests and geometric checks,
    /// using the scalar absolute/relative tolerance policy per component.
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr bool NearlyEqual(
        const Vector3<T>& lhs,
        const Vector3<T>& rhs,
        T epsilon = std::numeric_limits<T>::epsilon() * T(10)) noexcept
    {
        return
            NearlyEqual(lhs.x, rhs.x, epsilon) &&
            NearlyEqual(lhs.y, rhs.y, epsilon) &&
            NearlyEqual(lhs.z, rhs.z, epsilon);
    }

    /// Input: two vectors and a scalar tolerance factor.
    /// Output: true when every component is nearly equal.
    /// Task: vector floating-point comparison for tests and geometric checks,
    /// using the scalar absolute/relative tolerance policy per component.
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr bool NearlyEqual(
        const Vector4<T>& lhs,
        const Vector4<T>& rhs,
        T epsilon = std::numeric_limits<T>::epsilon() * T(10)) noexcept
    {
        return
            NearlyEqual(lhs.x, rhs.x, epsilon) &&
            NearlyEqual(lhs.y, rhs.y, epsilon) &&
            NearlyEqual(lhs.z, rhs.z, epsilon) &&
            NearlyEqual(lhs.w, rhs.w, epsilon);
    }

    //=========================================================
    // Component-Wise Operations
    //=========================================================

    /// Input: two vectors. Output: component-wise product.
    /// Task: explicit name for Hadamard multiplication used by shading/color math.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector2<T> Hadamard(const Vector2<T>& lhs, const Vector2<T>& rhs) noexcept
    {
        return lhs * rhs;
    }

    /// Input: two vectors. Output: component-wise product.
    /// Task: explicit name for Hadamard multiplication used by shading/color math.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector3<T> Hadamard(const Vector3<T>& lhs, const Vector3<T>& rhs) noexcept
    {
        return lhs * rhs;
    }

    /// Input: two vectors. Output: component-wise product.
    /// Task: explicit name for Hadamard multiplication used by shading/color math.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector4<T> Hadamard(const Vector4<T>& lhs, const Vector4<T>& rhs) noexcept
    {
        return lhs * rhs;
    }

    /// Input: vector. Output: component-wise absolute value.
    /// Task: remove signs for bounds, distances, and error metrics.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector2<T> Abs(const Vector2<T>& value) noexcept
    {
        return
        {
            value.x < T(0) ? -value.x : value.x,
            value.y < T(0) ? -value.y : value.y
        };
    }

    /// Input: vector. Output: component-wise absolute value.
    /// Task: remove signs for bounds, distances, and error metrics.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector3<T> Abs(const Vector3<T>& value) noexcept
    {
        return
        {
            value.x < T(0) ? -value.x : value.x,
            value.y < T(0) ? -value.y : value.y,
            value.z < T(0) ? -value.z : value.z
        };
    }

    /// Input: vector. Output: component-wise absolute value.
    /// Task: remove signs for bounds, distances, and error metrics.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector4<T> Abs(const Vector4<T>& value) noexcept
    {
        return
        {
            value.x < T(0) ? -value.x : value.x,
            value.y < T(0) ? -value.y : value.y,
            value.z < T(0) ? -value.z : value.z,
            value.w < T(0) ? -value.w : value.w
        };
    }

    /// Input: vector. Output: component-wise sign values (-1, 0, +1).
    /// Task: compact sign extraction for branch-light math.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector2<T> Sign(const Vector2<T>& value) noexcept
    {
        return
        {
            static_cast<T>((value.x > T(0)) - (value.x < T(0))),
            static_cast<T>((value.y > T(0)) - (value.y < T(0)))
        };
    }

    /// Input: vector. Output: component-wise sign values (-1, 0, +1).
    /// Task: compact sign extraction for branch-light math.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector3<T> Sign(const Vector3<T>& value) noexcept
    {
        return
        {
            static_cast<T>((value.x > T(0)) - (value.x < T(0))),
            static_cast<T>((value.y > T(0)) - (value.y < T(0))),
            static_cast<T>((value.z > T(0)) - (value.z < T(0)))
        };
    }

    /// Input: vector. Output: component-wise sign values (-1, 0, +1).
    /// Task: compact sign extraction for branch-light math.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector4<T> Sign(const Vector4<T>& value) noexcept
    {
        return
        {
            static_cast<T>((value.x > T(0)) - (value.x < T(0))),
            static_cast<T>((value.y > T(0)) - (value.y < T(0))),
            static_cast<T>((value.z > T(0)) - (value.z < T(0))),
            static_cast<T>((value.w > T(0)) - (value.w < T(0)))
        };
    }

    /// Input: two vectors. Output: component-wise minimum.
    /// Task: construct bounds and clamp ranges without branches at call sites.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector2<T> Min(const Vector2<T>& lhs, const Vector2<T>& rhs) noexcept
    {
        return { Min(lhs.x, rhs.x), Min(lhs.y, rhs.y) };
    }

    /// Input: two vectors. Output: component-wise minimum.
    /// Task: construct bounds and clamp ranges without branches at call sites.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector3<T> Min(const Vector3<T>& lhs, const Vector3<T>& rhs) noexcept
    {
        return { Min(lhs.x, rhs.x), Min(lhs.y, rhs.y), Min(lhs.z, rhs.z) };
    }

    /// Input: two vectors. Output: component-wise minimum.
    /// Task: construct bounds and clamp ranges without branches at call sites.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector4<T> Min(const Vector4<T>& lhs, const Vector4<T>& rhs) noexcept
    {
        return { Min(lhs.x, rhs.x), Min(lhs.y, rhs.y), Min(lhs.z, rhs.z), Min(lhs.w, rhs.w) };
    }

    /// Input: two vectors. Output: component-wise maximum.
    /// Task: construct bounds and clamp ranges without branches at call sites.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector2<T> Max(const Vector2<T>& lhs, const Vector2<T>& rhs) noexcept
    {
        return { Max(lhs.x, rhs.x), Max(lhs.y, rhs.y) };
    }

    /// Input: two vectors. Output: component-wise maximum.
    /// Task: construct bounds and clamp ranges without branches at call sites.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector3<T> Max(const Vector3<T>& lhs, const Vector3<T>& rhs) noexcept
    {
        return { Max(lhs.x, rhs.x), Max(lhs.y, rhs.y), Max(lhs.z, rhs.z) };
    }

    /// Input: two vectors. Output: component-wise maximum.
    /// Task: construct bounds and clamp ranges without branches at call sites.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector4<T> Max(const Vector4<T>& lhs, const Vector4<T>& rhs) noexcept
    {
        return { Max(lhs.x, rhs.x), Max(lhs.y, rhs.y), Max(lhs.z, rhs.z), Max(lhs.w, rhs.w) };
    }

    /// Input: vector and inclusive vector bounds.
    /// Output: component-wise clamped vector.
    /// Task: constrain each component independently.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector2<T> Clamp(
        const Vector2<T>& value,
        const Vector2<T>& minValue,
        const Vector2<T>& maxValue) noexcept
    {
        return Min(Max(value, minValue), maxValue);
    }

    /// Input: vector and inclusive vector bounds.
    /// Output: component-wise clamped vector.
    /// Task: constrain each component independently.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector3<T> Clamp(
        const Vector3<T>& value,
        const Vector3<T>& minValue,
        const Vector3<T>& maxValue) noexcept
    {
        return Min(Max(value, minValue), maxValue);
    }

    /// Input: vector and inclusive vector bounds.
    /// Output: component-wise clamped vector.
    /// Task: constrain each component independently.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector4<T> Clamp(
        const Vector4<T>& value,
        const Vector4<T>& minValue,
        const Vector4<T>& maxValue) noexcept
    {
        return Min(Max(value, minValue), maxValue);
    }

    //=========================================================
    // Geometric Helpers
    //=========================================================

    /// Input: two floating-point vectors.
    /// Output: angle between them in radians; zero for near-zero input.
    /// Task: robustly measure angular separation. The cosine is clamped to
    /// avoid acos domain errors from floating-point drift.
    template<FloatingPoint T>
    [[nodiscard]]
    T AngleBetween(const Vector2<T>& lhs, const Vector2<T>& rhs) noexcept
    {
        const T denominator = lhs.Length() * rhs.Length();
        if (denominator <= std::numeric_limits<T>::epsilon())
        {
            return T(0);
        }

        return std::acos(Clamp(Dot(lhs, rhs) / denominator, T(-1), T(1)));
    }

    /// Input: two floating-point vectors.
    /// Output: angle between them in radians; zero for near-zero input.
    /// Task: robust angular separation for camera, physics, and shading code.
    template<FloatingPoint T>
    [[nodiscard]]
    T AngleBetween(const Vector3<T>& lhs, const Vector3<T>& rhs) noexcept
    {
        const T denominator = lhs.Length() * rhs.Length();
        if (denominator <= std::numeric_limits<T>::epsilon())
        {
            return T(0);
        }

        return std::acos(Clamp(Dot(lhs, rhs) / denominator, T(-1), T(1)));
    }

    /// Input: two floating-point vectors.
    /// Output: angle between them in radians; zero for near-zero input.
    /// Task: robust angular separation for homogeneous/vectorized math.
    template<FloatingPoint T>
    [[nodiscard]]
    T AngleBetween(const Vector4<T>& lhs, const Vector4<T>& rhs) noexcept
    {
        const T denominator = lhs.Length() * rhs.Length();
        if (denominator <= std::numeric_limits<T>::epsilon())
        {
            return T(0);
        }

        return std::acos(Clamp(Dot(lhs, rhs) / denominator, T(-1), T(1)));
    }

    /// Input: vector and fallback.
    /// Output: normalized vector, or fallback if input is near zero.
    /// Task: normalize safely when zero has semantic meaning chosen by caller.
    template<FloatingPoint T>
    [[nodiscard]]
    Vector2<T> SafeNormalize(
        const Vector2<T>& vector,
        const Vector2<T>& fallback = Vector2<T>::Zero()) noexcept
    {
        const T length = vector.Length();
        if (length <= std::numeric_limits<T>::epsilon())
        {
            return fallback;
        }
        return vector / length;
    }

    /// Input: vector and fallback.
    /// Output: normalized vector, or fallback if input is near zero.
    /// Task: normalize safely for normals, directions, and camera vectors.
    template<FloatingPoint T>
    [[nodiscard]]
    Vector3<T> SafeNormalize(
        const Vector3<T>& vector,
        const Vector3<T>& fallback = Vector3<T>::Zero()) noexcept
    {
        const T length = vector.Length();
        if (length <= std::numeric_limits<T>::epsilon())
        {
            return fallback;
        }
        return vector / length;
    }

    /// Input: vector and fallback.
    /// Output: normalized vector, or fallback if input is near zero.
    /// Task: normalize safely for 4D values.
    template<FloatingPoint T>
    [[nodiscard]]
    Vector4<T> SafeNormalize(
        const Vector4<T>& vector,
        const Vector4<T>& fallback = Vector4<T>::Zero()) noexcept
    {
        const T length = vector.Length();
        if (length <= std::numeric_limits<T>::epsilon())
        {
            return fallback;
        }
        return vector / length;
    }

    /// Input: incident direction, surface normal, and etaRatio = n1 / n2.
    /// Output: refracted direction.
    /// Task: compute Snell refraction for shading and ray tracing.
    /// Important: incident and normal must already be normalized. If they are
    /// not, the result is physically wrong and may hide bugs in PBR code.
    template<FloatingPoint T>
    [[nodiscard]]
    Vector2<T> Refract(
        const Vector2<T>& incident,
        const Vector2<T>& normal,
        T etaRatio) noexcept
    {
        const T cosTheta = Min(Dot(-incident, normal), T(1));
        const Vector2<T> perpendicular = etaRatio * (incident + (cosTheta * normal));
        const T parallelLengthSquared = T(1) - perpendicular.LengthSquared();
        if (parallelLengthSquared < T(0))
        {
            return Vector2<T>::Zero();
        }

        const Vector2<T> parallel = -std::sqrt(parallelLengthSquared) * normal;
        return perpendicular + parallel;
    }

    /// Input: incident direction, surface normal, and etaRatio = n1 / n2.
    /// Output: refracted direction.
    /// Task: compute Snell refraction for PBR and CPU ray tracing.
    /// Important: incident and normal must already be normalized.
    template<FloatingPoint T>
    [[nodiscard]]
    Vector3<T> Refract(
        const Vector3<T>& incident,
        const Vector3<T>& normal,
        T etaRatio) noexcept
    {
        const T cosTheta = Min(Dot(-incident, normal), T(1));
        const Vector3<T> perpendicular = etaRatio * (incident + (cosTheta * normal));
        const T parallelLengthSquared = T(1) - perpendicular.LengthSquared();
        if (parallelLengthSquared < T(0))
        {
            return Vector3<T>::Zero();
        }

        const Vector3<T> parallel = -std::sqrt(parallelLengthSquared) * normal;
        return perpendicular + parallel;
    }

    /// Input: incident direction, surface normal, and etaRatio = n1 / n2.
    /// Output: refracted direction.
    /// Task: compute Snell refraction for 4D/vectorized math.
    /// Important: incident and normal must already be normalized.
    template<FloatingPoint T>
    [[nodiscard]]
    Vector4<T> Refract(
        const Vector4<T>& incident,
        const Vector4<T>& normal,
        T etaRatio) noexcept
    {
        const T cosTheta = Min(Dot(-incident, normal), T(1));
        const Vector4<T> perpendicular = etaRatio * (incident + (cosTheta * normal));
        const T parallelLengthSquared = T(1) - perpendicular.LengthSquared();
        if (parallelLengthSquared < T(0))
        {
            return Vector4<T>::Zero();
        }

        const Vector4<T> parallel = -std::sqrt(parallelLengthSquared) * normal;
        return perpendicular + parallel;
    }

    /// Input: vector and normalized surface normal.
    /// Output: reflected vector.
    /// Task: mirror a direction around a normal for lighting and ray bounces.
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr Vector2<T> Reflect(const Vector2<T>& vector, const Vector2<T>& normal) noexcept
    {
        return vector - ((T(2) * Dot(vector, normal)) * normal);
    }

    /// Input: vector and normalized surface normal.
    /// Output: reflected vector.
    /// Task: mirror a direction around a normal for lighting and ray bounces.
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr Vector3<T> Reflect(const Vector3<T>& vector, const Vector3<T>& normal) noexcept
    {
        return vector - ((T(2) * Dot(vector, normal)) * normal);
    }

    /// Input: vector and normalized surface normal.
    /// Output: reflected vector.
    /// Task: mirror a direction around a normal for 4D/vectorized math.
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr Vector4<T> Reflect(const Vector4<T>& vector, const Vector4<T>& normal) noexcept
    {
        return vector - ((T(2) * Dot(vector, normal)) * normal);
    }

    /// Input: vector and destination axis.
    /// Output: projection of vector onto that axis.
    /// Task: extract the component of a vector along a direction. The axis does
    /// not need to be normalized; zero-length axes return zero.
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr Vector2<T> Project(const Vector2<T>& vector, const Vector2<T>& onto) noexcept
    {
        const T denominator = Dot(onto, onto);
        if (denominator <= std::numeric_limits<T>::epsilon())
        {
            return Vector2<T>::Zero();
        }

        return onto * (Dot(vector, onto) / denominator);
    }

    /// Input: vector and destination axis.
    /// Output: projection of vector onto that axis.
    /// Task: extract the component of a vector along a direction. The axis does
    /// not need to be normalized; zero-length axes return zero.
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr Vector3<T> Project(const Vector3<T>& vector, const Vector3<T>& onto) noexcept
    {
        const T denominator = Dot(onto, onto);
        if (denominator <= std::numeric_limits<T>::epsilon())
        {
            return Vector3<T>::Zero();
        }

        return onto * (Dot(vector, onto) / denominator);
    }

    /// Input: vector and destination axis.
    /// Output: projection of vector onto that axis.
    /// Task: extract the component of a vector along a direction. The axis does
    /// not need to be normalized; zero-length axes return zero.
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr Vector4<T> Project(const Vector4<T>& vector, const Vector4<T>& onto) noexcept
    {
        const T denominator = Dot(onto, onto);
        if (denominator <= std::numeric_limits<T>::epsilon())
        {
            return Vector4<T>::Zero();
        }

        return onto * (Dot(vector, onto) / denominator);
    }

    /// Input: vector and axis to reject from.
    /// Output: component of vector perpendicular to onto.
    /// Task: subtract the projected component from the original vector.
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr Vector2<T> Reject(const Vector2<T>& vector, const Vector2<T>& onto) noexcept
    {
        return vector - Project(vector, onto);
    }

    /// Input: vector and axis to reject from.
    /// Output: component of vector perpendicular to onto.
    /// Task: subtract the projected component from the original vector.
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr Vector3<T> Reject(const Vector3<T>& vector, const Vector3<T>& onto) noexcept
    {
        return vector - Project(vector, onto);
    }

    /// Input: vector and axis to reject from.
    /// Output: component of vector perpendicular to onto.
    /// Task: subtract the projected component from the original vector.
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr Vector4<T> Reject(const Vector4<T>& vector, const Vector4<T>& onto) noexcept
    {
        return vector - Project(vector, onto);
    }


    /// Input: two points. Output: squared distance between them.
    /// Task: compare distances without sqrt.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr T DistanceSquared(const Vector2<T>& lhs, const Vector2<T>& rhs) noexcept
    {
        return (rhs - lhs).LengthSquared();
    }

    /// Input: two points. Output: squared distance between them.
    /// Task: compare distances without sqrt.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr T DistanceSquared(const Vector3<T>& lhs, const Vector3<T>& rhs) noexcept
    {
        return (rhs - lhs).LengthSquared();
    }

    /// Input: two points. Output: squared distance between them.
    /// Task: compare distances without sqrt.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr T DistanceSquared(const Vector4<T>& lhs, const Vector4<T>& rhs) noexcept
    {
        return (rhs - lhs).LengthSquared();
    }

    /// Input: two points. Output: Euclidean distance.
    /// Task: compute true distance when sqrt cost is acceptable.
    template<FloatingPoint T>
    [[nodiscard]]
    T Distance(const Vector2<T>& lhs, const Vector2<T>& rhs) noexcept
    {
        return (rhs - lhs).Length();
    }

    /// Input: two points. Output: Euclidean distance.
    /// Task: compute true distance when sqrt cost is acceptable.
    template<FloatingPoint T>
    [[nodiscard]]
    T Distance(const Vector3<T>& lhs, const Vector3<T>& rhs) noexcept
    {
        return (rhs - lhs).Length();
    }

    /// Input: two points. Output: Euclidean distance.
    /// Task: compute true distance when sqrt cost is acceptable.
    template<FloatingPoint T>
    [[nodiscard]]
    T Distance(const Vector4<T>& lhs, const Vector4<T>& rhs) noexcept
    {
        return (rhs - lhs).Length();
    }

    /// Input: floating-point vector.
    /// Output: normalized copy, or zero for near-zero input.
    /// Task: free-function equivalent of v.Normalized() for generic code.
    template<FloatingPoint T>
    [[nodiscard]]
    Vector2<T> Normalize(const Vector2<T>& vector) noexcept
    {
        return vector.Normalized();
    }

    /// Input: floating-point vector.
    /// Output: normalized copy, or zero for near-zero input.
    /// Task: free-function equivalent of v.Normalized() for generic code.
    template<FloatingPoint T>
    [[nodiscard]]
    Vector3<T> Normalize(const Vector3<T>& vector) noexcept
    {
        return vector.Normalized();
    }

    /// Input: floating-point vector.
    /// Output: normalized copy, or zero for near-zero input.
    /// Task: free-function equivalent of v.Normalized() for generic code.
    template<FloatingPoint T>
    [[nodiscard]]
    Vector4<T> Normalize(const Vector4<T>& vector) noexcept
    {
        return vector.Normalized();
    }

    /// Input: endpoints a and b, and interpolation value alpha.
    /// Output: a + (b - a) * alpha.
    /// Task: linear interpolation. Higher-order interpolation belongs in a
    /// future Foundation.Math.Interpolation module, not in Vector.
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr Vector2<T> Lerp(const Vector2<T>& a, const Vector2<T>& b, T alpha) noexcept
    {
        return a + ((b - a) * alpha);
    }

    /// Input: endpoints a and b, and interpolation value alpha.
    /// Output: a + (b - a) * alpha.
    /// Task: linear interpolation for positions, colors, and parameters.
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr Vector3<T> Lerp(const Vector3<T>& a, const Vector3<T>& b, T alpha) noexcept
    {
        return a + ((b - a) * alpha);
    }

    /// Input: endpoints a and b, and interpolation value alpha.
    /// Output: a + (b - a) * alpha.
    /// Task: linear interpolation for 4D values.
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr Vector4<T> Lerp(const Vector4<T>& a, const Vector4<T>& b, T alpha) noexcept
    {
        return a + ((b - a) * alpha);
    }

    //=========================================================
    // Aliases
    //=========================================================

    using Vec2f = Vector2<float>;
    using Vec2d = Vector2<double>;
    using Vec2i = Vector2<std::int32_t>;
    using Vec2u = Vector2<std::uint32_t>;

    using Vec3f = Vector3<float>;
    using Vec3d = Vector3<double>;
    using Vec3i = Vector3<std::int32_t>;
    using Vec3u = Vector3<std::uint32_t>;

    using Vec4f = Vector4<float>;
    using Vec4d = Vector4<double>;
    using Vec4i = Vector4<std::int32_t>;
    using Vec4u = Vector4<std::uint32_t>;

    //=========================================================
    // Common Constants
    //=========================================================

    namespace constants
    {
        /// Named constants mirror the static constructors for convenient use in
        /// non-template code and debugger watch expressions.
        inline constexpr Vec2f Zero2f = Vec2f::Zero();
        inline constexpr Vec3f Zero3f = Vec3f::Zero();
        inline constexpr Vec4f Zero4f = Vec4f::Zero();

        inline constexpr Vec2f One2f = Vec2f::One();
        inline constexpr Vec3f One3f = Vec3f::One();
        inline constexpr Vec4f One4f = Vec4f::One();

        inline constexpr Vec2f UnitX2f = Vec2f::UnitX();
        inline constexpr Vec2f UnitY2f = Vec2f::UnitY();

        inline constexpr Vec3f UnitX3f = Vec3f::UnitX();
        inline constexpr Vec3f UnitY3f = Vec3f::UnitY();
        inline constexpr Vec3f UnitZ3f = Vec3f::UnitZ();

        inline constexpr Vec4f UnitX4f = Vec4f::UnitX();
        inline constexpr Vec4f UnitY4f = Vec4f::UnitY();
        inline constexpr Vec4f UnitZ4f = Vec4f::UnitZ();
        inline constexpr Vec4f UnitW4f = Vec4f::UnitW();

        /// Right-handed engine convention:
        /// +X = right, +Y = up, -Z = forward.
        /// Keeping this in constants documents the convention without forcing
        /// renderer-specific coordinate assumptions into the vector type.
        inline constexpr Vec3f Right3f = Vec3f::Right();
        inline constexpr Vec3f Left3f = Vec3f::Left();
        inline constexpr Vec3f Up3f = Vec3f::Up();
        inline constexpr Vec3f Down3f = Vec3f::Down();
        inline constexpr Vec3f Forward3f = Vec3f::Forward();
        inline constexpr Vec3f Backward3f = Vec3f::Backward();
    }

    //=========================================================
    // Compile-Time Validation
    //=========================================================

    static_assert(Vector2<float>::Size == 2);
    static_assert(Vector3<float>::Size == 3);
    static_assert(Vector4<float>::Size == 4);

    static_assert(std::is_trivially_copyable_v<Vec2f>);
    static_assert(std::is_trivially_copyable_v<Vec3f>);
    static_assert(std::is_trivially_copyable_v<Vec4f>);

    static_assert(std::is_standard_layout_v<Vec2f>);
    static_assert(std::is_standard_layout_v<Vec3f>);
    static_assert(std::is_standard_layout_v<Vec4f>);

    static_assert(sizeof(Vec2f) == sizeof(float) * 2);
    static_assert(sizeof(Vec3f) == sizeof(float) * 3);
    static_assert(sizeof(Vec4f) == sizeof(float) * 4);

    static_assert(alignof(Vec4f) == alignof(float));
    static_assert(alignof(Vec4d) == alignof(double));

} // namespace kairo::foundation::math
