module;

#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

export module Foundation.Math.Matrix;

import Foundation.Math.Vector;


export namespace foundation::math
{
    // Matrix convention for the whole module:
    // - Row-major memory layout.
    // - Column-vector multiplication: result = matrix * vector.
    // - Transform matrices store translation in the final column.
    // - Composition order follows column-vector math: A * B * v applies B first.
    // - Projection helpers use a Vulkan/Metal-friendly depth range of [0, 1].

    //=========================================================
    // Forward Declarations
    //=========================================================

    template<Arithmetic T>
    struct Matrix3;

    template<Arithmetic T>
    struct Matrix4;

    //=========================================================
    // Matrix3
    //=========================================================

    /// A 3x3 row-major matrix.
    ///
    /// Design notes:
    /// - Storage is contiguous and allocation-free for cache-friendly CPU math,
    ///   serialization, and graphics API upload paths.
    /// - The matrix is row-major in memory and uses column vectors for
    ///   multiplication: result = matrix * vector.
    /// - The type is intentionally generic over arithmetic scalars, while
    ///   geometric operations that require division/sqrt are restricted to
    ///   floating-point types.
    template<Arithmetic T>
    struct Matrix3 final
    {
        using ValueType = T;

        static constexpr std::size_t Rows = 3;
        static constexpr std::size_t Columns = 3;
        static constexpr std::size_t ElementCount = Rows * Columns;

        std::array<T, ElementCount> data {};

        /// Input: none.
        /// Output: all-zero matrix.
        /// Task: provide a named zero value for initialization and clearing.
        [[nodiscard]]
        static constexpr Matrix3 Zero() noexcept
        {
            return {};
        }

        /// Input: none.
        /// Output: all-one matrix.
        /// Task: useful for tests, masks, and component-wise initialization. It
        /// is not an identity matrix.
        [[nodiscard]]
        static constexpr Matrix3 One() noexcept
        {
            Matrix3 result;
            for (auto& element : result.data)
            {
                element = T(1);
            }
            return result;
        }

        /// Input: none.
        /// Output: identity matrix.
        /// Task: neutral element for matrix multiplication and transforms.
        [[nodiscard]]
        static constexpr Matrix3 Identity() noexcept
        {
            Matrix3 result;
            result(0, 0) = T(1);
            result(1, 1) = T(1);
            result(2, 2) = T(1);
            return result;
        }

        /// Input: none.
        /// Output: zero-initialized matrix.
        /// Task: deterministic default construction.
        constexpr Matrix3() noexcept = default;

        /// Input: diagonal scalar.
        /// Output: matrix with diagonal set to scalar and other elements zero.
        /// Task: concise construction for identity-like matrices.
        constexpr explicit Matrix3(T diagonal) noexcept
        {
            (*this)(0, 0) = diagonal;
            (*this)(1, 1) = diagonal;
            (*this)(2, 2) = diagonal;
        }

        /// Input: nine row-major values.
        /// Output: matrix containing those values.
        /// Task: explicit construction without hidden transposition.
        constexpr Matrix3(
            T m00, T m01, T m02,
            T m10, T m11, T m12,
            T m20, T m21, T m22) noexcept
            : data
            {
                m00, m01, m02,
                m10, m11, m12,
                m20, m21, m22
            }
        {
        }

        /// Input: row and column in valid ranges.
        /// Output: mutable element reference.
        /// Task: 2D element access for readable matrix math. The asserts catch
        /// programmer errors; out-of-range indexing is not recoverable runtime
        /// input for this low-level type.
        [[nodiscard]]
        constexpr T& operator()(std::size_t row, std::size_t column) noexcept
        {
            assert(row < Rows);
            assert(column < Columns);
            return data[(row * Columns) + column];
        }

        /// Input: row and column in valid ranges.
        /// Output: const element reference.
        /// Task: read-only 2D element access.
        [[nodiscard]]
        constexpr const T& operator()(std::size_t row, std::size_t column) const noexcept
        {
            assert(row < Rows);
            assert(column < Columns);
            return data[(row * Columns) + column];
        }

        /// Input: linear index in [0, ElementCount).
        /// Output: mutable element reference.
        /// Task: low-level iteration for algorithms, serialization, and tests.
        [[nodiscard]]
        constexpr T& operator[](std::size_t index) noexcept
        {
            assert(index < ElementCount);
            return data[index];
        }

        /// Input: linear index in [0, ElementCount).
        /// Output: const element reference.
        /// Task: read-only low-level iteration.
        [[nodiscard]]
        constexpr const T& operator[](std::size_t index) const noexcept
        {
            assert(index < ElementCount);
            return data[index];
        }

        /// Input: none.
        /// Output: pointer to the first row-major element.
        /// Task: expose contiguous memory for Vulkan/Metal/OpenGL uploads,
        /// binary serialization, and span construction.
        [[nodiscard]]
        constexpr T* Data() noexcept
        {
            return data.data();
        }

        /// Input: none.
        /// Output: const pointer to the first row-major element.
        /// Task: read-only upload/serialization access.
        [[nodiscard]]
        constexpr const T* Data() const noexcept
        {
            return data.data();
        }

        /// Input: row index.
        /// Output: selected row as a Vector3.
        /// Task: support dot-product multiplication and row-basis inspection.
        [[nodiscard]]
        constexpr Vector3<T> Row(std::size_t row) const noexcept
        {
            assert(row < Rows);
            return { (*this)(row, 0), (*this)(row, 1), (*this)(row, 2) };
        }

        /// Input: column index.
        /// Output: selected column as a Vector3.
        /// Task: extract basis vectors and scale magnitudes from transforms.
        [[nodiscard]]
        constexpr Vector3<T> Column(std::size_t column) const noexcept
        {
            assert(column < Columns);
            return { (*this)(0, column), (*this)(1, column), (*this)(2, column) };
        }

        /// Input: row index and replacement row.
        /// Output: none; mutates this matrix.
        /// Task: build camera, basis, and decomposition matrices cleanly.
        constexpr void SetRow(std::size_t row, const Vector3<T>& value) noexcept
        {
            assert(row < Rows);
            (*this)(row, 0) = value.x;
            (*this)(row, 1) = value.y;
            (*this)(row, 2) = value.z;
        }

        /// Input: column index and replacement column.
        /// Output: none; mutates this matrix.
        /// Task: install basis vectors during transform construction.
        constexpr void SetColumn(std::size_t column, const Vector3<T>& value) noexcept
        {
            assert(column < Columns);
            (*this)(0, column) = value.x;
            (*this)(1, column) = value.y;
            (*this)(2, column) = value.z;
        }

        /// Input: another matrix.
        /// Output: exact element-wise equality.
        /// Task: deterministic comparisons for integer and exact matrices. Use
        /// NearlyEqual() for floating-point tolerance checks.
        [[nodiscard]]
        constexpr bool operator==(const Matrix3&) const noexcept = default;
    };

    //=========================================================
    // Matrix4
    //=========================================================

    /// A 4x4 row-major matrix for transforms, camera/view/projection matrices,
    /// homogeneous coordinates, and future renderer/physics integration.
    template<Arithmetic T>
    struct Matrix4 final
    {
        using ValueType = T;

        static constexpr std::size_t Rows = 4;
        static constexpr std::size_t Columns = 4;
        static constexpr std::size_t ElementCount = Rows * Columns;

        std::array<T, ElementCount> data {};

        /// Input: none. Output: all-zero matrix. Task: named zero value.
        [[nodiscard]]
        static constexpr Matrix4 Zero() noexcept
        {
            return {};
        }

        /// Input: none.
        /// Output: all-one matrix.
        /// Task: useful for tests and component-wise initialization; not identity.
        [[nodiscard]]
        static constexpr Matrix4 One() noexcept
        {
            Matrix4 result;
            for (auto& element : result.data)
            {
                element = T(1);
            }
            return result;
        }

        /// Input: none.
        /// Output: identity matrix.
        /// Task: neutral matrix for transforms and multiplication.
        [[nodiscard]]
        static constexpr Matrix4 Identity() noexcept
        {
            Matrix4 result;
            result(0, 0) = T(1);
            result(1, 1) = T(1);
            result(2, 2) = T(1);
            result(3, 3) = T(1);
            return result;
        }

        /// Input: none. Output: zero-initialized matrix. Task: deterministic construction.
        constexpr Matrix4() noexcept = default;

        /// Input: diagonal scalar.
        /// Output: matrix with diagonal set to scalar and other elements zero.
        /// Task: concise identity-like construction.
        constexpr explicit Matrix4(T diagonal) noexcept
        {
            (*this)(0, 0) = diagonal;
            (*this)(1, 1) = diagonal;
            (*this)(2, 2) = diagonal;
            (*this)(3, 3) = diagonal;
        }

        /// Input: sixteen row-major values.
        /// Output: matrix containing those values.
        /// Task: explicit construction with no hidden storage convention changes.
        constexpr Matrix4(
            T m00, T m01, T m02, T m03,
            T m10, T m11, T m12, T m13,
            T m20, T m21, T m22, T m23,
            T m30, T m31, T m32, T m33) noexcept
            : data
            {
                m00, m01, m02, m03,
                m10, m11, m12, m13,
                m20, m21, m22, m23,
                m30, m31, m32, m33
            }
        {
        }

        /// Input: row and column in valid ranges. Output: mutable element reference.
        /// Task: 2D element access for readable matrix math.
        [[nodiscard]]
        constexpr T& operator()(std::size_t row, std::size_t column) noexcept
        {
            assert(row < Rows);
            assert(column < Columns);
            return data[(row * Columns) + column];
        }

        /// Input: row and column in valid ranges. Output: const element reference.
        /// Task: read-only 2D element access.
        [[nodiscard]]
        constexpr const T& operator()(std::size_t row, std::size_t column) const noexcept
        {
            assert(row < Rows);
            assert(column < Columns);
            return data[(row * Columns) + column];
        }

        /// Input: linear index in [0, ElementCount).
        /// Output: mutable element reference.
        /// Task: low-level iteration over row-major storage.
        [[nodiscard]]
        constexpr T& operator[](std::size_t index) noexcept
        {
            assert(index < ElementCount);
            return data[index];
        }

        /// Input: linear index in [0, ElementCount).
        /// Output: const element reference.
        /// Task: read-only low-level iteration over row-major storage.
        [[nodiscard]]
        constexpr const T& operator[](std::size_t index) const noexcept
        {
            assert(index < ElementCount);
            return data[index];
        }

        /// Input: none.
        /// Output: pointer to first row-major element.
        /// Task: direct access for renderer upload and serialization.
        [[nodiscard]]
        constexpr T* Data() noexcept
        {
            return data.data();
        }

        /// Input: none.
        /// Output: const pointer to first row-major element.
        /// Task: read-only renderer upload and serialization access.
        [[nodiscard]]
        constexpr const T* Data() const noexcept
        {
            return data.data();
        }

        /// Input: row index.
        /// Output: selected row as a Vector4.
        /// Task: support dot-product multiplication and matrix inspection.
        [[nodiscard]]
        constexpr Vector4<T> Row(std::size_t row) const noexcept
        {
            assert(row < Rows);
            return { (*this)(row, 0), (*this)(row, 1), (*this)(row, 2), (*this)(row, 3) };
        }

        /// Input: column index.
        /// Output: selected column as a Vector4.
        /// Task: extract basis vectors, translation, and scale from transforms.
        [[nodiscard]]
        constexpr Vector4<T> Column(std::size_t column) const noexcept
        {
            assert(column < Columns);
            return { (*this)(0, column), (*this)(1, column), (*this)(2, column), (*this)(3, column) };
        }

        /// Input: row index and replacement row.
        /// Output: none; mutates this matrix.
        /// Task: build view/projection/transform matrices clearly.
        constexpr void SetRow(std::size_t row, const Vector4<T>& value) noexcept
        {
            assert(row < Rows);
            (*this)(row, 0) = value.x;
            (*this)(row, 1) = value.y;
            (*this)(row, 2) = value.z;
            (*this)(row, 3) = value.w;
        }

        /// Input: column index and replacement column.
        /// Output: none; mutates this matrix.
        /// Task: install basis, translation, or projection columns.
        constexpr void SetColumn(std::size_t column, const Vector4<T>& value) noexcept
        {
            assert(column < Columns);
            (*this)(0, column) = value.x;
            (*this)(1, column) = value.y;
            (*this)(2, column) = value.z;
            (*this)(3, column) = value.w;
        }

        /// Input: another matrix.
        /// Output: exact element-wise equality.
        /// Task: deterministic comparisons for exact matrices. Use NearlyEqual()
        /// for floating-point tolerance checks.
        [[nodiscard]]
        constexpr bool operator==(const Matrix4&) const noexcept = default;
    };

    //=========================================================
    // Compound Operators
    //=========================================================

    /// Input: two matrices. Output: lhs after element-wise addition.
    /// Task: in-place addition without allocating temporaries.
    template<Arithmetic T>
    constexpr Matrix3<T>& operator+=(Matrix3<T>& lhs, const Matrix3<T>& rhs) noexcept
    {
        for (std::size_t i = 0; i < Matrix3<T>::ElementCount; ++i)
        {
            lhs[i] += rhs[i];
        }
        return lhs;
    }

    /// Input: two matrices. Output: lhs after element-wise subtraction.
    /// Task: in-place subtraction without allocating temporaries.
    template<Arithmetic T>
    constexpr Matrix3<T>& operator-=(Matrix3<T>& lhs, const Matrix3<T>& rhs) noexcept
    {
        for (std::size_t i = 0; i < Matrix3<T>::ElementCount; ++i)
        {
            lhs[i] -= rhs[i];
        }
        return lhs;
    }

    /// Input: matrix and scalar. Output: matrix after uniform scaling.
    /// Task: in-place scalar multiplication.
    template<Arithmetic T>
    constexpr Matrix3<T>& operator*=(Matrix3<T>& matrix, T scalar) noexcept
    {
        for (auto& element : matrix.data)
        {
            element *= scalar;
        }
        return matrix;
    }

    /// Input: matrix and non-zero scalar. Output: matrix after uniform division.
    /// Task: in-place scalar division; asserts on programmer error.
    template<Arithmetic T>
    constexpr Matrix3<T>& operator/=(Matrix3<T>& matrix, T scalar) noexcept
    {
        assert(scalar != T(0));
        if constexpr (FloatingPoint<T>)
        {
            const T inverseScalar = T(1) / scalar;
            for (auto& element : matrix.data)
            {
                element *= inverseScalar;
            }
        }
        else
        {
            for (auto& element : matrix.data)
            {
                element /= scalar;
            }
        }
        return matrix;
    }

    /// Input: two matrices. Output: lhs after element-wise addition.
    /// Task: in-place addition without allocating temporaries.
    template<Arithmetic T>
    constexpr Matrix4<T>& operator+=(Matrix4<T>& lhs, const Matrix4<T>& rhs) noexcept
    {
        for (std::size_t i = 0; i < Matrix4<T>::ElementCount; ++i)
        {
            lhs[i] += rhs[i];
        }
        return lhs;
    }

    /// Input: two matrices. Output: lhs after element-wise subtraction.
    /// Task: in-place subtraction without allocating temporaries.
    template<Arithmetic T>
    constexpr Matrix4<T>& operator-=(Matrix4<T>& lhs, const Matrix4<T>& rhs) noexcept
    {
        for (std::size_t i = 0; i < Matrix4<T>::ElementCount; ++i)
        {
            lhs[i] -= rhs[i];
        }
        return lhs;
    }

    /// Input: matrix and scalar. Output: matrix after uniform scaling.
    /// Task: in-place scalar multiplication.
    template<Arithmetic T>
    constexpr Matrix4<T>& operator*=(Matrix4<T>& matrix, T scalar) noexcept
    {
        for (auto& element : matrix.data)
        {
            element *= scalar;
        }
        return matrix;
    }

    /// Input: matrix and non-zero scalar. Output: matrix after uniform division.
    /// Task: in-place scalar division; asserts on programmer error.
    template<Arithmetic T>
    constexpr Matrix4<T>& operator/=(Matrix4<T>& matrix, T scalar) noexcept
    {
        assert(scalar != T(0));
        if constexpr (FloatingPoint<T>)
        {
            const T inverseScalar = T(1) / scalar;
            for (auto& element : matrix.data)
            {
                element *= inverseScalar;
            }
        }
        else
        {
            for (auto& element : matrix.data)
            {
                element /= scalar;
            }
        }
        return matrix;
    }

    //=========================================================
    // Binary Operators
    //=========================================================

    /// Input: two matrices. Output: element-wise sum. Task: value addition.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Matrix3<T> operator+(Matrix3<T> lhs, const Matrix3<T>& rhs) noexcept
    {
        return lhs += rhs;
    }

    /// Input: two matrices. Output: element-wise difference. Task: value subtraction.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Matrix3<T> operator-(Matrix3<T> lhs, const Matrix3<T>& rhs) noexcept
    {
        return lhs -= rhs;
    }

    /// Input: matrix and scalar. Output: scaled matrix. Task: value scaling.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Matrix3<T> operator*(Matrix3<T> lhs, T scalar) noexcept
    {
        return lhs *= scalar;
    }

    /// Input: scalar and matrix. Output: scaled matrix. Task: symmetric scalar multiply.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Matrix3<T> operator*(T scalar, Matrix3<T> rhs) noexcept
    {
        return rhs *= scalar;
    }

    /// Input: matrix and non-zero scalar. Output: divided matrix.
    /// Task: value scalar division.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Matrix3<T> operator/(Matrix3<T> lhs, T scalar) noexcept
    {
        return lhs /= scalar;
    }

    /// Input: two matrices. Output: element-wise sum. Task: value addition.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Matrix4<T> operator+(Matrix4<T> lhs, const Matrix4<T>& rhs) noexcept
    {
        return lhs += rhs;
    }

    /// Input: two matrices. Output: element-wise difference. Task: value subtraction.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Matrix4<T> operator-(Matrix4<T> lhs, const Matrix4<T>& rhs) noexcept
    {
        return lhs -= rhs;
    }

    /// Input: matrix and scalar. Output: scaled matrix. Task: value scaling.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Matrix4<T> operator*(Matrix4<T> lhs, T scalar) noexcept
    {
        return lhs *= scalar;
    }

    /// Input: scalar and matrix. Output: scaled matrix. Task: symmetric scalar multiply.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Matrix4<T> operator*(T scalar, Matrix4<T> rhs) noexcept
    {
        return rhs *= scalar;
    }

    /// Input: matrix and non-zero scalar. Output: divided matrix.
    /// Task: value scalar division.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Matrix4<T> operator/(Matrix4<T> lhs, T scalar) noexcept
    {
        return lhs /= scalar;
    }

    //=========================================================
    // Matrix Multiplication
    //=========================================================

    /// Input: two 3x3 matrices.
    /// Output: matrix product lhs * rhs.
    /// Task: compose linear transforms. The implementation is deliberately
    /// simple and readable; SIMD/tiled paths can be added later behind the same API.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Matrix3<T> operator*(const Matrix3<T>& lhs, const Matrix3<T>& rhs) noexcept
    {
        Matrix3<T> result;

        for (std::size_t row = 0; row < Matrix3<T>::Rows; ++row)
        {
            for (std::size_t column = 0; column < Matrix3<T>::Columns; ++column)
            {
                T sum = T(0);
                for (std::size_t k = 0; k < Matrix3<T>::Columns; ++k)
                {
                    sum += lhs(row, k) * rhs(k, column);
                }
                result(row, column) = sum;
            }
        }

        return result;
    }

    /// Input: two 4x4 matrices.
    /// Output: matrix product lhs * rhs.
    /// Task: compose transforms. With column-vector convention, `A * B * v`
    /// applies B first, then A.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Matrix4<T> operator*(const Matrix4<T>& lhs, const Matrix4<T>& rhs) noexcept
    {
        Matrix4<T> result;

        for (std::size_t row = 0; row < Matrix4<T>::Rows; ++row)
        {
            for (std::size_t column = 0; column < Matrix4<T>::Columns; ++column)
            {
                T sum = T(0);
                for (std::size_t k = 0; k < Matrix4<T>::Columns; ++k)
                {
                    sum += lhs(row, k) * rhs(k, column);
                }
                result(row, column) = sum;
            }
        }

        return result;
    }

    /// Input: 3x3 matrix and 3D vector.
    /// Output: transformed vector.
    /// Task: apply a linear transform using column-vector convention.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector3<T> operator*(const Matrix3<T>& matrix, const Vector3<T>& vector) noexcept
    {
        return
        {
            Dot(matrix.Row(0), vector),
            Dot(matrix.Row(1), vector),
            Dot(matrix.Row(2), vector)
        };
    }

    /// Input: 4x4 matrix and 4D vector.
    /// Output: transformed vector.
    /// Task: apply a homogeneous transform using column-vector convention.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Vector4<T> operator*(const Matrix4<T>& matrix, const Vector4<T>& vector) noexcept
    {
        return
        {
            Dot(matrix.Row(0), vector),
            Dot(matrix.Row(1), vector),
            Dot(matrix.Row(2), vector),
            Dot(matrix.Row(3), vector)
        };
    }

    //=========================================================
    // Basic Matrix Algorithms
    //=========================================================

    /// Input: matrix.
    /// Output: transpose where rows and columns are swapped.
    /// Task: convert between row/column access and support inverse-transpose normal math.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Matrix3<T> Transpose(const Matrix3<T>& matrix) noexcept
    {
        Matrix3<T> result;
        for (std::size_t row = 0; row < Matrix3<T>::Rows; ++row)
        {
            for (std::size_t column = 0; column < Matrix3<T>::Columns; ++column)
            {
                result(column, row) = matrix(row, column);
            }
        }
        return result;
    }

    /// Input: matrix.
    /// Output: transpose where rows and columns are swapped.
    /// Task: support inverse-transpose normal math and projection manipulation.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Matrix4<T> Transpose(const Matrix4<T>& matrix) noexcept
    {
        Matrix4<T> result;
        for (std::size_t row = 0; row < Matrix4<T>::Rows; ++row)
        {
            for (std::size_t column = 0; column < Matrix4<T>::Columns; ++column)
            {
                result(column, row) = matrix(row, column);
            }
        }
        return result;
    }

    /// Input: matrix.
    /// Output: sum of diagonal elements.
    /// Task: useful for diagnostics and future quaternion extraction.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr T Trace(const Matrix3<T>& matrix) noexcept
    {
        return matrix(0, 0) + matrix(1, 1) + matrix(2, 2);
    }

    /// Input: matrix.
    /// Output: sum of diagonal elements.
    /// Task: useful for diagnostics and future decomposition work.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr T Trace(const Matrix4<T>& matrix) noexcept
    {
        return matrix(0, 0) + matrix(1, 1) + matrix(2, 2) + matrix(3, 3);
    }

    /// Input: two matrices and absolute epsilon.
    /// Output: true when every element is within epsilon.
    /// Task: floating-point comparison for tests and transform validation.
    template<FloatingPoint T>
    [[nodiscard]]
    bool NearlyEqual(
        const Matrix3<T>& lhs,
        const Matrix3<T>& rhs,
        T epsilon = std::numeric_limits<T>::epsilon() * T(10)) noexcept
    {
        for (std::size_t i = 0; i < Matrix3<T>::ElementCount; ++i)
        {
            if (!NearlyEqual(lhs[i], rhs[i], epsilon))
            {
                return false;
            }
        }
        return true;
    }

    /// Input: two matrices and absolute epsilon.
    /// Output: true when every element is within epsilon.
    /// Task: floating-point comparison for tests and transform validation.
    template<FloatingPoint T>
    [[nodiscard]]
    bool NearlyEqual(
        const Matrix4<T>& lhs,
        const Matrix4<T>& rhs,
        T epsilon = std::numeric_limits<T>::epsilon() * T(10)) noexcept
    {
        for (std::size_t i = 0; i < Matrix4<T>::ElementCount; ++i)
        {
            if (!NearlyEqual(lhs[i], rhs[i], epsilon))
            {
                return false;
            }
        }
        return true;
    }

    /// Input: matrix and absolute epsilon.
    /// Output: true when matrix is nearly identity.
    /// Task: validate transforms after inversion/composition.
    template<FloatingPoint T>
    [[nodiscard]]
    bool IsIdentity(
        const Matrix3<T>& matrix,
        T epsilon = std::numeric_limits<T>::epsilon() * T(10)) noexcept
    {
        return NearlyEqual(matrix, Matrix3<T>::Identity(), epsilon);
    }

    /// Input: matrix and absolute epsilon.
    /// Output: true when matrix is nearly identity.
    /// Task: validate transforms after inversion/composition.
    template<FloatingPoint T>
    [[nodiscard]]
    bool IsIdentity(
        const Matrix4<T>& matrix,
        T epsilon = std::numeric_limits<T>::epsilon() * T(10)) noexcept
    {
        return NearlyEqual(matrix, Matrix4<T>::Identity(), epsilon);
    }

    //=========================================================
    // Determinants and Inverses
    //=========================================================

    /// Input: 3x3 matrix.
    /// Output: determinant.
    /// Task: measure invertibility, signed area/volume scale, and orientation.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr T Determinant(const Matrix3<T>& matrix) noexcept
    {
        return
            matrix(0, 0) *
            (
                matrix(1, 1) * matrix(2, 2) -
                matrix(1, 2) * matrix(2, 1)
            )
            -
            matrix(0, 1) *
            (
                matrix(1, 0) * matrix(2, 2) -
                matrix(1, 2) * matrix(2, 0)
            )
            +
            matrix(0, 2) *
            (
                matrix(1, 0) * matrix(2, 1) -
                matrix(1, 1) * matrix(2, 0)
            );
    }

    /// Input: 3x3 matrix.
    /// Output: adjugate matrix.
    /// Task: helper for inverse computation. The cofactor matrix is transposed
    /// at the end to form the adjugate.
    template<Arithmetic T>
    [[nodiscard]]
    constexpr Matrix3<T> Adjugate(const Matrix3<T>& matrix) noexcept
    {
        Matrix3<T> cofactor;

        cofactor(0, 0) = matrix(1, 1) * matrix(2, 2) - matrix(1, 2) * matrix(2, 1);
        cofactor(0, 1) = -(matrix(1, 0) * matrix(2, 2) - matrix(1, 2) * matrix(2, 0));
        cofactor(0, 2) = matrix(1, 0) * matrix(2, 1) - matrix(1, 1) * matrix(2, 0);

        cofactor(1, 0) = -(matrix(0, 1) * matrix(2, 2) - matrix(0, 2) * matrix(2, 1));
        cofactor(1, 1) = matrix(0, 0) * matrix(2, 2) - matrix(0, 2) * matrix(2, 0);
        cofactor(1, 2) = -(matrix(0, 0) * matrix(2, 1) - matrix(0, 1) * matrix(2, 0));

        cofactor(2, 0) = matrix(0, 1) * matrix(1, 2) - matrix(0, 2) * matrix(1, 1);
        cofactor(2, 1) = -(matrix(0, 0) * matrix(1, 2) - matrix(0, 2) * matrix(1, 0));
        cofactor(2, 2) = matrix(0, 0) * matrix(1, 1) - matrix(0, 1) * matrix(1, 0);

        return Transpose(cofactor);
    }

    /// Input: invertible 3x3 floating-point matrix.
    /// Output: inverse matrix, or identity when singular in release builds.
    /// Task: compute inverse for normal matrices and rotation/scale extraction.
    /// Note: singular inversion is a programmer/data error. The assert makes the
    /// problem loud in debug builds while preserving noexcept behavior.
    template<FloatingPoint T>
    [[nodiscard]]
    Matrix3<T> Inverse(const Matrix3<T>& matrix) noexcept
    {
        const T determinant = Determinant(matrix);
        assert(std::abs(determinant) > std::numeric_limits<T>::epsilon());

        if (std::abs(determinant) <= std::numeric_limits<T>::epsilon())
        {
            return Matrix3<T>::Identity();
        }

        return Adjugate(matrix) / determinant;
    }

    /// Input: 4x4 matrix.
    /// Output: determinant.
    /// Task: measure invertibility and volume scale for homogeneous transforms.
    template<Arithmetic T>
    [[nodiscard]]
    T Determinant(const Matrix4<T>& matrix) noexcept
    {
        const T a0 = matrix(0, 0) * matrix(1, 1) - matrix(1, 0) * matrix(0, 1);
        const T a1 = matrix(0, 0) * matrix(1, 2) - matrix(1, 0) * matrix(0, 2);
        const T a2 = matrix(0, 0) * matrix(1, 3) - matrix(1, 0) * matrix(0, 3);
        const T a3 = matrix(0, 1) * matrix(1, 2) - matrix(1, 1) * matrix(0, 2);
        const T a4 = matrix(0, 1) * matrix(1, 3) - matrix(1, 1) * matrix(0, 3);
        const T a5 = matrix(0, 2) * matrix(1, 3) - matrix(1, 2) * matrix(0, 3);

        const T b0 = matrix(2, 0) * matrix(3, 1) - matrix(3, 0) * matrix(2, 1);
        const T b1 = matrix(2, 0) * matrix(3, 2) - matrix(3, 0) * matrix(2, 2);
        const T b2 = matrix(2, 0) * matrix(3, 3) - matrix(3, 0) * matrix(2, 3);
        const T b3 = matrix(2, 1) * matrix(3, 2) - matrix(3, 1) * matrix(2, 2);
        const T b4 = matrix(2, 1) * matrix(3, 3) - matrix(3, 1) * matrix(2, 3);
        const T b5 = matrix(2, 2) * matrix(3, 3) - matrix(3, 2) * matrix(2, 3);

        return
            a0 * b5 -
            a1 * b4 +
            a2 * b3 +
            a3 * b2 -
            a4 * b1 +
            a5 * b0;
    }

    /// Input: invertible 4x4 floating-point matrix.
    /// Output: inverse matrix, or identity when singular in release builds.
    /// Task: compute inverse for cameras, picking rays, transforms, normal
    /// matrices, and transform decomposition.
    ///
    /// Implementation note:
    /// This uses Gauss-Jordan elimination with partial pivoting. It is readable,
    /// robust enough for foundation code, allocation-free, and easy to replace
    /// later with a SIMD-specialized affine inverse for hot transform paths.
    template<FloatingPoint T>
    [[nodiscard]]
    Matrix4<T> Inverse(const Matrix4<T>& matrix) noexcept
    {
        Matrix4<T> working = matrix;
        Matrix4<T> inverse = Matrix4<T>::Identity();

        for (std::size_t column = 0; column < Matrix4<T>::Columns; ++column)
        {
            std::size_t pivotRow = column;
            T pivotMagnitude = std::abs(working(column, column));

            for (std::size_t row = column + 1; row < Matrix4<T>::Rows; ++row)
            {
                const T candidateMagnitude = std::abs(working(row, column));
                if (candidateMagnitude > pivotMagnitude)
                {
                    pivotMagnitude = candidateMagnitude;
                    pivotRow = row;
                }
            }

            assert(pivotMagnitude > std::numeric_limits<T>::epsilon());
            if (pivotMagnitude <= std::numeric_limits<T>::epsilon())
            {
                return Matrix4<T>::Identity();
            }

            if (pivotRow != column)
            {
                for (std::size_t currentColumn = 0; currentColumn < Matrix4<T>::Columns; ++currentColumn)
                {
                    const T workingTemp = working(column, currentColumn);
                    working(column, currentColumn) = working(pivotRow, currentColumn);
                    working(pivotRow, currentColumn) = workingTemp;

                    const T inverseTemp = inverse(column, currentColumn);
                    inverse(column, currentColumn) = inverse(pivotRow, currentColumn);
                    inverse(pivotRow, currentColumn) = inverseTemp;
                }
            }

            const T pivot = working(column, column);
            for (std::size_t currentColumn = 0; currentColumn < Matrix4<T>::Columns; ++currentColumn)
            {
                working(column, currentColumn) /= pivot;
                inverse(column, currentColumn) /= pivot;
            }

            for (std::size_t row = 0; row < Matrix4<T>::Rows; ++row)
            {
                if (row == column)
                {
                    continue;
                }

                const T factor = working(row, column);
                for (std::size_t currentColumn = 0; currentColumn < Matrix4<T>::Columns; ++currentColumn)
                {
                    working(row, currentColumn) -= factor * working(column, currentColumn);
                    inverse(row, currentColumn) -= factor * inverse(column, currentColumn);
                }
            }
        }

        return inverse;
    }

    //=========================================================
    // Transform Helpers
    //=========================================================

    /// Input: translation vector.
    /// Output: 4x4 translation matrix.
    /// Task: build a transform that moves points but does not affect directions.
    template<FloatingPoint T>
    [[nodiscard]]
    Matrix4<T> MakeTranslation(const Vector3<T>& translation) noexcept
    {
        Matrix4<T> result = Matrix4<T>::Identity();
        result(0, 3) = translation.x;
        result(1, 3) = translation.y;
        result(2, 3) = translation.z;
        return result;
    }

    /// Input: scale vector.
    /// Output: 4x4 scale matrix.
    /// Task: build a transform that scales points and directions around origin.
    template<FloatingPoint T>
    [[nodiscard]]
    Matrix4<T> MakeScale(const Vector3<T>& scale) noexcept
    {
        Matrix4<T> result = Matrix4<T>::Identity();
        result(0, 0) = scale.x;
        result(1, 1) = scale.y;
        result(2, 2) = scale.z;
        return result;
    }

    /// Input: angle in radians.
    /// Output: 4x4 rotation matrix around the X axis.
    /// Task: construct right-handed X-axis rotation.
    template<FloatingPoint T>
    [[nodiscard]]
    Matrix4<T> MakeRotationX(T radians) noexcept
    {
        const T cosine = std::cos(radians);
        const T sine = std::sin(radians);

        Matrix4<T> result = Matrix4<T>::Identity();
        result(1, 1) = cosine;
        result(1, 2) = -sine;
        result(2, 1) = sine;
        result(2, 2) = cosine;
        return result;
    }

    /// Input: angle in radians.
    /// Output: 4x4 rotation matrix around the Y axis.
    /// Task: construct right-handed Y-axis rotation.
    template<FloatingPoint T>
    [[nodiscard]]
    Matrix4<T> MakeRotationY(T radians) noexcept
    {
        const T cosine = std::cos(radians);
        const T sine = std::sin(radians);

        Matrix4<T> result = Matrix4<T>::Identity();
        result(0, 0) = cosine;
        result(0, 2) = sine;
        result(2, 0) = -sine;
        result(2, 2) = cosine;
        return result;
    }

    /// Input: angle in radians.
    /// Output: 4x4 rotation matrix around the Z axis.
    /// Task: construct right-handed Z-axis rotation.
    template<FloatingPoint T>
    [[nodiscard]]
    Matrix4<T> MakeRotationZ(T radians) noexcept
    {
        const T cosine = std::cos(radians);
        const T sine = std::sin(radians);

        Matrix4<T> result = Matrix4<T>::Identity();
        result(0, 0) = cosine;
        result(0, 1) = -sine;
        result(1, 0) = sine;
        result(1, 1) = cosine;
        return result;
    }

    /// Input: point and transform matrix.
    /// Output: transformed point.
    /// Task: transform a position using homogeneous w = 1, so translation affects it.
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr Vector3<T> TransformPoint(const Matrix4<T>& matrix, const Vector3<T>& point) noexcept
    {
        const Vector4<T> result = matrix * Vector4<T>{ point, T(1) };
        return { result.x, result.y, result.z };
    }

    /// Input: direction and transform matrix.
    /// Output: transformed direction.
    /// Task: transform a vector using homogeneous w = 0, so translation is ignored.
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr Vector3<T> TransformDirection(const Matrix4<T>& matrix, const Vector3<T>& direction) noexcept
    {
        const Vector4<T> result = matrix * Vector4<T>{ direction, T(0) };
        return { result.x, result.y, result.z };
    }

    /// Input: vector and transform matrix.
    /// Output: transformed direction.
    /// Task: semantic alias for TransformDirection().
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr Vector3<T> TransformVector(const Matrix4<T>& matrix, const Vector3<T>& vector) noexcept
    {
        return TransformDirection(matrix, vector);
    }

    /// Input: surface normal and transform matrix.
    /// Output: transformed and normalized normal.
    /// Task: transform normals correctly under non-uniform scale using the
    /// inverse-transpose matrix. Directions and normals are not equivalent when
    /// scale or shear is present.
    template<FloatingPoint T>
    [[nodiscard]]
    Vector3<T> TransformNormal(const Matrix4<T>& matrix, const Vector3<T>& normal) noexcept
    {
        const Matrix4<T> normalMatrix = Transpose(Inverse(matrix));
        return SafeNormalize(TransformDirection(normalMatrix, normal));
    }

    //=========================================================
    // Camera and Projection
    //=========================================================

    /// Input: vertical field of view in radians, aspect ratio, near plane, far plane.
    /// Output: right-handed perspective projection matrix with depth range [0, 1].
    /// Task: build a Vulkan/Metal-friendly projection. Callers must pass valid
    /// positive planes, aspect ratio, and non-zero near/far separation.
    template<FloatingPoint T>
    [[nodiscard]]
    Matrix4<T> Perspective(
        T fieldOfViewRadians,
        T aspectRatio,
        T nearPlane,
        T farPlane) noexcept
    {
        assert(aspectRatio != T(0));
        assert(nearPlane != farPlane);

        const T tanHalfFov = std::tan(fieldOfViewRadians / T(2));
        assert(tanHalfFov != T(0));

        Matrix4<T> result {};
        result(0, 0) = T(1) / (aspectRatio * tanHalfFov);
        result(1, 1) = T(1) / tanHalfFov;
        result(2, 2) = farPlane / (nearPlane - farPlane);
        result(2, 3) = (farPlane * nearPlane) / (nearPlane - farPlane);
        result(3, 2) = T(-1);
        return result;
    }

    /// Input: box bounds and near/far planes.
    /// Output: right-handed orthographic projection matrix with depth range [0, 1].
    /// Task: build an orthographic camera/projection matrix for Vulkan-style depth.
    template<FloatingPoint T>
    [[nodiscard]]
    Matrix4<T> Orthographic(
        T left,
        T right,
        T bottom,
        T top,
        T nearPlane,
        T farPlane) noexcept
    {
        assert(left != right);
        assert(bottom != top);
        assert(nearPlane != farPlane);

        Matrix4<T> result = Matrix4<T>::Identity();
        result(0, 0) = T(2) / (right - left);
        result(1, 1) = T(2) / (top - bottom);
        result(2, 2) = T(1) / (nearPlane - farPlane);
        result(0, 3) = -(right + left) / (right - left);
        result(1, 3) = -(top + bottom) / (top - bottom);
        result(2, 3) = nearPlane / (nearPlane - farPlane);
        return result;
    }

    /// Input: eye position, target position, and approximate up direction.
    /// Output: right-handed view matrix.
    /// Task: construct a camera matrix using the engine convention +X right,
    /// +Y up, and -Z forward. The up vector should not be parallel to view direction.
    template<FloatingPoint T>
    [[nodiscard]]
    Matrix4<T> LookAt(
        const Vector3<T>& eye,
        const Vector3<T>& target,
        const Vector3<T>& up) noexcept
    {
        const Vector3<T> forward = SafeNormalize(eye - target);
        const Vector3<T> right = SafeNormalize(Cross(up, forward));
        const Vector3<T> correctedUp = Cross(forward, right);

        Matrix4<T> result = Matrix4<T>::Identity();
        result.SetRow(0, Vector4<T>{ right, -Dot(right, eye) });
        result.SetRow(1, Vector4<T>{ correctedUp, -Dot(correctedUp, eye) });
        result.SetRow(2, Vector4<T>{ forward, -Dot(forward, eye) });
        return result;
    }

    //=========================================================
    // Transform Decomposition Helpers
    //=========================================================

    /// Input: transform matrix.
    /// Output: translation component.
    /// Task: extract the position stored in the final column of this row-major,
    /// column-vector transform convention.
    template<FloatingPoint T>
    [[nodiscard]]
    constexpr Vector3<T> ExtractTranslation(const Matrix4<T>& matrix) noexcept
    {
        return { matrix(0, 3), matrix(1, 3), matrix(2, 3) };
    }

    /// Input: transform matrix.
    /// Output: magnitudes of the basis columns.
    /// Task: extract non-uniform scale from an affine transform. Negative scale
    /// and shear require more advanced decomposition later.
    template<FloatingPoint T>
    [[nodiscard]]
    Vector3<T> ExtractScale(const Matrix4<T>& matrix) noexcept
    {
        return
        {
            matrix.Column(0).XYZ().Length(),
            matrix.Column(1).XYZ().Length(),
            matrix.Column(2).XYZ().Length()
        };
    }

    /// Input: affine transform matrix.
    /// Output: 3x3 rotation matrix with scale removed from basis columns.
    /// Task: prepare rotation extraction for future Quaternion and Transform
    /// modules. If a scale axis is zero, that column falls back to zero because
    /// no stable rotation axis can be recovered from a collapsed basis.
    template<FloatingPoint T>
    [[nodiscard]]
    Matrix3<T> ExtractRotationMatrix(const Matrix4<T>& matrix) noexcept
    {
        const Vector3<T> scale = ExtractScale(matrix);

        Matrix3<T> result = Matrix3<T>::Identity();
        const Vector3<T> xAxis = matrix.Column(0).XYZ();
        const Vector3<T> yAxis = matrix.Column(1).XYZ();
        const Vector3<T> zAxis = matrix.Column(2).XYZ();

        result.SetColumn(
            0,
            scale.x > std::numeric_limits<T>::epsilon()
                ? xAxis / scale.x
                : Vector3<T>::Zero());

        result.SetColumn(
            1,
            scale.y > std::numeric_limits<T>::epsilon()
                ? yAxis / scale.y
                : Vector3<T>::Zero());

        result.SetColumn(
            2,
            scale.z > std::numeric_limits<T>::epsilon()
                ? zAxis / scale.z
                : Vector3<T>::Zero());

        return result;
    }

    //=========================================================
    // Aliases
    //=========================================================

    using Mat3f = Matrix3<float>;
    using Mat3d = Matrix3<double>;

    using Mat4f = Matrix4<float>;
    using Mat4d = Matrix4<double>;

    //=========================================================
    // Compile-Time Validation
    //=========================================================

    static_assert(Matrix3<float>::Rows == 3);
    static_assert(Matrix3<float>::Columns == 3);
    static_assert(Matrix3<float>::ElementCount == 9);

    static_assert(Matrix4<float>::Rows == 4);
    static_assert(Matrix4<float>::Columns == 4);
    static_assert(Matrix4<float>::ElementCount == 16);

    static_assert(std::is_trivially_copyable_v<Mat3f>);
    static_assert(std::is_trivially_copyable_v<Mat4f>);

    static_assert(std::is_standard_layout_v<Mat3f>);
    static_assert(std::is_standard_layout_v<Mat4f>);

    static_assert(sizeof(Mat3f) == sizeof(float) * 9);
    static_assert(sizeof(Mat4f) == sizeof(float) * 16);

} // namespace foundation::math
