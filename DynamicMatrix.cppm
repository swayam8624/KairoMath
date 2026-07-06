module;

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <ostream>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

export module Kairo.Foundation.Math.DynamicMatrix;

import Kairo.Foundation.Math.Vector;

export namespace kairo::foundation::math
{
    /// A dynamically-sized row-major matrix.
    template<Arithmetic T>
    class DynamicMatrix final
    {
    private:
        std::size_t m_rows = 0;
        std::size_t m_cols = 0;
        std::vector<T> m_data;

        [[nodiscard]]
        static std::size_t CheckedElementCount(std::size_t rows, std::size_t cols)
        {
            if ((rows == 0) != (cols == 0))
            {
                throw std::invalid_argument("DynamicMatrix shape error: rows and columns must both be zero or both be non-zero.");
            }

            if (cols != 0 && rows > std::numeric_limits<std::size_t>::max() / cols)
            {
                throw std::overflow_error("DynamicMatrix size overflow: rows * columns exceeds size_t.");
            }

            return rows * cols;
        }

        [[nodiscard]]
        std::size_t CheckedOffset(std::size_t row, std::size_t col) const
        {
            if (row >= m_rows || col >= m_cols)
            {
                throw std::out_of_range("DynamicMatrix index error: row or column is out of range.");
            }

            return (row * m_cols) + col;
        }

        [[nodiscard]]
        std::size_t CheckedRowOffset(std::size_t row) const
        {
            if (row >= m_rows)
            {
                throw std::out_of_range("DynamicMatrix row error: row is out of range.");
            }

            return row * m_cols;
        }

    public:
        using ValueType = T;

        /// Default construction: empty 0x0 matrix.
        constexpr DynamicMatrix() noexcept = default;

        DynamicMatrix(const DynamicMatrix&) = default;

        DynamicMatrix& operator=(const DynamicMatrix&) = default;

        /// Input: source matrix to move from.
        /// Output: this matrix receives the source storage; source becomes 0x0.
        /// Task: preserve the class invariant that empty storage implies an
        /// empty shape even after move construction.
        DynamicMatrix(DynamicMatrix&& other) noexcept
            : m_rows(std::exchange(other.m_rows, 0))
            , m_cols(std::exchange(other.m_cols, 0))
            , m_data(std::move(other.m_data))
        {
            other.m_data.clear();
        }

        /// Input: source matrix to move from.
        /// Output: this matrix receives the source storage; source becomes 0x0.
        /// Task: keep moved-from matrices valid and safely reusable.
        DynamicMatrix& operator=(DynamicMatrix&& other) noexcept
        {
            if (this != &other)
            {
                m_rows = std::exchange(other.m_rows, 0);
                m_cols = std::exchange(other.m_cols, 0);
                m_data = std::move(other.m_data);
                other.m_data.clear();
            }

            return *this;
        }

        /// Sized construction: zero-filled rows x cols matrix.
        DynamicMatrix(std::size_t rows, std::size_t cols)
            : m_rows(rows)
            , m_cols(cols)
            , m_data(CheckedElementCount(rows, cols), T(0))
        {
        }

        /// Sized construction with initial value: rows x cols matrix filled with initialValue.
        DynamicMatrix(std::size_t rows, std::size_t cols, T initialValue)
            : m_rows(rows)
            , m_cols(cols)
            , m_data(CheckedElementCount(rows, cols), initialValue)
        {
        }

        /// Sized construction with vector backing: rows x cols matrix filled with supplied vector.
        DynamicMatrix(std::size_t rows, std::size_t cols, std::vector<T> data)
            : m_rows(rows)
            , m_cols(cols)
            , m_data(std::move(data))
        {
            if (m_data.size() != CheckedElementCount(rows, cols))
            {
                throw std::invalid_argument("DynamicMatrix constructor error: vector size does not match rows * cols.");
            }
        }

        /// Input: matrix shape and row-major initializer-list values.
        /// Output: rows x cols matrix containing the supplied values.
        /// Task: make tests, examples, and small literal matrices explicit
        /// without relying on implicit initializer-list-to-vector conversion.
        DynamicMatrix(std::size_t rows, std::size_t cols, std::initializer_list<T> values)
            : DynamicMatrix(rows, cols, std::vector<T>(values))
        {
        }

        /// Get row count.
        [[nodiscard]]
        constexpr std::size_t Rows() const noexcept
        {
            return m_rows;
        }

        /// Get column count.
        [[nodiscard]]
        constexpr std::size_t Columns() const noexcept
        {
            return m_cols;
        }

        /// Get total element count.
        [[nodiscard]]
        constexpr std::size_t Size() const noexcept
        {
            return m_data.size();
        }

        /// Check if matrix has 0 rows or columns.
        [[nodiscard]]
        constexpr bool Empty() const noexcept
        {
            return m_rows == 0 || m_cols == 0;
        }

        /// Expose underlying contiguous memory pointer.
        [[nodiscard]]
        constexpr T* Data() noexcept
        {
            return m_data.data();
        }

        /// Expose underlying contiguous memory pointer (const).
        [[nodiscard]]
        constexpr const T* Data() const noexcept
        {
            return m_data.data();
        }

        /// 2D element indexing (mutable).
        [[nodiscard]]
        constexpr T& operator()(std::size_t row, std::size_t col) noexcept
        {
            assert(row < m_rows);
            assert(col < m_cols);
            return m_data[(row * m_cols) + col];
        }

        /// 2D element indexing (const).
        [[nodiscard]]
        constexpr const T& operator()(std::size_t row, std::size_t col) const noexcept
        {
            assert(row < m_rows);
            assert(col < m_cols);
            return m_data[(row * m_cols) + col];
        }

        /// Input: row and column from runtime/user input.
        /// Output: mutable element reference.
        /// Task: checked 2D access for public APIs and data-driven callers.
        [[nodiscard]]
        T& At(std::size_t row, std::size_t col)
        {
            return m_data[CheckedOffset(row, col)];
        }

        /// Input: row and column from runtime/user input.
        /// Output: const element reference.
        /// Task: checked read-only 2D access for public APIs and data-driven callers.
        [[nodiscard]]
        const T& At(std::size_t row, std::size_t col) const
        {
            return m_data[CheckedOffset(row, col)];
        }

        /// Linear indexing (mutable).
        [[nodiscard]]
        constexpr T& operator[](std::size_t index) noexcept
        {
            assert(index < m_data.size());
            return m_data[index];
        }

        /// Linear indexing (const).
        [[nodiscard]]
        constexpr const T& operator[](std::size_t index) const noexcept
        {
            assert(index < m_data.size());
            return m_data[index];
        }

        /// Input: row index.
        /// Output: mutable pointer to the first element in that row.
        /// Task: efficient row-wise algorithms without repeated row*columns math.
        [[nodiscard]]
        T* RowData(std::size_t row)
        {
            return m_data.data() + CheckedRowOffset(row);
        }

        /// Input: row index.
        /// Output: const pointer to the first element in that row.
        /// Task: efficient read-only row-wise algorithms.
        [[nodiscard]]
        const T* RowData(std::size_t row) const
        {
            return m_data.data() + CheckedRowOffset(row);
        }

        /// Input: row index.
        /// Output: mutable span covering one contiguous row.
        /// Task: expose safe row views for algorithms that operate on rows.
        [[nodiscard]]
        std::span<T> RowSpan(std::size_t row)
        {
            return { RowData(row), m_cols };
        }

        /// Input: row index.
        /// Output: const span covering one contiguous row.
        /// Task: expose safe read-only row views for algorithms that operate on rows.
        [[nodiscard]]
        std::span<const T> RowSpan(std::size_t row) const
        {
            return { RowData(row), m_cols };
        }

        /// Exact equality comparison.
        [[nodiscard]]
        constexpr bool operator==(const DynamicMatrix& other) const noexcept
        {
            return m_rows == other.m_rows && m_cols == other.m_cols && m_data == other.m_data;
        }

        //=========================================================
        // Factories
        //=========================================================

        /// Create a rows x cols matrix filled with zeros.
        [[nodiscard]]
        static DynamicMatrix Zero(std::size_t rows, std::size_t cols)
        {
            return DynamicMatrix(rows, cols, T(0));
        }

        /// Create an identity matrix of size x size.
        [[nodiscard]]
        static DynamicMatrix Identity(std::size_t size)
        {
            DynamicMatrix result(size, size, T(0));
            for (std::size_t i = 0; i < size; ++i)
            {
                result(i, i) = T(1);
            }
            return result;
        }

        /// Create a diagonal square matrix from a list/vector of diagonal values.
        [[nodiscard]]
        static DynamicMatrix Diagonal(const std::vector<T>& values)
        {
            std::size_t size = values.size();
            DynamicMatrix result(size, size, T(0));
            for (std::size_t i = 0; i < size; ++i)
            {
                result(i, i) = values[i];
            }
            return result;
        }

        //=========================================================
        // Utilities
        //=========================================================

        /// Swap contents of row r1 and row r2.
        constexpr void SwapRows(std::size_t r1, std::size_t r2) noexcept
        {
            assert(r1 < m_rows);
            assert(r2 < m_rows);
            if (r1 == r2) return;

            for (std::size_t c = 0; c < m_cols; ++c)
            {
                std::swap((*this)(r1, c), (*this)(r2, c));
            }
        }

        /// Swap contents of column c1 and column c2.
        constexpr void SwapColumns(std::size_t c1, std::size_t c2) noexcept
        {
            assert(c1 < m_cols);
            assert(c2 < m_cols);
            if (c1 == c2) return;

            for (std::size_t r = 0; r < m_rows; ++r)
            {
                std::swap((*this)(r, c1), (*this)(r, c2));
            }
        }

        //=========================================================
        // Matrix Operations
        //=========================================================

        /// Compute transpose of this matrix.
        [[nodiscard]]
        DynamicMatrix Transpose() const
        {
            DynamicMatrix result(m_cols, m_rows);
            for (std::size_t r = 0; r < m_rows; ++r)
            {
                for (std::size_t c = 0; c < m_cols; ++c)
                {
                    result(c, r) = (*this)(r, c);
                }
            }
            return result;
        }

        /// Compute sum of diagonal elements (must be square).
        [[nodiscard]]
        T Trace() const
        {
            if (m_rows != m_cols)
            {
                throw std::invalid_argument("DynamicMatrix::Trace failed: matrix must be square.");
            }

            T sum = T(0);
            for (std::size_t i = 0; i < m_rows; ++i)
            {
                sum += (*this)(i, i);
            }
            return sum;
        }

        /// Compute the Frobenius Norm of this matrix.
        [[nodiscard]]
        T FrobeniusNorm() const noexcept
            requires FloatingPoint<T>
        {
            T sum = T(0);
            for (const auto& val : m_data)
            {
                sum += val * val;
            }
            return std::sqrt(sum);
        }
    };

    //=========================================================
    // Compound Assignment Operators
    //=========================================================

    template<Arithmetic T>
    DynamicMatrix<T>& operator+=(DynamicMatrix<T>& lhs, const DynamicMatrix<T>& rhs)
    {
        if (lhs.Rows() != rhs.Rows() || lhs.Columns() != rhs.Columns())
        {
            throw std::invalid_argument("DynamicMatrix addition failed: dimensions must match.");
        }

        for (std::size_t i = 0; i < lhs.Size(); ++i)
        {
            lhs[i] += rhs[i];
        }
        return lhs;
    }

    template<Arithmetic T>
    DynamicMatrix<T>& operator-=(DynamicMatrix<T>& lhs, const DynamicMatrix<T>& rhs)
    {
        if (lhs.Rows() != rhs.Rows() || lhs.Columns() != rhs.Columns())
        {
            throw std::invalid_argument("DynamicMatrix subtraction failed: dimensions must match.");
        }

        for (std::size_t i = 0; i < lhs.Size(); ++i)
        {
            lhs[i] -= rhs[i];
        }
        return lhs;
    }

    template<Arithmetic T>
    DynamicMatrix<T>& operator*=(DynamicMatrix<T>& matrix, T scalar) noexcept
    {
        for (std::size_t i = 0; i < matrix.Size(); ++i)
        {
            matrix[i] *= scalar;
        }
        return matrix;
    }

    template<Arithmetic T>
    DynamicMatrix<T>& operator/=(DynamicMatrix<T>& matrix, T scalar)
    {
        if (scalar == T(0))
        {
            throw std::domain_error("DynamicMatrix division failed: scalar must be non-zero.");
        }

        if constexpr (FloatingPoint<T>)
        {
            T inv = T(1) / scalar;
            for (std::size_t i = 0; i < matrix.Size(); ++i)
            {
                matrix[i] *= inv;
            }
        }
        else
        {
            for (std::size_t i = 0; i < matrix.Size(); ++i)
            {
                matrix[i] /= scalar;
            }
        }
        return matrix;
    }

    //=========================================================
    // Binary Operators
    //=========================================================

    template<Arithmetic T>
    [[nodiscard]]
    DynamicMatrix<T> operator+(DynamicMatrix<T> lhs, const DynamicMatrix<T>& rhs)
    {
        return lhs += rhs;
    }

    template<Arithmetic T>
    [[nodiscard]]
    DynamicMatrix<T> operator-(DynamicMatrix<T> lhs, const DynamicMatrix<T>& rhs)
    {
        return lhs -= rhs;
    }

    template<Arithmetic T>
    [[nodiscard]]
    DynamicMatrix<T> operator*(DynamicMatrix<T> matrix, T scalar)
    {
        return matrix *= scalar;
    }

    template<Arithmetic T>
    [[nodiscard]]
    DynamicMatrix<T> operator*(T scalar, DynamicMatrix<T> matrix)
    {
        return matrix *= scalar;
    }

    template<Arithmetic T>
    [[nodiscard]]
    DynamicMatrix<T> operator/(DynamicMatrix<T> matrix, T scalar)
    {
        return matrix /= scalar;
    }

    /// Matrix multiplication: lhs * rhs.
    /// Optimized loop ordering (i-k-j) for cache-friendly access pattern.
    template<Arithmetic T>
    [[nodiscard]]
    DynamicMatrix<T> operator*(const DynamicMatrix<T>& lhs, const DynamicMatrix<T>& rhs)
    {
        if (lhs.Columns() != rhs.Rows())
        {
            throw std::invalid_argument("DynamicMatrix multiplication failed: lhs.Columns() must equal rhs.Rows().");
        }

        std::size_t r = lhs.Rows();
        std::size_t c = rhs.Columns();
        std::size_t k_size = lhs.Columns();

        DynamicMatrix<T> result(r, c, T(0));
        for (std::size_t i = 0; i < r; ++i)
        {
            for (std::size_t k = 0; k < k_size; ++k)
            {
                T factor = lhs(i, k);
                if (factor == T(0)) continue;
                for (std::size_t j = 0; j < c; ++j)
                {
                    result(i, j) += factor * rhs(k, j);
                }
            }
        }
        return result;
    }

    /// Input: two same-shaped floating-point matrices and a tolerance factor.
    /// Output: true when every element is close under shared scalar NearlyEqual.
    /// Task: matrix-level comparison that inherits the absolute/relative
    /// tolerance policy used by Vector and fixed-size Matrix helpers.
    template<FloatingPoint T>
    [[nodiscard]]
    bool NearlyEqual(
        const DynamicMatrix<T>& lhs,
        const DynamicMatrix<T>& rhs,
        T epsilon = std::numeric_limits<T>::epsilon() * T(10)) noexcept
    {
        if (lhs.Rows() != rhs.Rows() || lhs.Columns() != rhs.Columns())
        {
            return false;
        }
        for (std::size_t i = 0; i < lhs.Size(); ++i)
        {
            if (!kairo::foundation::math::NearlyEqual(lhs[i], rhs[i], epsilon))
            {
                return false;
            }
        }
        return true;
    }

    /// Input: floating-point square matrix and a tolerance factor.
    /// Output: true when diagonal/off-diagonal values match identity within tolerance.
    /// Task: identity validation that shares the scalar relative tolerance policy.
    template<FloatingPoint T>
    [[nodiscard]]
    bool IsIdentity(
        const DynamicMatrix<T>& matrix,
        T epsilon = std::numeric_limits<T>::epsilon() * T(10)) noexcept
    {
        if (matrix.Rows() != matrix.Columns())
        {
            return false;
        }
        for (std::size_t r = 0; r < matrix.Rows(); ++r)
        {
            for (std::size_t c = 0; c < matrix.Columns(); ++c)
            {
                T expected = (r == c) ? T(1) : T(0);
                if (!kairo::foundation::math::NearlyEqual(matrix(r, c), expected, epsilon))
                {
                    return false;
                }
            }
        }
        return true;
    }

} // namespace kairo::foundation::math
