/** \file
 * \brief Wraps around a math library and provides abstracted types and functions.
 */

#ifndef MATH_HPP
#define MATH_HPP

#include <type_traits>
#include <memory>

#include <blaze/Math.h>

#include "core.hpp"
#include "tmp.hpp"

template <typename ET>
using Vector = blaze::DynamicVector<ET>;

template <typename ET>
using Matrix = blaze::DynamicMatrix<ET>;

template <typename ET>
using SparseMatrix = blaze::CompressedMatrix<ET>;

template <typename ET>
using IdMatrix = blaze::IdentityMatrix<ET>;

using DVector = Vector<double>;
using CDVector = Vector<std::complex<double>>;

using DMatrix = Matrix<double>;
using CDMatrix = Matrix<std::complex<double>>;

using DSparseMatrix = SparseMatrix<double>;
using CDSparseMatrix = SparseMatrix<std::complex<double>>;

template <typename T>
struct ValueType
{
    using type = T;
};

template <typename T>
struct ValueType<std::complex<T>>
{
    using type = T;
};

template <typename T>
using ValueType_t = typename ValueType<T>::type;

template <typename T, typename = void>
struct ElementType
{
    static_assert(tmp::AlwaysFalse_v<T>, "Cannot deduce element type.");
};

template <typename T>
struct ElementType<T, std::enable_if_t<std::is_arithmetic<T>::value
                                       || tmp::IsSpecialization<std::complex, T>::value>>
{
    using type = T;
};

template <typename ET, bool TF>
struct ElementType<blaze::DynamicVector<ET, TF>>
{
    using type = ET;
};

template <typename ET, bool TF>
struct ElementType<blaze::DynamicMatrix<ET, TF>>
{
    using type = ET;
};

template <typename ET, bool TF>
struct ElementType<blaze::CompressedMatrix<ET, TF>>
{
    using type = ET;
};

template <typename T>
using ElementType_t = typename ElementType<T>::type;

template <typename T>
constexpr T pi = static_cast<T>(3.1415926535897932384626433832795028841971693993751058209749L);

template <typename VT>
std::size_t getNt(const VT &stVec, const std::size_t nx) noexcept(ndebug)
{
#ifndef NDEBUG
    if (stVec.size() % nx != 0) {
        throw std::runtime_error("Vector dimension does not match, expected a spacetime vector.");
    }
#endif
    return stVec.size() / nx;
}

template <typename RT>
std::complex<RT> toFirstLogBranch(const std::complex<RT> &x)
{
    return { std::real(x), std::remainder(std::imag(x), 2 * pi<RT>) };
}

template <typename VT>
decltype(auto) spacevec(VT &&vec, const std::size_t t, const std::size_t nx) noexcept(ndebug)
{
    // some rudimentary bounds check, no idea how to do this in general...
#ifndef NDEBUG
    if (t == static_cast<std::size_t>(-1))
        throw std::runtime_error("t is -1");
    if (t == static_cast<std::size_t>(-2))
        throw std::runtime_error("t is -2");
    return blaze::subvector(std::forward<VT>(vec), t * nx, nx);
#else
    return blaze::subvector(std::forward<VT>(vec), t * nx, nx, blaze::unchecked);
#endif
}

template <typename MT>
decltype(auto) spacemat(MT &&mat, const std::size_t tp, const std::size_t t,
                        const std::size_t nx) noexcept(ndebug)
{
    // some rudimentary bounds check, no idea how to do this in general...
#ifndef NDEBUG
    if (tp == static_cast<std::size_t>(-1))
        throw std::runtime_error("tp is -1");
    if (tp == static_cast<std::size_t>(-2))
        throw std::runtime_error("tp is -2");
    if (t == static_cast<std::size_t>(-1))
        throw std::runtime_error("t is -1");
    if (t == static_cast<std::size_t>(-2))
        throw std::runtime_error("t is -2");
    return blaze::submatrix(std::forward<MT>(mat), tp * nx, t * nx, nx, nx);
#else
    return blaze::submatrix(std::forward<MT>(mat), tp * nx, t * nx, nx, nx, blaze::unchecked);
#endif
}

template <typename ET>
void invert(Matrix<ET> &mat, std::unique_ptr<int[]> &ipiv)
{
    blaze::getrf(mat, ipiv.get());
    blaze::getri(mat, ipiv.get());
}

template <typename MT>
bool isInvertible(MT mat, const double eps = 1e-15)
{
    CDVector eigenvals;
    blaze::geev(mat, eigenvals);
    return blaze::min(blaze::abs(eigenvals)) > eps;
}

inline DMatrix expmSym(const DMatrix &mat)
{
    // compute eigenvalues evals and eigenvectors U
    DMatrix U = mat;
    DVector evals;
    blaze::syev(U, evals, 'V', 'U');

    // diagonalize mat and exponentiate
    DMatrix diag(mat.rows(), mat.columns(), 0);
    blaze::diagonal(diag) = blaze::exp(blaze::diagonal(U * mat * blaze::trans(U)));
    // transform back to non-diagonal matrix
    return blaze::trans(U) * diag * U;
}

template <typename MT>
auto ilogdet(MT &matrix)
{
    static_assert(blaze::IsDenseMatrix<MT>::value, "logdet needs dense matrices");

    using ET = ValueType_t<typename MT::ElementType>;
    const std::size_t n = matrix.rows();
#ifndef NDEBUG
    if (n != matrix.columns())
        throw std::invalid_argument("Invalid non-square matrix provided");
#endif

    // pivot indices
    std::unique_ptr<int[]> ipiv = std::make_unique<int[]>(n);
    // perform LU decomposition (mat = PLU)
    blaze::getrf(matrix, ipiv.get());

    std::complex<ET> res = 0;
    bool negDetP = false;  // if true det(P) == -1, else det(P) == +1
    for (std::size_t i = 0; i < n; ++i) {
        // determinant of pivot matrix P
        if (ipiv[i] - 1 != blaze::numeric_cast<int>(i)) {
            negDetP = !negDetP;
        }
        // log det of U (diagonal elements)
        res += std::log(std::complex<ET>{ matrix(i, i) });
    }
    // combine log dets and project to (-pi, pi]
    return toFirstLogBranch(res + (negDetP ? std::complex<ET>{ 0, pi<ET> } : 0));
}

template <typename MT>
auto logdet(const MT &matrix)
{
    static_assert(blaze::IsDenseMatrix<MT>::value, "logdet needs dense matrices");

    using ET = ValueType_t<typename MT::ElementType>;
    MT mat{ matrix };  // need to copy here in order to disambiguate from overload for rvalues
    const auto n = mat.rows();
#ifndef NDEBUG
    if (n != mat.columns())
        throw std::invalid_argument("Invalid non-square matrix provided");
#endif

    // pivot indices
    auto ipiv = std::make_unique<int[]>(n);
    // perform LU-decomposition, afterwards matrix = PLU
    blaze::getrf(mat, ipiv.get());

    std::complex<ET> res = 0;
    bool negDetP = false;  // if true det(P) == -1, else det(P) == +1
    for (std::size_t i = 0; i < n; ++i) {
        // determinant of pivot matrix P
        if (ipiv[i] - 1 != blaze::numeric_cast<int>(i)) {
            negDetP = !negDetP;
        }
        // log det of U (diagonal elements)
        res += std::log(std::complex<ET>{ mat(i, i) });
    }
    // combine log dets and project to (-pi, pi]
    return toFirstLogBranch(res + (negDetP ? std::complex<ET>{ 0, pi<ET> } : 0));
}

#endif  // ndef MATH_HPP
