#include "math.hpp"

#include <memory>

#include <blaze/Blaze.h>

std::complex<double> toFirstLogBranch(const std::complex<double> &x)
{
    return { std::real(x), std::remainder(std::imag(x), 2 * pi<double>) };
}

void expmSym(std::vector<double> &mat, size_t const nx)
{
    // compute eigenvalues evals and eigenvectors U
    blaze::DynamicMatrix<double> matrix(nx, nx, mat.data());
    blaze::DynamicMatrix<double> U = matrix;
    blaze::DynamicVector<double> evals;
    blaze::syev(U, evals, 'V', 'U');

    // diagonalize mat and exponentiate
    blaze::DynamicMatrix<double> diag(nx, nx, 0);
    blaze::diagonal(diag) = blaze::exp(blaze::diagonal(U * matrix * blaze::trans(U)));
    // transform back to non-diagonal matrix
    matrix = blaze::trans(U) * diag * U;

    for (size_t i = 0; i < matrix.rows(); ++i) {
        for (std::size_t j = 0; j < matrix.columns(); ++j) {
            mat[j * nx + i] = matrix(i, j);
        }
    }
}

std::complex<double> logdet(const double *matrix, size_t const nx)
{
    blaze::DynamicMatrix<double, blaze::columnMajor> mat(nx, nx, matrix);

    // pivot indices
    auto ipiv = std::make_unique<int[]>(nx);
    // perform LU-decomposition, afterwards matrix = PLU
    blaze::getrf(mat, ipiv.get());

    std::complex<double> res = 0;
    bool negDetP = false;  // if true det(P) == -1, else det(P) == +1
    for (std::size_t i = 0; i < nx; ++i) {
        // determinant of pivot matrix P
        if (ipiv[i] - 1 != blaze::numeric_cast<int>(i)) {
            negDetP = !negDetP;
        }
        // log det of U (diagonal elements)
        res += std::log(std::complex<double>{ mat(i, i) });
    }
    // combine log dets and project to (-pi, pi]
    return toFirstLogBranch(res + (negDetP ? std::complex<double>{ 0, pi<double> } : 0));
}


std::complex<double> logdet(const std::complex<double> *matrix, size_t const nx)
{
    blaze::DynamicMatrix<std::complex<double>, blaze::columnMajor> mat(nx, nx, matrix);

    // pivot indices
    auto ipiv = std::make_unique<int[]>(nx);
    // perform LU-decomposition, afterwards matrix = PLU
    blaze::getrf(mat, ipiv.get());

    std::complex<double> res = 0;
    bool negDetP = false;  // if true det(P) == -1, else det(P) == +1
    for (std::size_t i = 0; i < nx; ++i) {
        // determinant of pivot matrix P
        if (ipiv[i] - 1 != blaze::numeric_cast<int>(i)) {
            negDetP = !negDetP;
        }
        // log det of U (diagonal elements)
        res += std::log(std::complex<double>{ mat(i, i) });
    }
    // combine log dets and project to (-pi, pi]
    return toFirstLogBranch(res + (negDetP ? std::complex<double>{ 0, pi<double> } : 0));
}
