#ifndef ISLE_MATH_HPP
#define ISLE_MATH_HPP

#include <complex>
#include <vector>

template <typename T>
constexpr T pi = static_cast<T>(3.1415926535897932384626433832795028841971693993751058209749L);

std::complex<double> toFirstLogBranch(const std::complex<double> &x);
void expmSym(std::vector<double> &mat, size_t nx);
std::complex<double> logdet(const double *mat, size_t const nx);
std::complex<double> logdet(const std::complex<double> *mat, size_t const nx);

#endif  // ndef ISLE_MATH_HPP
