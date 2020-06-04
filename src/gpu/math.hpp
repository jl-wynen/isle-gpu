#ifndef ISLE_MATH_HPP
#define ISLE_MATH_HPP

#include <complex>
#include <vector>

std::complex<double> toFirstLogBranch(const std::complex<double> &x);
void expmSym(std::vector<double> &mat, size_t nx);
std::complex<double> logdet(const double *mat, size_t const nx);
std::complex<double> logdet(const std::complex<double> *mat, size_t const nx);

#endif  // ndef ISLE_MATH_HPP
