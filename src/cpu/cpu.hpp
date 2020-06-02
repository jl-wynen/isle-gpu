#ifndef ISLE_CPU_HPP
#define ISLE_CPU_HPP

#include <complex>
#include <array>

#include <blaze/Blaze.h>

namespace cpu {
std::array<std::complex<double>, 2> logdetM(const blaze::CompressedMatrix<double> &hopping,
                                            const blaze::DynamicVector<std::complex<double>> &phi,
                                            double beta);
}

#endif  // ndef ISLE_CPU_HPP
