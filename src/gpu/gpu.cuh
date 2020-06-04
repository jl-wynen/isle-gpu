#ifndef ISLE_GPU_CUH
#define ISLE_GPU_CUH

#include <complex>
#include <array>

namespace gpu {
std::array<std::complex<double>, 2> logdetM(double const *hopping,
                                            std::complex<double> const *phi,
                                            size_t nx,
                                            size_t nt,
                                            double beta);
}

#endif  // ndef ISLE_GPU_HPP
