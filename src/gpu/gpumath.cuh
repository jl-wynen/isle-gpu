#ifndef ISLE_GPUMATH_CUH
#define ISLE_GPUMATH_CUH

#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <cublas_v2.h>

thrust::complex<double> ilogdet(thrust::device_vector<thrust::complex<double>> mat,
                                int const n,
                                cublasHandle_t handle);

#endif  // ndef ISLE_GPUMATH_HPP
