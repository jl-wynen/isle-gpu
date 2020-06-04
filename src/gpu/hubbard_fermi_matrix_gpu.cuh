#ifndef ISLE_HUBBARD_FERMI_MATRIX_GPU_CUH
#define ISLE_HUBBARD_FERMI_MATRIX_GPU_CUH

#include <complex>

#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <cublas_v2.h>

#include "species.hpp"


using CDDVector = thrust::device_vector<thrust::complex<double>>;


class HubbardFermiMatrixGPU
{
public:
    HubbardFermiMatrixGPU(double const *kappaTilde, size_t nx,
                          double muTilde, std::int8_t sigmaKappa);
    // Not sure how to handle cuBLAS handles, so better just delete all copy and move functions.
    HubbardFermiMatrixGPU(const HubbardFermiMatrixGPU &) = delete;
    HubbardFermiMatrixGPU &operator=(const HubbardFermiMatrixGPU &) = delete;
    HubbardFermiMatrixGPU(HubbardFermiMatrixGPU &&other) = delete;
    HubbardFermiMatrixGPU &operator=(HubbardFermiMatrixGPU &&other) = delete;
    ~HubbardFermiMatrixGPU();

    const thrust::device_vector<double> &expKappa(Species species, const bool inv) const;
    std::complex<double> logdetExpKappa(Species species, const bool inv) const;

    void F(CDDVector &f, std::size_t tp, const CDDVector &phi,
           Species species, bool inv = false) const;

    const thrust::device_vector<double> &kappaTilde() const noexcept;
    double muTilde() const noexcept;
    std::int8_t sigmaKappa() const noexcept;
    std::size_t nx() const noexcept;
    cublasHandle_t cublasHandle() const noexcept;

// private:
    cublasHandle_t _cublasHandle;

    thrust::device_vector<double> _kappa;
    size_t _nx;
    double _mu;
    std::int8_t _sigmaKappa;

    thrust::device_vector<double> _expKappap;
    thrust::device_vector<double> _expKappapInv;
    thrust::device_vector<double> _expKappah;
    thrust::device_vector<double> _expKappahInv;
    std::complex<double> _logdetExpKappahInv;
};

std::complex<double> logdetM(const HubbardFermiMatrixGPU &hfm,
                             const thrust::device_vector<thrust::complex<double>> &phi,
                             Species species);

#endif  // ISLE_HUBBARD_FERMI_MATRIX_GPU_CUH
