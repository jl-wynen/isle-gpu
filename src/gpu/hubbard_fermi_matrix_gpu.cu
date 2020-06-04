#include "hubbard_fermi_matrix_gpu.cuh"

#include <cuComplex.h>
#include <thrust/fill.h>

#include "math.hpp"

using namespace std::complex_literals;

namespace {
template <typename VT>
std::size_t getNt(const VT &stVec, const std::size_t nx)
{
#ifndef NDEBUG
    if (stVec.size() % nx != 0) {
        throw std::runtime_error("Vector dimension does not match, expected a spacetime vector.");
    }
#endif
    return stVec.size() / nx;
}

auto computeExponential(double const *kappaTilde,
                        const std::size_t nx,
                        const std::int8_t sigmaKappa,
                        const Species species,
                        const bool inv)
{
    double sign = +1.0;
    switch (species) {
    case Species::PARTICLE:
        if (inv) {
            sign *= -1.0;
        }
        break;
    case Species::HOLE:
        if (inv) {
            sign *= -sigmaKappa;
        } else {
            sign *= sigmaKappa;
        }
        break;
    }
    std::vector<double> signedKappa(kappaTilde, kappaTilde + (nx * nx));
    for (double &x : signedKappa) {
        x *= sign;
    }

    expmSym(signedKappa, nx);
    return signedKappa;
}

void setCUDAMatrix(double const *hostMatrix, double *deviceMatrix, size_t nrows, size_t ncolumns)
{
    auto result = cublasSetMatrix(nrows, ncolumns, sizeof(double),
                                  hostMatrix, ncolumns,
                                  deviceMatrix, ncolumns);
    if (result != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to copy matrix to device: " + std::to_string(result));
    }
}
}  // namespace

HubbardFermiMatrixGPU::HubbardFermiMatrixGPU(double const *kappaTilde, const size_t nx,
                                             const double muTilde, const std::int8_t sigmaKappa)
  : _kappa(nx * nx), _nx(nx), _mu(muTilde), _sigmaKappa(sigmaKappa),
    _expKappap(nx * nx),
    _expKappapInv(nx * nx),
    _expKappah(nx * nx),
    _expKappahInv(nx * nx)
{
    if (muTilde != 0.0) {
        throw std::invalid_argument("mu must be zero");
    }

    auto result = cublasCreate(&_cublasHandle);
    if (result != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to create cuBLAS handle: " + std::to_string(result));
    }

    setCUDAMatrix(kappaTilde, thrust::raw_pointer_cast(_kappa.data()), nx, nx);

    std::vector<double> aux = computeExponential(kappaTilde, nx, sigmaKappa, Species::PARTICLE, false);
    setCUDAMatrix(aux.data(), thrust::raw_pointer_cast(_expKappap.data()), nx, nx);
    aux = computeExponential(kappaTilde, nx, sigmaKappa, Species::PARTICLE, true);
    setCUDAMatrix(aux.data(), thrust::raw_pointer_cast(_expKappapInv.data()), nx, nx);
    aux = computeExponential(kappaTilde, nx, sigmaKappa, Species::HOLE, false);
    setCUDAMatrix(aux.data(), thrust::raw_pointer_cast(_expKappah.data()), nx, nx);
    aux = computeExponential(kappaTilde, nx, sigmaKappa, Species::HOLE, true);
    setCUDAMatrix(aux.data(), thrust::raw_pointer_cast(_expKappahInv.data()), nx, nx);
    _logdetExpKappahInv = logdet(aux.data(), nx);
}

HubbardFermiMatrixGPU::~HubbardFermiMatrixGPU()
{
    auto result = cublasDestroy(_cublasHandle);
    if (result != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to destroy cuBLAS handle: " + std::to_string(result));
    }
}

const thrust::device_vector<double> &HubbardFermiMatrixGPU::expKappa(const Species species, const bool inv) const
{
    switch (species) {
    case Species::PARTICLE:
        if (inv) {
            return _expKappapInv;
        } else {
            return _expKappap;
        }
    case Species::HOLE:
        if (inv) {
            return _expKappahInv;
        } else {
            return _expKappah;
        }
    }
    // Strictly speaking impossible to reach but gcc complains.
    throw std::invalid_argument("Unknown species");
}

std::complex<double>
  HubbardFermiMatrixGPU::logdetExpKappa(const Species species,
                                        const bool inv) const
{
    if (species == Species::HOLE && inv) {
        return _logdetExpKappahInv;
    }
    throw std::runtime_error("logdetExpKappa is only implemented for holes and inv=true");
}

const thrust::device_vector<double> &HubbardFermiMatrixGPU::kappaTilde() const noexcept
{
    return _kappa;
}

double HubbardFermiMatrixGPU::muTilde() const noexcept
{
    return _mu;
}

std::int8_t HubbardFermiMatrixGPU::sigmaKappa() const noexcept
{
    return _sigmaKappa;
}

std::size_t HubbardFermiMatrixGPU::nx() const noexcept
{
    return _nx;
}

cublasHandle_t HubbardFermiMatrixGPU::cublasHandle() const noexcept
{
    return _cublasHandle;
}

__device__ __forceinline__ thrust::complex<double> cexp(thrust::complex<double> const z)
{
    double t = exp(z.real());
    double real, imag;
    sincos(z.imag(), &imag, &real);
    real *= t;
    imag *= t;
    return thrust::complex<double>(real, imag);
}

__global__ void constructF(thrust::complex<double> *f,
                           thrust::complex<double> const *phi,
                           double const *expKappa,
                           int nx,
                           thrust::complex<double> sign,
                           bool inv)
{
    if (inv) {
        // f = e^phi * e^kappa  (up to signs in exponents)
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < nx; ++j) {
                f[j * nx + i] = cexp(sign * phi[i]) * expKappa[j * nx + i];
            }
        }
    } else {
        // f = e^kappa * e^phi  (up to signs in exponents)
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < nx; ++j) {
                f[j * nx + i] = cexp(sign * phi[j]) * expKappa[j * nx + i];
            }
        }
    }
}

void HubbardFermiMatrixGPU::F(CDDVector &f, std::size_t tp, const CDDVector &phi,
                              Species species, bool inv) const
{
    const std::size_t NX = nx();
    const std::size_t NT = getNt(phi, NX);
    const std::size_t tm1 = tp == 0 ? NT - 1 : tp - 1;  // t' - 1

    // the sign in the exponential of phi
    auto const sign = ((species == Species::PARTICLE && !inv)
                       || (species == Species::HOLE && inv))
                        ? +1.0i
                        : -1.0i;
    constructF<<<1, 1>>>(thrust::raw_pointer_cast(f.data()),
                         thrust::raw_pointer_cast(phi.data()) + (tm1 * NX),
                         thrust::raw_pointer_cast(expKappa(species, inv).data()),
                         NX,
                         sign,
                         inv);
}

__global__ void addIdentiy(thrust::complex<double> *matrix, int n)
{
    for (int i = 0; i < n; ++i) {
        matrix[i * n + i] += 1.0;
    }
}

namespace {
// Use version log(det(1+hat{A})).
std::complex<double> logdetM_p(const HubbardFermiMatrixGPU &hfm,
                               const CDDVector &phi)
{
    const auto NX = hfm.nx();
    const auto NT = getNt(phi, NX);
    const auto species = Species::PARTICLE;

    // first factor F
    CDDVector B(NX * NX);
    CDDVector F(NX * NX);
    hfm.F(B, 0, phi, species, false);
    cuDoubleComplex one{ 1.0, 0.0 };

    // other factors
    CDDVector aux(NX * NX);
    for (std::size_t t = 1; t < NT; ++t) {
        // B = f * B;
        thrust::fill(aux.begin(), aux.end(), 0.0);
        hfm.F(F, t, phi, species, false);
        auto result = cublasZgemm(hfm.cublasHandle(),
                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                  NX, NX, NX,
                                  &one,
                                  reinterpret_cast<cuDoubleComplex *>(thrust::raw_pointer_cast(F.data())), NX,
                                  reinterpret_cast<cuDoubleComplex *>(thrust::raw_pointer_cast(B.data())), NX,
                                  &one,
                                  reinterpret_cast<cuDoubleComplex *>(thrust::raw_pointer_cast(aux.data())), NX);
        if (result != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Multiply f * B: " + std::to_string(result));
        }
        swap(aux, B);
    }

    addIdentiy<<<1, 1>>>(thrust::raw_pointer_cast(B.data()), NX);

    thrust::host_vector<thrust::complex<double>> hB = B;
    return toFirstLogBranch(logdet(reinterpret_cast<std::complex<double> *>(thrust::raw_pointer_cast(hB.data())),
                                   NX));
}

//// Use version -i Phi - N_t log(det(e^{-sigmaKappa*kappa-mu})) + log(det(1+hat{A}^{-1})).
//std::complex<double> logdetM_h(const HubbardFermiMatrixCPU &hfm,
//                               const CDVector &phi)
//{
//    const auto NX = hfm.nx();
//    const auto NT = getNt(phi, NX);
//
//    // build product of F^{-1}
//    auto f = hfm.F(0, phi, Species::HOLE, true);
//    CDMatrix aux = f;  // the matrix under the determinant
//    for (std::size_t t = 1; t < NT; ++t) {
//        hfm.F(f, t, phi, Species::HOLE, true);
//        aux = aux * f;
//    }
//    aux += IdMatrix<std::complex<double>>(NX);
//
//    // add Phi and return
//    return toFirstLogBranch(-static_cast<double>(NT) * hfm.logdetExpKappa(Species::HOLE, true)
//                            - 1.0i * blaze::sum(phi)
//                            + ilogdet(aux));
//}

}  // namespace


std::complex<double> logdetM(const HubbardFermiMatrixGPU &hfm,
                             const thrust::device_vector<thrust::complex<double>> &phi,
                             Species species)
{
    if (hfm.muTilde() != 0) {
        throw std::runtime_error("Called logdetM with mu != 0. This is not supported yet because the algorithm is unstable.");
    }

    switch (species) {
    case Species::PARTICLE:
        //return 0.0;
        return logdetM_p(hfm, phi);
    case Species::HOLE:
        return 0.0;
        //      return logdetM_h(hfm, phi);
    }
    // Strictly speaking impossible to reach but gcc complains.
    throw std::invalid_argument("Unknown species");
}
