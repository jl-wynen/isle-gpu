#include "gpu.cuh"

#include <thrust/complex.h>
#include <thrust/device_vector.h>

#include "hubbard_fermi_matrix_gpu.cuh"

namespace gpu {
std::array<std::complex<double>, 2> logdetM(double const *kappaTilde,
                                            std::complex<double> const *phi,
                                            size_t nx,
                                            size_t nt,
                                            double const beta)
{
    HubbardFermiMatrixGPU hfm(kappaTilde, nx, 0.0, -1);

    thrust::device_vector<thrust::complex<double>> devicePhi(nx * nt);
    auto result = cudaMemcpy(thrust::raw_pointer_cast(devicePhi.data()),
                             phi,
                             nx * nt * sizeof(phi[0]),
                             cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        throw std::runtime_error("Failed to copy phi to device: " + std::to_string(result));
    }

    return { logdetM(hfm, devicePhi, Species::PARTICLE),
             logdetM(hfm, devicePhi, Species::HOLE) };
}

}  // namespace gpu
