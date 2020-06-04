#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include <blaze/Blaze.h>

#include "lattice.hpp"
#include "util.hpp"
#include "cpu/cpu.hpp"
#include "gpu/gpu.cuh"


int main()
{
    auto hopping = makeTriangle();
    constexpr size_t nt = 4;
    constexpr double U = 4;
    constexpr double beta = 6;

    std::mt19937 rng(12456);
    auto phi = makeRandomConfig(0, sqrt(U * beta / nt),
                                hopping.rows() * nt, rng);

    auto cpuLDM = cpu::logdetM(hopping, phi, beta);
    std::cout << "CPU: " << cpuLDM[0] << ' ' << cpuLDM[1] << '\n';


    auto gpuLDM = gpu::logdetM(flatHoppingMatrix(hopping * beta / nt).data(),
                               phi.data(),
                               hopping.rows(),
                               nt,
                               beta);
    std::cout << "GPU: " << gpuLDM[0] << ' ' << gpuLDM[1] << '\n';
}
