#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include <blaze/Blaze.h>

#include "lattice.hpp"
#include "cpu/cpu.hpp"
#include "gpu/gpu.cuh"


int main()
{
    auto hopping = makeTriangle();
    constexpr size_t nt = 4;
    constexpr double U = 4;
    constexpr double beta = 6;

    blaze::DynamicVector<std::complex<double>> phi(hopping.rows() * nt, 1.0);
    std::mt19937 rng(12456);
    std::normal_distribution<double> dist(0, sqrt(U * beta / nt));
    std::generate(phi.begin(), phi.end(), [&]() mutable { return dist(rng); });

    auto cpuLDM = cpu::logdetM(hopping, phi, beta);
    std::cout << "CPU: " << cpuLDM[0] << ' ' << cpuLDM[1] << '\n';


    auto gpuLDM = gpu::logdetM(flatMatrix(blaze::DynamicMatrix<double, blaze::columnMajor>(hopping * beta / nt)).data(),
                               phi.data(),
                               hopping.rows(),
                               nt,
                               beta);
    std::cout << "GPU: " << gpuLDM[0] << ' ' << gpuLDM[1] << '\n';
}
