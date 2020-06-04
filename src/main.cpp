#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include <blaze/Blaze.h>

#include "cpu/cpu.hpp"
#include "gpu/gpu.cuh"

auto makeTriangle()
{
    blaze::CompressedMatrix<double> hopping(3, 3);
    hopping.set(0, 1, 1.0);
    hopping.set(1, 0, 1.0);
    hopping.set(0, 2, 1.0);
    hopping.set(2, 0, 1.0);
    hopping.set(1, 2, 1.0);
    hopping.set(2, 1, 1.0);
    return hopping;
}

template <typename T>
auto flatMatrix(blaze::DynamicMatrix<T, blaze::columnMajor> const &matrix)
{
    std::vector<T> flat(matrix.rows() * matrix.columns());
    for (std::size_t i = 0; i < matrix.rows(); ++i) {
        for (std::size_t j = 0; j < matrix.columns(); ++j) {
            flat[j * matrix.rows() + i] = matrix(i, j);
        }
    }
    return flat;
}


int main()
{
    auto hopping = makeTriangle();
    constexpr size_t nt = 64;
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
