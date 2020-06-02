#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include <blaze/Blaze.h>

#include "cpu/cpu.hpp"

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


int main()
{
    auto hopping = makeTriangle();
    constexpr size_t nt = 32;
    constexpr double U = 4;
    constexpr double beta = 6;

    blaze::DynamicVector<std::complex<double>> phi(hopping.rows()*nt, 0.0);

    auto cpuLDM = cpu::logdetM(hopping, phi, beta);
    std::cout << "CPU: " << cpuLDM[0] << ' ' << cpuLDM[1] << '\n';

}
