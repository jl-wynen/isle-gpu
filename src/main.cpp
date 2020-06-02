#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "cpu/math.hpp"
#include "cpu/hubbardFermiMatrixExp.hpp"

DSparseMatrix makeTriangle()
{
    DSparseMatrix hopping(3, 3);
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

    HubbardFermiMatrixCPU hfm(hopping * beta / nt, 0, -1);

    CDVector phi(hopping.rows() * nt);
    std::cout << logdetM(hfm, phi, Species::PARTICLE);
}
