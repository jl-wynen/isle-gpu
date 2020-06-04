#include "lattice.hpp"


HoppingMatrix makeTriangle()
{
    HoppingMatrix hopping(3, 3);
    hopping.set(0, 1, 1.0);
    hopping.set(1, 0, 1.0);
    hopping.set(0, 2, 1.0);
    hopping.set(2, 0, 1.0);
    hopping.set(1, 2, 1.0);
    hopping.set(2, 1, 1.0);
    return hopping;
}


std::vector<double> flatHoppingMatrix(HoppingMatrix const &matrix)
{
    std::vector<double> flat(matrix.rows() * matrix.columns());
    for (std::size_t i = 0; i < matrix.rows(); ++i) {
        for (std::size_t j = 0; j < matrix.columns(); ++j) {
            flat[j * matrix.rows() + i] = matrix(i, j);
        }
    }
    return flat;
}
