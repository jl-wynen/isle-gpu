#ifndef ISLE_LATTICE_HPP
#define ISLE_LATTICE_HPP

#include <string>
#include <vector>

#include <blaze/math/CompressedMatrix.h>

using HoppingMatrix = blaze::CompressedMatrix<double>;

HoppingMatrix makeHopping(std::string const &name);



/// Return a matrix as a flat std::vector in column-major layout.
std::vector<double> flatHoppingMatrix(HoppingMatrix const &matrix);

#endif  // ndef ISLE_LATTICE_HPP
