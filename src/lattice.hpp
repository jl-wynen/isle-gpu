#ifndef ISLE_LATTICE_HPP
#define ISLE_LATTICE_HPP

#include <blaze/math/CompressedMatrix.h>

using HoppingMatrix = blaze::CompressedMatrix<double>;

HoppingMatrix makeTriangle();

#endif  // ndef ISLE_LATTICE_HPP
