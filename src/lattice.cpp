#include "lattice.hpp"

#include <utility>

namespace {
using Link = std::pair<std::size_t, std::size_t>;

template <size_t N>
using Graph = std::array<Link, N>;

constexpr Graph<1> TWO_SITES{ { { 0, 1 } } };
constexpr Graph<3> TRIANGLE{ { { 0, 1 },
                               { 0, 2 },
                               { 1, 2 } } };
constexpr Graph<6> TETRAHEDRON{ { { 0, 1 },
                                  { 0, 2 },
                                  { 0, 3 },
                                  { 1, 2 },
                                  { 1, 3 },
                                  { 2, 3 } } };
constexpr Graph<5> PENTAGON{ { { 0, 1 },
                               { 1, 2 },
                               { 2, 3 },
                               { 3, 4 },
                               { 4, 0 } } };
constexpr Graph<18> TUBE_33_1{ { { 0, 1 },
                                 { 0, 11 },
                                 { 0, 11 },
                                 { 1, 2 },
                                 { 1, 2 },
                                 { 2, 3 },
                                 { 3, 4 },
                                 { 3, 4 },
                                 { 4, 5 },
                                 { 5, 6 },
                                 { 5, 6 },
                                 { 6, 7 },
                                 { 7, 8 },
                                 { 7, 8 },
                                 { 8, 9 },
                                 { 9, 10 },
                                 { 9, 10 },
                                 { 10, 11 } } };


template <size_t N>
constexpr std::size_t computeNumSites(Graph<N> const &graph)
{
    std::size_t nsites = 0;
    for (Link const &link : graph) {
        if (link.first + 1 > nsites) {
            nsites = link.first + 1;
        }
        if (link.second + 1 > nsites) {
            nsites = link.second + 1;
        }
    }
    return nsites;
}

void setNeighbours(HoppingMatrix &mat, size_t const i, size_t const j, double const kappa = 1.0)
{
    mat.set(i, j, kappa);
    mat.set(j, i, kappa);
}

template <size_t N>
HoppingMatrix buildHoppingMatrix(Graph<N> const &graph)
{
    auto const nsites = computeNumSites(graph);
    HoppingMatrix hopping(nsites, nsites);
    for (Link const &link : graph) {
        setNeighbours(hopping, link.first, link.second);
    }
    return hopping;
}
}  // namespace


HoppingMatrix makeHopping(std::string const &name)
{
    if (name == "two_sites") {
        return buildHoppingMatrix(TWO_SITES);
    }
    if (name == "triangle") {
        return buildHoppingMatrix(TRIANGLE);
    }
    if (name == "tetrahedron") {
        return buildHoppingMatrix(TETRAHEDRON);
    }
    if (name == "pentagon") {
        return buildHoppingMatrix(PENTAGON);
    }
    if (name == "tube_33_1") {
        return buildHoppingMatrix(TUBE_33_1);
    }
    throw std::invalid_argument("Unknown lattice: " + name);
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
