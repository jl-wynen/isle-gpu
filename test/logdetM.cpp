#include "catch.hpp"

#include <vector>

#include "lattice.hpp"
#include "util.hpp"
#include "cpu/cpu.hpp"
#include "gpu/gpu.cuh"

namespace {
/// Number of repetitions.
constexpr size_t NREP = 10;

/// Construct all lattices (hopping matrices) for testing.
auto makeLattices()
{
    return std::vector<HoppingMatrix>{
        makeTriangle()
    };
}
}  // namespace

TEST_CASE("Compare logdetM on CPU and GPU", "[logdetM]")
{
    auto const &hopping = GENERATE_REF(from_range(makeLattices()));
    size_t const nt = GENERATE(1u, 2u, 4u, 5u, 8u, 64u, 128u);
    double const beta = GENERATE(6.0, 8.0);
    double const phiWidth = GENERATE(take(NREP, random(0.01, 1.0)));
    CAPTURE(hopping.rows(), nt, beta, phiWidth);

    auto const phi = makeRandomConfig(0, randomWidth(rng),
                                      hopping.rows() * nt, rng);

    auto const cpuLDM = cpu::logdetM(hopping, phi, beta);
    auto const gpuLDM = gpu::logdetM(flatHoppingMatrix(hopping * beta / nt).data(),
                                     phi.data(),
                                     hopping.rows(),
                                     nt,
                                     beta);

    SECTION("Particles")
    {
        REQUIRE(cpuLDM[0].real() == Approx(gpuLDM[0].real()));
        REQUIRE(cpuLDM[0].imag() == Approx(gpuLDM[0].imag()));
    }
    SECTION("Holes")
    {
//        REQUIRE(cpuLDM[1].real() == Approx(gpuLDM[1].real()));
//        REQUIRE(cpuLDM[1].imag() == Approx(gpuLDM[1].imag()));
    }
}
