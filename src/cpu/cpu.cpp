#include "cpu.hpp"

#include "hubbardFermiMatrixExp.hpp"

namespace cpu {
std::array<std::complex<double>, 2> logdetM(const blaze::CompressedMatrix<double> &hopping,
                                            const blaze::DynamicVector<std::complex<double>> &phi,
                                            double const beta)
{

    size_t const nt = phi.size() / hopping.rows();

    HubbardFermiMatrixCPU hfm(hopping * beta / nt, 0, -1);
    return { logdetM(hfm, phi, Species::PARTICLE),
             logdetM(hfm, phi, Species::HOLE) };
}
}  // namespace cpu
