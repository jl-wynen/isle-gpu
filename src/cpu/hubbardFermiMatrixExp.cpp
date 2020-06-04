#include "hubbardFermiMatrixExp.hpp"

#include <memory>
#include <limits>
#include <cmath>

using namespace std::complex_literals;

namespace {
template <typename MT>
void resizeMatrix(MT &mat, const std::size_t target)
{
#ifndef NDEBUG
    if (mat.rows() != mat.columns()) {
        throw std::invalid_argument("Matrix is not square.");
    }
#endif
    if (mat.rows() != target) {
        mat.resize(target, target, false);
    }
}

auto computeExponential(const DSparseMatrix &kappa,
                        const double mu,
                        const std::int8_t sigmaKappa,
                        const Species species,
                        const bool inv)
{
    switch (species) {
    case Species::PARTICLE:
        if (inv) {
            return expmSym(-kappa + mu * IdMatrix<double>(kappa.rows()));
        } else {
            return expmSym(kappa - mu * IdMatrix<double>(kappa.rows()));
        }
    case Species::HOLE:
        if (inv) {
            return expmSym(-sigmaKappa * kappa - mu * IdMatrix<double>(kappa.rows()));
        } else {
            return expmSym(sigmaKappa * kappa + mu * IdMatrix<double>(kappa.rows()));
        }
    }
    // Strictly speaking impossible to reach but gcc complains.
    throw std::invalid_argument("Unknown species");
}
}  // namespace

HubbardFermiMatrixCPU::HubbardFermiMatrixCPU(const DSparseMatrix &kappaTilde,
                                             const double muTilde,
                                             const std::int8_t sigmaKappa)
  : _kappa{ kappaTilde }, _mu{ muTilde }, _sigmaKappa{ sigmaKappa },
    _expKappap{ computeExponential(kappaTilde, muTilde, sigmaKappa, Species::PARTICLE, false) },
    _expKappapInv{ computeExponential(kappaTilde, muTilde, sigmaKappa, Species::PARTICLE, true) },
    _expKappah{ computeExponential(kappaTilde, muTilde, sigmaKappa, Species::HOLE, false) },
    _expKappahInv{ computeExponential(kappaTilde, muTilde, sigmaKappa, Species::HOLE, true) },
    _logdetExpKappahInv{ logdet(_expKappahInv) }
{
    if (kappaTilde.rows() != kappaTilde.columns()) {
        throw std::invalid_argument("Hopping matrix is not square.");
    }
    if (muTilde != 0.0) {
        throw std::invalid_argument("mu must be zero");
    }
}

const DMatrix &HubbardFermiMatrixCPU::expKappa(const Species species, const bool inv) const
{
    switch (species) {
    case Species::PARTICLE:
        if (inv) {
            return _expKappapInv;
        } else {
            return _expKappap;
        }
    case Species::HOLE:
        if (inv) {
            return _expKappahInv;
        } else {
            return _expKappah;
        }
    }
    // Strictly speaking impossible to reach but gcc complains.
    throw std::invalid_argument("Unknown species");
}

std::complex<double>
  HubbardFermiMatrixCPU::logdetExpKappa(const Species species,
                                        const bool inv) const
{
    if (species == Species::HOLE && inv) {
        return _logdetExpKappahInv;
    }
    throw std::runtime_error("logdetExpKappa is only implemented for holes and inv=true");
}

void HubbardFermiMatrixCPU::F(CDMatrix &f,
                              const std::size_t tp, const CDVector &phi,
                              const Species species, const bool inv) const
{
    const std::size_t NX = nx();
    const std::size_t NT = getNt(phi, NX);
    const std::size_t tm1 = tp == 0 ? NT - 1 : tp - 1;  // t' - 1
    resizeMatrix(f, NX);

    // the sign in the exponential of phi
    auto const sign = ((species == Species::PARTICLE && !inv)
                       || (species == Species::HOLE && inv))
                        ? +1.0i
                        : -1.0i;

    if (inv) {
        // f = e^phi * e^kappa  (up to signs in exponents)
        f = expand(exp(sign * spacevec(phi, tm1, NX)), NX)
            % expKappa(species, inv);
    } else {
        // f = e^kappa * e^phi  (up to signs in exponents)
        f = expKappa(species, inv)
            % expand(trans(exp(sign * spacevec(phi, tm1, NX))), NX);
    }
}

CDMatrix HubbardFermiMatrixCPU::F(const std::size_t tp, const CDVector &phi,
                                  const Species species, const bool inv) const
{
    CDMatrix f;
    F(f, tp, phi, species, inv);
    return f;
}

const SparseMatrix<double> &HubbardFermiMatrixCPU::kappaTilde() const noexcept
{
    return _kappa;
}

double HubbardFermiMatrixCPU::muTilde() const noexcept
{
    return _mu;
}

std::int8_t HubbardFermiMatrixCPU::sigmaKappa() const noexcept
{
    return _sigmaKappa;
}

std::size_t HubbardFermiMatrixCPU::nx() const noexcept
{
    return _kappa.rows();
}

namespace {
// Use version log(det(1+hat{A})).
std::complex<double> logdetM_p(const HubbardFermiMatrixCPU &hfm,
                               const CDVector &phi)
{
    const auto NX = hfm.nx();
    const auto NT = getNt(phi, NX);
    const auto species = Species::PARTICLE;

    // first factor F
    CDMatrix f;
    CDMatrix B = hfm.F(0, phi, species, false);
    // other factors
    for (std::size_t t = 1; t < NT; ++t) {
        hfm.F(f, t, phi, species, false);
        B = f * B;
    }

    B += IdMatrix<std::complex<double>>(NX);
    return toFirstLogBranch(ilogdet(B));
}

// Use version -i Phi - N_t log(det(e^{-sigmaKappa*kappa-mu})) + log(det(1+hat{A}^{-1})).
std::complex<double> logdetM_h(const HubbardFermiMatrixCPU &hfm,
                               const CDVector &phi)
{
    const auto NX = hfm.nx();
    const auto NT = getNt(phi, NX);

    // build product of F^{-1}
    auto f = hfm.F(0, phi, Species::HOLE, true);
    CDMatrix aux = f;  // the matrix under the determinant
    for (std::size_t t = 1; t < NT; ++t) {
        hfm.F(f, t, phi, Species::HOLE, true);
        aux = aux * f;
    }
    aux += IdMatrix<std::complex<double>>(NX);

    // add Phi and return
    return toFirstLogBranch(-static_cast<double>(NT) * hfm.logdetExpKappa(Species::HOLE, true)
                            - 1.0i * blaze::sum(phi)
                            + ilogdet(aux));
}
}  // namespace

std::complex<double> logdetM(const HubbardFermiMatrixCPU &hfm,
                             const CDVector &phi, const Species species)
{
    if (hfm.muTilde() != 0) {
        throw std::runtime_error("Called logdetM with mu != 0. This is not supported yet because the algorithm is unstable.");
    }

    switch (species) {
    case Species::PARTICLE:
        return logdetM_p(hfm, phi);
    case Species::HOLE:
        return logdetM_h(hfm, phi);
    }
    // Strictly speaking impossible to reach but gcc complains.
    throw std::invalid_argument("Unknown species");
}

CDMatrix solveM(const HubbardFermiMatrixCPU &hfm, const CDVector &phi,
                const Species species, const CDMatrix &rhss)
{
    if (hfm.muTilde() != 0) {
        throw std::runtime_error("Exponential hopping is not supported for mu != 0");
    }

    const std::size_t NX = hfm.nx();
    const std::size_t NT = getNt(phi, NX);
    const std::size_t NRHS = rhss.rows();
    // the results (vectors x in the end, z at intermediate stage)
    CDMatrix res(rhss.rows(), rhss.columns());

    // construct all partial A^{-1} and the complete one
    std::vector<CDMatrix> partialAinv;
    partialAinv.reserve(NT);
    partialAinv.emplace_back(hfm.F(0, phi, species, true));
    for (std::size_t t = 1; t < NT; ++t) {
        partialAinv.emplace_back(partialAinv[t - 1] * hfm.F(t, phi, species, true));
    }

    // calculate all z's and store in res
    blaze::submatrix(res, 0, 0, NRHS, NX) = blaze::submatrix(rhss, 0, 0, NRHS, NX) * blaze::trans(partialAinv[0]);
    for (std::size_t t = 1; t < NT; ++t) {
        blaze::submatrix(res, 0, t * NX, NRHS, NX) = blaze::submatrix(rhss, 0, t * NX, NRHS, NX) * blaze::trans(partialAinv[t])
                                                     + blaze::submatrix(res, 0, (t - 1) * NX, NRHS, NX);
    }
    // now res = z

    // LU-decompose all partial A^{-1} in place
    std::vector<int> ipiv(NX * NT);  // all pivots for all matrices in time-major order
    for (std::size_t t = 0; t < NT - 1; ++t) {
        blaze::getrf(partialAinv[t], &ipiv[t * NX]);
    }
    // NT-1 is special
    partialAinv[NT - 1] += IdMatrix<std::complex<double>>(NX);
    blaze::getrf(partialAinv[NT - 1], &ipiv[(NT - 1) * NX]);

    // solve for x
    CDMatrix matLast = blaze::submatrix(res, 0, (NT - 1) * NX, NRHS, NX);
    // transpose because LAPACK wants column-major layout
    blaze::getrs(partialAinv[NT - 1], matLast, 'T', &ipiv[(NT - 1) * NX]);
    blaze::submatrix(res, 0, (NT - 1) * NX, NRHS, NX) = matLast;
    for (std::size_t t = 0; t < NT - 1; ++t) {
        CDMatrix mat = blaze::submatrix(res, 0, t * NX, NRHS, NX) - matLast;
        blaze::getrs(partialAinv[t], mat, 'T', &ipiv[t * NX]);
        blaze::submatrix(res, 0, t * NX, NRHS, NX) = mat;
    }

    return res;
}
