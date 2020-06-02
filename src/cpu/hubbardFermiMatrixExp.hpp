#ifndef HUBBARD_FERMI_MATRIX_EXP_HPP
#define HUBBARD_FERMI_MATRIX_EXP_HPP

#include <vector>

#include "math.hpp"
#include "species.hpp"

class HubbardFermiMatrixExp
{
public:
    HubbardFermiMatrixExp(const SparseMatrix<double> &kappaTilde,
                          double muTilde, std::int8_t sigmaKappa);
    HubbardFermiMatrixExp(const HubbardFermiMatrixExp &) = default;
    HubbardFermiMatrixExp &operator=(const HubbardFermiMatrixExp &) = default;
    HubbardFermiMatrixExp(HubbardFermiMatrixExp &&) = default;
    HubbardFermiMatrixExp &operator=(HubbardFermiMatrixExp &&) = default;
    ~HubbardFermiMatrixExp() = default;

    const DMatrix &expKappa(Species species, const bool inv) const;
    std::complex<double> logdetExpKappa(Species species, const bool inv) const;

    void K(DSparseMatrix &k, Species species) const;
    DMatrix Kinv(Species species) const;
    std::complex<double> logdetKinv(Species species) const;

    void F(CDMatrix &f, std::size_t tp, const CDVector &phi,
           Species species, bool inv = false) const;
    CDMatrix F(std::size_t tp, const CDVector &phi,
               Species species, bool inv) const;

    const DSparseMatrix &kappaTilde() const noexcept;
    double muTilde() const noexcept;
    std::int8_t sigmaKappa() const noexcept;
    std::size_t nx() const noexcept;

private:
    DSparseMatrix _kappa;
    double _mu;
    std::int8_t _sigmaKappa;

    DMatrix _expKappap;
    DMatrix _expKappapInv;
    DMatrix _expKappah;
    DMatrix _expKappahInv;
    std::complex<double> _logdetExpKappahInv;
};

std::complex<double> logdetM(const HubbardFermiMatrixExp &hfm, const CDVector &phi,
                             Species species);

CDMatrix solveM(const HubbardFermiMatrixExp &hfm, const CDVector &phi,
                Species species, const CDMatrix &rhss);

#endif  // ndef HUBBARD_FERMI_MATRIX_HPP
