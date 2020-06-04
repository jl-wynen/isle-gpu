#ifndef ISLE_UTIL_HPP
#define ISLE_UTIL_HPP

#include <complex>
#include <random>

#include <blaze/math/DynamicVector.h>

using Configuration = blaze::DynamicVector<std::complex<double>>;

template <typename RNG>
Configuration makeRandomConfig(double const mean, double const std,
                               size_t const n, RNG &rng)
{
    Configuration phi(n);
    std::normal_distribution<double> dist(mean, std);
    std::generate(phi.begin(), phi.end(), [&]() mutable { return dist(rng); });
    return phi;
}

#endif  // ndef ISLE_UTIL_HPP
