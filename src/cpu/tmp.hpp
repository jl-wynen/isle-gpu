#ifndef TMP_HPP
#define TMP_HPP

#include <utility>

namespace tmp {

template <template <typename...> class C, typename T>
struct IsSpecialization : public std::false_type
{
};

template <template <typename...> class C, typename... Args>
struct IsSpecialization<C, C<Args...>> : public std::true_type
{
};

template <typename...>
using void_t = void;

template <typename T>
struct AlwaysFalse
{
    static constexpr bool value = false;
};
template <typename T>
constexpr bool AlwaysFalse_v = AlwaysFalse<T>::value;

template <typename Orig, typename Other, typename = void>
struct Rebind
{
    using type = Other;
};

template <typename Orig, typename Other>
struct Rebind<Orig, Other,
              void_t<typename Orig::template Rebind<Other>::Other>>
{
    using type = typename Orig::template Rebind<Other>::Other;
};

template <typename Orig, typename Other>
using Rebind_t = typename Rebind<Orig, Other>::type;

}  // namespace tmp

#endif  // ndef TMP_HPP
