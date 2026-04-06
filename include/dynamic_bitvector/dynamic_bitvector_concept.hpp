#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

using namespace std;

namespace dbv {

// Contract used by the test harness:
// - access(i): bit at index i, where 0 <= i < size()
// - rank1(i): number of 1 bits in prefix [0, i)
// - select1(k): index of k-th one (0-indexed), or size() if k is out of range
// - insert(i, b): insert bit b at position i, where 0 <= i <= size()
// - erase(i): erase bit at i, where 0 <= i < size()
// - set(i, b): overwrite bit at i
// - flip(i): toggle bit at i
// - clear(): remove all bits

template <typename T, typename = void>
struct is_query_bitvector : false_type {};

template <typename T>
struct is_query_bitvector<T, void_t<
    decltype(T{}),
    decltype(declval<T&>().size()),
    decltype(declval<T&>().access(size_t{})),
    decltype(declval<T&>().rank1(size_t{})),
    decltype(declval<T&>().select1(size_t{}))>> : true_type {};

template <typename T>
inline constexpr bool is_query_bitvector_v = is_query_bitvector<T>::value;

template <typename T, typename = void>
struct is_dynamic_bitvector : false_type {};

template <typename T>
struct is_dynamic_bitvector<T, void_t<
    decltype(declval<T&>().insert(size_t{}, bool{})),
    decltype(declval<T&>().erase(size_t{})),
    decltype(declval<T&>().set(size_t{}, bool{})),
    decltype(declval<T&>().flip(size_t{})),
    decltype(declval<T&>().clear())>> : bool_constant<is_query_bitvector_v<T>> {};

template <typename T>
inline constexpr bool is_dynamic_bitvector_v = is_dynamic_bitvector<T>::value;

} // namespace dbv
