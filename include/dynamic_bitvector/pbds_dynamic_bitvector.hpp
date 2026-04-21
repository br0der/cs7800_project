#pragma once

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(__has_include)
#if __has_include(<ext/pb_ds/assoc_container.hpp>) && __has_include(<ext/pb_ds/tree_policy.hpp>)
#define DBV_HAS_PBDS 1
#endif
#endif

#ifndef DBV_HAS_PBDS
#define DBV_HAS_PBDS 0
#endif

#if DBV_HAS_PBDS
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>

namespace dbv {

class PbdsDynamicBitVector {
    using BitTree = __gnu_pbds::tree<
        std::size_t,
        bool,
        std::less<std::size_t>,
        __gnu_pbds::rb_tree_tag,
        __gnu_pbds::tree_order_statistics_node_update>;

    using OneTree = __gnu_pbds::tree<
        std::size_t,
        __gnu_pbds::null_type,
        std::less<std::size_t>,
        __gnu_pbds::rb_tree_tag,
        __gnu_pbds::tree_order_statistics_node_update>;

public:
    std::size_t size() const noexcept {
        return bits_.size();
    }

    std::size_t count_bits() const noexcept {
        const std::size_t n = size();
        if (n == 0) {
            return 0;
        }

        const std::size_t coord_width = bits_for_value(n);
        const std::size_t bit_tree_node_bits = coord_width + 1 + coord_width + 1;
        const std::size_t one_tree_node_bits = coord_width + bits_for_value(ones_.size()) + 1;
        return n * bit_tree_node_bits + ones_.size() * one_tree_node_bits;
    }

    bool access(std::size_t i) const {
        const auto it = locate(i, "access index out of range");
        return it->second;
    }

    std::size_t rank1(std::size_t i) const {
        require(i <= size(), "rank1 index out of range");
        if (i == size()) {
            return ones_.size();
        }
        return ones_.order_of_key(i);
    }

    std::size_t select1(std::size_t k) const noexcept {
        const auto it = ones_.find_by_order(k);
        return it == ones_.end() ? size() : *it;
    }

    void insert(std::size_t i, bool b) {
        require(i <= size(), "insert index out of range");
        shift_suffix_right(i);
        bits_.insert({i, b});
        if (b) {
            ones_.insert(i);
        }
    }

    void erase(std::size_t i) {
        auto it = locate(i, "erase index out of range");
        const bool was_one = it->second;
        bits_.erase(it);
        if (was_one) {
            ones_.erase(i);
        }
        shift_suffix_left(i + 1);
    }

    void set(std::size_t i, bool b) {
        auto it = locate(i, "set index out of range");
        const bool old = it->second;
        if (old == b) {
            return;
        }
        it->second = b;
        if (b) {
            ones_.insert(i);
        } else {
            ones_.erase(i);
        }
    }

    void flip(std::size_t i) {
        auto it = locate(i, "flip index out of range");
        it->second = !it->second;
        if (it->second) {
            ones_.insert(i);
        } else {
            ones_.erase(i);
        }
    }

    void clear() noexcept {
        bits_.clear();
        ones_.clear();
    }

    bool check_invariants() const {
        std::size_t expected = 0;
        std::size_t ones = 0;

        for (auto it = bits_.begin(); it != bits_.end(); ++it, ++expected) {
            if (it->first != expected) {
                return false;
            }

            const bool is_one = it->second;
            if (is_one) {
                ++ones;
                if (ones_.find(expected) == ones_.end()) {
                    return false;
                }
            } else if (ones_.find(expected) != ones_.end()) {
                return false;
            }
        }

        return ones == ones_.size();
    }

private:
    using bit_iterator = BitTree::iterator;
    using const_bit_iterator = BitTree::const_iterator;

    static void require(bool cond, const char* msg) {
        if (!cond) {
            throw std::out_of_range(msg);
        }
    }

    static std::size_t ceil_log2(std::size_t x) {
        if (x <= 1) {
            return 0;
        }
        --x;
        std::size_t bits = 0;
        while (x != 0) {
            ++bits;
            x >>= 1;
        }
        return bits;
    }

    static std::size_t bits_for_value(std::size_t max_value) {
        return std::max<std::size_t>(1, ceil_log2(max_value + 1));
    }

    bit_iterator locate(std::size_t i, const char* msg) {
        require(i < size(), msg);
        auto it = bits_.find_by_order(i);
        require(it != bits_.end() && it->first == i, "PBDS position invariant failure");
        return it;
    }

    const_bit_iterator locate(std::size_t i, const char* msg) const {
        require(i < size(), msg);
        auto it = bits_.find_by_order(i);
        require(it != bits_.end() && it->first == i, "PBDS position invariant failure");
        return it;
    }

    void shift_suffix_right(std::size_t first) {
        std::vector<std::pair<std::size_t, bool>> moved;
        for (auto it = bits_.lower_bound(first); it != bits_.end(); ++it) {
            moved.push_back({it->first, it->second});
        }

        for (const auto& [pos, bit] : moved) {
            bits_.erase(pos);
            if (bit) {
                ones_.erase(pos);
            }
        }

        for (const auto& [pos, bit] : moved) {
            bits_.insert({pos + 1, bit});
            if (bit) {
                ones_.insert(pos + 1);
            }
        }
    }

    void shift_suffix_left(std::size_t first) {
        std::vector<std::pair<std::size_t, bool>> moved;
        for (auto it = bits_.lower_bound(first); it != bits_.end(); ++it) {
            moved.push_back({it->first, it->second});
        }

        for (const auto& [pos, bit] : moved) {
            bits_.erase(pos);
            if (bit) {
                ones_.erase(pos);
            }
        }

        for (const auto& [pos, bit] : moved) {
            bits_.insert({pos - 1, bit});
            if (bit) {
                ones_.insert(pos - 1);
            }
        }
    }

    BitTree bits_;
    OneTree ones_;
};

} // namespace dbv
#endif
