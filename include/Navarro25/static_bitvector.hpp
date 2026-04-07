#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace Navarro25 {

struct PackedIntVector {
    std::size_t width = 1;
    std::size_t count = 0;
    std::vector<std::uint64_t> words;

    void reset(std::size_t new_count, std::size_t new_width) {
        count = new_count;
        width = std::max<std::size_t>(1, new_width);
        words.assign(word_count_for(count * width), 0);
    }

    std::size_t size() const noexcept {
        return count;
    }

    bool empty() const noexcept {
        return count == 0;
    }

    void clear() noexcept {
        width = 1;
        count = 0;
        words.clear();
    }

    void set(std::size_t index, std::size_t value) {
        for (std::size_t bit = 0; bit < width; ++bit) {
            const std::size_t pos = index * width + bit;
            const std::uint64_t mask = std::uint64_t{1} << (pos % 64);
            auto& word = words[pos / 64];
            if (((value >> bit) & std::size_t{1}) != 0) {
                word |= mask;
            } else {
                word &= ~mask;
            }
        }
    }

    std::size_t get(std::size_t index) const {
        std::size_t value = 0;
        for (std::size_t bit = 0; bit < width; ++bit) {
            const std::size_t pos = index * width + bit;
            if (((words[pos / 64] >> (pos % 64)) & std::uint64_t{1}) != 0) {
                value |= std::size_t{1} << bit;
            }
        }
        return value;
    }

    static std::size_t word_count_for(std::size_t bit_count) {
        return bit_count == 0 ? 0 : 1 + (bit_count - 1) / 64;
    }
};

struct StaticBitVector {
    std::size_t bit_count = 0;
    std::size_t one_count = 0;
    std::size_t superblock_bits = 1;
    std::size_t block_bits = 1;
    std::size_t select_sample_rate = 1;
    std::vector<std::uint64_t> words;
    PackedIntVector superblock_rank;
    PackedIntVector block_rank;
    PackedIntVector select1_samples;
    PackedIntVector select0_samples;

    void build(const std::vector<unsigned char>& bits) {
        clear();
        bit_count = bits.size();
        words.assign(word_count_for(bit_count), 0);

        for (std::size_t i = 0; i < bits.size(); ++i) {
            if (bits[i] != 0) {
                words[i / 64] |= std::uint64_t{1} << (i % 64);
            }
        }
        mask_unused_tail_bits();

        choose_parameters();
        build_rank_index();
        build_select_index();
    }

    void clear() noexcept {
        bit_count = 0;
        one_count = 0;
        superblock_bits = 1;
        block_bits = 1;
        select_sample_rate = 1;
        words.clear();
        superblock_rank.clear();
        block_rank.clear();
        select1_samples.clear();
        select0_samples.clear();
    }

    std::size_t size() const noexcept {
        return bit_count;
    }

    std::size_t ones() const noexcept {
        return one_count;
    }

    bool access(std::size_t i) const {
        require(i < bit_count, "StaticBitVector access index out of range");
        return get_bit(i);
    }

    std::size_t rank1(std::size_t i) const {
        require(i <= bit_count, "StaticBitVector rank1 index out of range");
        if (i == bit_count) {
            return one_count;
        }
        if (i == 0) {
            return 0;
        }

        const std::size_t super = i / superblock_bits;
        const std::size_t block = i / block_bits;
        const std::size_t block_start = block * block_bits;
        return superblock_rank.get(super) + block_rank.get(block) + popcount_range(block_start, i);
    }

    std::size_t rank0(std::size_t i) const {
        require(i <= bit_count, "StaticBitVector rank0 index out of range");
        return i - rank1(i);
    }

    std::size_t select1(std::size_t k) const {
        if (k >= one_count) {
            return bit_count;
        }
        return select_by_rank(k, true);
    }

    std::size_t select0(std::size_t k) const {
        const std::size_t zero_count = bit_count - one_count;
        if (k >= zero_count) {
            return bit_count;
        }
        return select_by_rank(k, false);
    }

    void append_bits(std::vector<unsigned char>& out) const {
        out.reserve(out.size() + bit_count);
        for (std::size_t i = 0; i < bit_count; ++i) {
            out.push_back(static_cast<unsigned char>(get_bit(i)));
        }
    }

    void choose_parameters() {
        const std::size_t lg = std::max<std::size_t>(1, ceil_log2(std::max<std::size_t>(2, bit_count)));
        block_bits = std::max<std::size_t>(1, lg / 2);
        superblock_bits = std::max<std::size_t>(block_bits, lg * lg);
        superblock_bits = round_up_to_multiple(superblock_bits, block_bits);
        select_sample_rate = std::max<std::size_t>(1, lg * lg);
    }

    void build_rank_index() {
        if (bit_count == 0) {
            one_count = 0;
            superblock_rank.clear();
            block_rank.clear();
            return;
        }

        const std::size_t superblocks = ceil_div(bit_count, superblock_bits);
        const std::size_t blocks = ceil_div(bit_count, block_bits);
        superblock_rank.reset(superblocks, bits_for_value(bit_count));
        block_rank.reset(blocks, bits_for_value(superblock_bits));

        std::size_t rank = 0;
        std::size_t current_super_rank = 0;
        for (std::size_t i = 0; i < bit_count; ++i) {
            if (i % superblock_bits == 0) {
                current_super_rank = rank;
                superblock_rank.set(i / superblock_bits, current_super_rank);
            }
            if (i % block_bits == 0) {
                block_rank.set(i / block_bits, rank - current_super_rank);
            }
            rank += get_bit(i) ? 1 : 0;
        }
        one_count = rank;
    }

    void build_select_index() {
        const std::size_t zero_count = bit_count - one_count;
        select1_samples.reset(ceil_div(one_count, select_sample_rate), bits_for_value(bit_count));
        select0_samples.reset(ceil_div(zero_count, select_sample_rate), bits_for_value(bit_count));

        std::size_t ones_seen = 0;
        std::size_t zeros_seen = 0;
        for (std::size_t i = 0; i < bit_count; ++i) {
            if (get_bit(i)) {
                if (ones_seen % select_sample_rate == 0) {
                    select1_samples.set(ones_seen / select_sample_rate, i);
                }
                ++ones_seen;
            } else {
                if (zeros_seen % select_sample_rate == 0) {
                    select0_samples.set(zeros_seen / select_sample_rate, i);
                }
                ++zeros_seen;
            }
        }
    }

    std::size_t select_by_rank(std::size_t k, bool bit) const {
        const auto& samples = bit ? select1_samples : select0_samples;
        const std::size_t sample = k / select_sample_rate;
        std::size_t lo = samples.get(sample);
        std::size_t hi = sample + 1 < samples.size() ? samples.get(sample + 1) : bit_count;

        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2;
            const std::size_t count = bit ? rank1(mid + 1) : rank0(mid + 1);
            if (count > k) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }

    bool get_bit(std::size_t i) const {
        return ((words[i / 64] >> (i % 64)) & std::uint64_t{1}) != 0;
    }

    std::size_t popcount_range(std::size_t begin, std::size_t end) const {
        std::size_t count = 0;
        while (begin < end && (begin % 64) != 0) {
            count += get_bit(begin) ? 1 : 0;
            ++begin;
        }
        while (begin + 64 <= end) {
            count += std::popcount(words[begin / 64]);
            begin += 64;
        }
        while (begin < end) {
            count += get_bit(begin) ? 1 : 0;
            ++begin;
        }
        return count;
    }

    void mask_unused_tail_bits() {
        if (words.empty()) {
            return;
        }

        const std::size_t used = bit_count % 64;
        if (used != 0) {
            words.back() &= (std::uint64_t{1} << used) - 1;
        }
    }

    static std::size_t ceil_div(std::size_t x, std::size_t y) {
        return x == 0 ? 0 : 1 + (x - 1) / y;
    }

    static std::size_t word_count_for(std::size_t bits) {
        return ceil_div(bits, static_cast<std::size_t>(64));
    }

    static std::size_t round_up_to_multiple(std::size_t value, std::size_t multiple) {
        return multiple == 0 ? value : ceil_div(value, multiple) * multiple;
    }

    static std::size_t ceil_log2(std::size_t x) {
        if (x <= 1) {
            return 0;
        }
        --x;
        std::size_t result = 0;
        while (x != 0) {
            ++result;
            x >>= 1;
        }
        return result;
    }

    static std::size_t bits_for_value(std::size_t max_value) {
        return std::max<std::size_t>(1, ceil_log2(max_value + 1));
    }

    static void require(bool cond, const char* msg) {
        if (!cond) {
            throw std::out_of_range(msg);
        }
    }
};

} // namespace Navarro25
