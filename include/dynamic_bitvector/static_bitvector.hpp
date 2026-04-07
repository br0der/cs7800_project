#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace std;

namespace dbv {

// Succinct static bitvector supporting O(1) rank and select.
//
// Rank:   Jacobson (FOCS 1989) three-level approach
//           - superblocks of (lg^2 n) bits  → cumulative rank in lg n bits
//           - blocks of (½ lg n) bits       → relative rank in lg lg n bits
//           - in-block query via popcount
//
// Select: Binary search over superblocks + linear scan within,
//         giving O(lg n) select (practical; true O(1) Clark-Munro is
//         complex and rarely faster in practice for reasonable n).
//
// Space:  n + o(n) bits total.
class RankSelectBitVector {
public:
    RankSelectBitVector() noexcept : n_(0), ones_(0) {}

    // Build from a vector<bool> or vector<unsigned char>.
    // After construction the structure is immutable.
    explicit RankSelectBitVector(const vector<unsigned char>& raw) {
        build(raw.data(), raw.size());
    }

    explicit RankSelectBitVector(const vector<bool>& raw) {
        vector<unsigned char> tmp(raw.size());
        for (size_t i = 0; i < raw.size(); ++i) {
            tmp[i] = static_cast<unsigned char>(raw[i]);
        }
        build(tmp.data(), tmp.size());
    }

    // Build from a packed uint64_t array and a bit count.
    void build(const uint64_t* words, size_t num_bits) {
        n_ = num_bits;
        if (n_ == 0) {
            ones_ = 0;
            return;
        }

        size_t num_words = (n_ + kWordBits - 1) / kWordBits;
        words_.assign(words, words + num_words);

        // Mask out any trailing bits beyond n_.
        size_t tail = n_ % kWordBits;
        if (tail != 0) {
            words_.back() &= (uint64_t(1) << tail) - 1;
        }

        build_index();
    }

    size_t size() const noexcept {
        return n_;
    }

    bool access(size_t i) const {
        require(i < n_, "access index out of range");
        return (words_[i / kWordBits] >> (i % kWordBits)) & 1;
    }

    // rank1(i): number of 1 bits in [0, i)
    size_t rank1(size_t i) const {
        require(i <= n_, "rank1 index out of range");
        if (i == 0) return 0;
        --i; // convert to inclusive index for internal computation

        size_t sb = i / kSuperBlockBits;
        size_t b  = i / kBlockBits;
        size_t w  = i / kWordBits;
        size_t bit = i % kWordBits;

        size_t r = super_blocks_[sb];
        r += blocks_[b];

        // Add popcount of whole words between block start and word w.
        size_t block_start_word = (b * kBlockBits) / kWordBits;
        for (size_t j = block_start_word; j < w; ++j) {
            r += popcount64(words_[j]);
        }

        // Add popcount of bits [0..bit] in word w.
        uint64_t mask = (bit == 63) ? ~uint64_t(0) : (uint64_t(1) << (bit + 1)) - 1;
        r += popcount64(words_[w] & mask);

        return r;
    }

    // rank0(i): number of 0 bits in [0, i)
    size_t rank0(size_t i) const {
        require(i <= n_, "rank0 index out of range");
        return i - rank1(i);
    }

    // select1(k): position of the k-th 1 bit (0-indexed), or size() if absent.
    size_t select1(size_t k) const noexcept {
        if (k >= ones_) return n_;

        // Binary search over superblocks.
        size_t lo = 0;
        size_t hi = super_blocks_.size() - 1;
        while (lo < hi) {
            size_t mid = lo + (hi - lo + 1) / 2;
            if (super_blocks_[mid] <= k) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        size_t sb = lo;
        size_t rem = k - super_blocks_[sb];

        // Linear scan over blocks within the superblock.
        size_t b_start = (sb * kSuperBlockBits) / kBlockBits;
        size_t b_end = min(b_start + kSuperBlockBits / kBlockBits,
                           blocks_.size());
        size_t b = b_start;
        for (size_t j = b_start + 1; j < b_end; ++j) {
            if (blocks_[j] <= rem) {
                b = j;
            } else {
                break;
            }
        }
        rem -= blocks_[b];

        // Linear scan over words within the block.
        size_t w_start = (b * kBlockBits) / kWordBits;
        size_t w_end = min(w_start + kBlockBits / kWordBits,
                           words_.size());
        size_t w = w_start;
        for (; w < w_end; ++w) {
            size_t pc = popcount64(words_[w]);
            if (pc > rem) break;
            rem -= pc;
        }
        if (w >= words_.size()) return n_;

        // Find the rem-th set bit within word w.
        return w * kWordBits + select_in_word(words_[w], rem);
    }

    // select0(k): position of the k-th 0 bit (0-indexed), or size() if absent.
    size_t select0(size_t k) const noexcept {
        size_t zeros = n_ - ones_;
        if (k >= zeros) return n_;

        // Binary search over superblocks using rank0.
        size_t lo = 0;
        size_t hi = super_blocks_.size() - 1;
        while (lo < hi) {
            size_t mid = lo + (hi - lo + 1) / 2;
            size_t zeros_before = mid * kSuperBlockBits - super_blocks_[mid];
            if (zeros_before <= k) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        size_t sb = lo;
        size_t rem = k - (sb * kSuperBlockBits - super_blocks_[sb]);

        // Linear scan over blocks.
        size_t b_start = (sb * kSuperBlockBits) / kBlockBits;
        size_t b_end = min(b_start + kSuperBlockBits / kBlockBits,
                           blocks_.size());
        size_t b = b_start;
        for (size_t j = b_start + 1; j < b_end; ++j) {
            size_t zeros_in_block = (j - b_start) * kBlockBits - blocks_[j];
            // blocks_[j] is relative rank1 from block b_start's start
            // Actually blocks_[j] stores relative rank1 from superblock start,
            // so zeros relative to superblock start:
            size_t z = j * kBlockBits - sb * kSuperBlockBits - blocks_[j];
            if (z <= rem) {
                b = j;
            } else {
                break;
            }
        }
        size_t z_before_b = b * kBlockBits - sb * kSuperBlockBits - blocks_[b];
        rem -= z_before_b;

        // Linear scan over words.
        size_t w_start = (b * kBlockBits) / kWordBits;
        size_t w_end = min(w_start + kBlockBits / kWordBits,
                           words_.size());
        size_t w = w_start;
        for (; w < w_end; ++w) {
            size_t pc0 = kWordBits - popcount64(words_[w]);
            // For the last word, only count actual bits.
            if (w == words_.size() - 1 && n_ % kWordBits != 0) {
                pc0 = (n_ % kWordBits) - popcount64(words_[w]);
            }
            if (pc0 > rem) break;
            rem -= pc0;
        }
        if (w >= words_.size()) return n_;

        // Find the rem-th zero bit in word w.
        uint64_t inverted = ~words_[w];
        // Mask to valid bits if last word.
        if (w == words_.size() - 1 && n_ % kWordBits != 0) {
            inverted &= (uint64_t(1) << (n_ % kWordBits)) - 1;
        }
        return w * kWordBits + select_in_word(inverted, rem);
    }

    // Convenience: insert/erase/set/flip are not supported (static structure).
    void insert(size_t, bool) {
        throw logic_error("RankSelectBitVector is static; insert not supported");
    }

    void erase(size_t) {
        throw logic_error("RankSelectBitVector is static; erase not supported");
    }

    void set(size_t, bool) {
        throw logic_error("RankSelectBitVector is static; set not supported");
    }

    void flip(size_t) {
        throw logic_error("RankSelectBitVector is static; flip not supported");
    }

    void clear() noexcept {
        n_ = 0;
        ones_ = 0;
        words_.clear();
        super_blocks_.clear();
        blocks_.clear();
    }

private:
    static constexpr size_t kWordBits = 64;
    // Superblock: (lg^2 n) bits — we fix at 4096 = 64 * 64 for practicality.
    static constexpr size_t kSuperBlockBits = 4096;
    // Block: (½ lg n) bits — we fix at 64 (one word) for simplicity and speed.
    static constexpr size_t kBlockBits = 64;

    size_t n_;
    size_t ones_;
    vector<uint64_t> words_;
    vector<size_t> super_blocks_;  // cumulative rank1 at superblock boundaries
    vector<uint16_t> blocks_;      // relative rank1 within superblock

    // Build from raw byte array.
    void build(const unsigned char* raw, size_t num_bits) {
        n_ = num_bits;
        if (n_ == 0) {
            ones_ = 0;
            return;
        }

        size_t num_words = (n_ + kWordBits - 1) / kWordBits;
        words_.resize(num_words, 0);

        for (size_t i = 0; i < n_; ++i) {
            if (raw[i]) {
                words_[i / kWordBits] |= uint64_t(1) << (i % kWordBits);
            }
        }

        build_index();
    }

    void build_index() {
        size_t num_words = words_.size();

        // Superblocks.
        size_t num_sb = (n_ + kSuperBlockBits - 1) / kSuperBlockBits;
        super_blocks_.resize(num_sb, 0);

        // Blocks (one per word since kBlockBits == 64).
        blocks_.resize(num_words, 0);

        size_t cumulative = 0;
        for (size_t w = 0; w < num_words; ++w) {
            size_t sb = (w * kWordBits) / kSuperBlockBits;
            if (w * kWordBits % kSuperBlockBits == 0) {
                super_blocks_[sb] = cumulative;
            }
            blocks_[w] = static_cast<uint16_t>(cumulative - super_blocks_[sb]);
            cumulative += popcount64(words_[w]);
        }
        ones_ = cumulative;
    }

    static size_t popcount64(uint64_t x) noexcept {
        return static_cast<size_t>(__builtin_popcountll(x));
    }

    // Find the k-th set bit (0-indexed) in word x.
    static size_t select_in_word(uint64_t x, size_t k) noexcept {
#ifdef __BMI2__
        // Use pdep/tzcnt for hardware-accelerated select.
        uint64_t placed = _pdep_u64(uint64_t(1) << k, x);
        return static_cast<size_t>(__builtin_ctzll(placed));
#else
        for (size_t i = 0; i < 64; ++i) {
            if (x & (uint64_t(1) << i)) {
                if (k == 0) return i;
                --k;
            }
        }
        return 64;
#endif
    }

    static void require(bool cond, const char* msg) {
        if (!cond) {
            throw out_of_range(msg);
        }
    }
};

} // namespace dbv