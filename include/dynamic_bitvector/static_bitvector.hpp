#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace std;

namespace dbv {

// Succinct static Jacobson/Munro bitvector (from class) supporting O(1) rank and select.
class RankSelectBitVector {
public:
    RankSelectBitVector() noexcept : n_(0), ones_(0) {}

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

    void build(const uint64_t* words, size_t num_bits) {
        n_ = num_bits;
        if (n_ == 0) {
            ones_ = 0;
            return;
        }

        size_t num_words = (n_ + kWordBits - 1) / kWordBits;
        words_.assign(words, words + num_words);

        size_t tail = n_ % kWordBits;
        if (tail != 0) {
            words_.back() &= (uint64_t(1) << tail) - 1;
        }

        build_index();
    }

    size_t size() const noexcept {
        return n_;
    }

    size_t count_bits() const noexcept {
        if (n_ == 0) {
            return 0;
        }

        size_t bits = n_;
        bits += 2 * bits_for_value(n_);
        bits += super_blocks_.size() * bits_for_value(n_);
        bits += blocks_.size() * bits_for_value(kSuperBlockBits);
        return bits;
    }

    bool access(size_t i) const {
        assert(i < n_);
        return (words_[i / kWordBits] >> (i % kWordBits)) & 1;
    }

    // rank1(i): number of 1 bits in [0, i)
    size_t rank1(size_t i) const {
        assert(i <= n_);
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

        return w * kWordBits + select_in_word(words_[w], rem);
    }

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
    static constexpr size_t kSuperBlockBits = 4096;
    static constexpr size_t kBlockBits = 64;

    size_t n_;
    size_t ones_;
    vector<uint64_t> words_;
    vector<size_t> super_blocks_; // cumulative rank1 at superblock boundaries
    vector<uint16_t> blocks_; // relative rank1 within superblock

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

        // Superblocks
        size_t num_sb = (n_ + kSuperBlockBits - 1) / kSuperBlockBits;
        super_blocks_.resize(num_sb, 0);

        // Blocks (one per word since kBlockBits == 64)
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

    static size_t ceil_log2(size_t x) noexcept {
        if (x <= 1) return 0;
        --x;
        size_t bits = 0;
        while (x != 0) {
            ++bits;
            x >>= 1;
        }
        return bits;
    }

    static size_t bits_for_value(size_t max_value) noexcept {
        return max<size_t>(1, ceil_log2(max_value + 1));
    }

    // Find the k-th set bit in word x.
    static size_t select_in_word(uint64_t x, size_t k) noexcept {

// Again, checking to see if we can use _pdep_u64 because brady's compiler sucks
#ifdef __BMI2__
        // Use pdep for hardware-accelerated select.
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

};

} // namespace dbv
