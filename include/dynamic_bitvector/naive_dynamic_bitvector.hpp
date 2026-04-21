#pragma once

#include <cassert>
#include <cstddef>
#include <vector>

using namespace std;

namespace dbv {

// Baseline implementation used to verify the test harness itself.
// Replace this class with adapters for paper implementations.
class NaiveDynamicBitVector {
public:
    size_t size() const noexcept {
        return bits_.size();
    }

    size_t count_bits() const noexcept {
        return bits_.size();
    }

    bool access(size_t i) const {
        assert(i < bits_.size());
        return bits_[i] != 0;
    }

    // rank1(i): number of 1 bits in [0, i)
    size_t rank1(size_t i) const {
        assert(i <= bits_.size());
        size_t c = 0;
        for (size_t p = 0; p < i; ++p) {
            c += bits_[p] != 0;
        }
        return c;
    }

    // select1(k): position of k-th one (0-indexed), or size() if absent
    size_t select1(size_t k) const noexcept {
        size_t seen = 0;
        for (size_t i = 0; i < bits_.size(); ++i) {
            if (bits_[i] != 0) {
                if (seen == k) {
                    return i;
                }
                ++seen;
            }
        }
        return bits_.size();
    }

    void insert(size_t i, bool b) {
        assert(i <= bits_.size());
        bits_.insert(bits_.begin() + static_cast<ptrdiff_t>(i), static_cast<unsigned char>(b));
    }

    void erase(size_t i) {
        assert(i < bits_.size());
        bits_.erase(bits_.begin() + static_cast<ptrdiff_t>(i));
    }

    void set(size_t i, bool b) {
        assert(i < bits_.size());
        bits_[i] = static_cast<unsigned char>(b);
    }

    void flip(size_t i) {
        assert(i < bits_.size());
        bits_[i] = static_cast<unsigned char>(bits_[i] == 0);
    }

    void clear() noexcept {
        bits_.clear();
    }

private:
    vector<unsigned char> bits_;
};

} // namespace dbv
