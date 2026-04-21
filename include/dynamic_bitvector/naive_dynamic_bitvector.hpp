#pragma once

#include <cstddef>
#include <stdexcept>
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
        require(i < bits_.size(), "access index out of range");
        return bits_[i] != 0;
    }

    // rank1(i): number of 1 bits in [0, i)
    size_t rank1(size_t i) const {
        require(i <= bits_.size(), "rank1 index out of range");
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
        require(i <= bits_.size(), "insert index out of range");
        bits_.insert(bits_.begin() + static_cast<ptrdiff_t>(i), static_cast<unsigned char>(b));
    }

    void erase(size_t i) {
        require(i < bits_.size(), "erase index out of range");
        bits_.erase(bits_.begin() + static_cast<ptrdiff_t>(i));
    }

    void set(size_t i, bool b) {
        require(i < bits_.size(), "set index out of range");
        bits_[i] = static_cast<unsigned char>(b);
    }

    void flip(size_t i) {
        require(i < bits_.size(), "flip index out of range");
        bits_[i] = static_cast<unsigned char>(bits_[i] == 0);
    }

    void clear() noexcept {
        bits_.clear();
    }

private:
    static void require(bool cond, const char* msg) {
        if (!cond) {
            throw out_of_range(msg);
        }
    }

    vector<unsigned char> bits_;
};

} // namespace dbv
