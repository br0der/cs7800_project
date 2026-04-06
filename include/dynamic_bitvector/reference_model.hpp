#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

using namespace std;

namespace dbv {

class ReferenceBitVector {
public:
    size_t size() const noexcept {
        return bits_.size();
    }

    bool access(size_t i) const {
        bounds_check(i);
        return bits_[i] != 0;
    }

    // rank1(i): number of 1 bits in prefix [0, i)
    size_t rank1(size_t i) const {
        if (i > bits_.size()) {
            throw out_of_range("rank1 index out of range");
        }
        size_t count = 0;
        for (size_t p = 0; p < i; ++p) {
            count += bits_[p] != 0;
        }
        return count;
    }

    // select1(k): index of k-th one bit (0-indexed), or size() if absent
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
        if (i > bits_.size()) {
            throw out_of_range("insert index out of range");
        }
        bits_.insert(bits_.begin() + static_cast<ptrdiff_t>(i), static_cast<unsigned char>(b));
    }

    void erase(size_t i) {
        bounds_check(i);
        bits_.erase(bits_.begin() + static_cast<ptrdiff_t>(i));
    }

    void set(size_t i, bool b) {
        bounds_check(i);
        bits_[i] = static_cast<unsigned char>(b);
    }

    void flip(size_t i) {
        bounds_check(i);
        bits_[i] = static_cast<unsigned char>(bits_[i] == 0);
    }

    void clear() noexcept {
        bits_.clear();
    }

private:
    void bounds_check(size_t i) const {
        if (i >= bits_.size()) {
            throw out_of_range("index out of range");
        }
    }

    vector<unsigned char> bits_;
};

} // namespace dbv
