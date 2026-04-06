#pragma once

#include <cstddef>
#include <cmath>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace std;

namespace dbv {

class RankSelectBitVector {
public:
    RankSelectBitVector() = default;

    explicit RankSelectBitVector(const vector<bool>& input) {
        bits_ = input;
        n_ = bits_.size();

        int lg = (n_ > 1 ? static_cast<int>(log2(n_)) : 1);
        superblock_size_ = max(1, lg * lg);
        block_size_ = max(1, lg / 2);

        build();
    }

    size_t size() const noexcept {
        return bits_.size();
    }

    bool access(size_t i) const {
        require(i < bits_.size(), "access index out of range");
        return bits_[i];
    }

    // rank1(i): number of 1 bits in [0, i)
    size_t rank1(size_t i) const {
        require(i <= bits_.size(), "rank1 index out of range");
        if (i == 0) {
            return 0;
        }
        return static_cast<size_t>(rank(static_cast<int>(i - 1)));
    }

    // select1(k): position of k-th one (0-indexed), or size() if absent
    size_t select1(size_t k) const noexcept {
        int ans = select(static_cast<int>(k + 1));
        if (ans == -1) {
            return bits_.size();
        }
        return static_cast<size_t>(ans);
    }

    void print_bits() const {
        for (bool b : bits_) {
            cout << b;
        }
        cout << '\n';
    }

private:
    vector<bool> bits_;
    vector<int> superblock_rank_;
    vector<int> block_rank_;

    size_t n_ = 0;
    int superblock_size_ = 1;
    int block_size_ = 1;

    void build() {
        int curr_rank = 0;

        int num_superblocks =
            static_cast<int>((n_ + static_cast<size_t>(superblock_size_) - 1) / static_cast<size_t>(superblock_size_));
        int num_blocks =
            static_cast<int>((n_ + static_cast<size_t>(block_size_) - 1) / static_cast<size_t>(block_size_));

        superblock_rank_.assign(num_superblocks, 0);
        block_rank_.assign(num_blocks, 0);

        for (int i = 0; i < static_cast<int>(n_); i++) {
            if (i % superblock_size_ == 0) {
                superblock_rank_[i / superblock_size_] = curr_rank;
            }
            if (i % block_size_ == 0) {
                block_rank_[i / block_size_] = curr_rank;
            }
            if (bits_[i]) {
                curr_rank++;
            }
        }
    }

    // rank(i): number of 1s in bits_[0..i]
    int rank(int i) const {
        assert(i >= 0 && i < static_cast<int>(n_));

        int sb = i / superblock_size_;
        int b = i / block_size_;

        int ans = superblock_rank_[sb];

        int superblock_start = sb * superblock_size_;
        int block_start = b * block_size_;

        for (int j = superblock_start; j < block_start; j++) {
            if (bits_[j]) {
                ans++;
            }
        }
        for (int j = block_start; j <= i; j++) {
            if (bits_[j]) {
                ans++;
            }
        }

        return ans;
    }

    // select(j): returns index of j-th 1, where j is 1-indexed
    int select(int j) const {
        assert(j > 0);

        int count = 0;
        for (int i = 0; i < static_cast<int>(n_); i++) {
            if (bits_[i]) {
                count++;
            }
            if (count == j) {
                return i;
            }
        }
        return -1;
    }

    static void require(bool cond, const char* msg) {
        if (!cond) {
            throw out_of_range(msg);
        }
    }
};

} // namespace dbv