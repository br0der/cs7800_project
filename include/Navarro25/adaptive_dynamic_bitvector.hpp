#pragma once

#include "Navarro25/static_bitvector.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Navarro25 {

enum node_kind {
    internal_node,
    dynamic_leaf,
    static_leaf
};

struct node {
    node_kind kind = dynamic_leaf;
    std::size_t level = 0;
    std::size_t bit_count = 0;
    std::size_t one_count = 0;
    mutable std::size_t queries = 0;
    std::vector<std::unique_ptr<node>> children;
    std::vector<std::uint64_t> words;
    StaticBitVector static_bits;
};

struct AdaptiveDynamicBitVector {
    mutable std::unique_ptr<node> root;
    std::size_t total_bits = 0;
    std::size_t arity = parameter_arity(0);
    std::size_t leaf_capacity = parameter_leaf_capacity(0);

    AdaptiveDynamicBitVector() {
        root = make_dynamic_leaf({}, 0);
    }

    std::size_t size() const noexcept {
        return total_bits;
    }

    bool access(std::size_t i) const {
        require(i < total_bits, "access index out of range");
        return access_impl(root, i);
    }

    std::size_t rank1(std::size_t i) const {
        require(i <= total_bits, "rank1 index out of range");
        if (total_bits == 0) {
            return 0;
        }
        return rank1_impl(root, i);
    }

    std::size_t rank0(std::size_t i) const {
        require(i <= total_bits, "rank0 index out of range");
        return i - rank1(i);
    }

    std::size_t select1(std::size_t k) const {
        if (!root || k >= root->one_count) {
            return total_bits;
        }
        return select1_impl(root, k);
    }

    std::size_t select0(std::size_t k) const {
        const std::size_t zeros = total_bits - (root ? root->one_count : 0);
        if (k >= zeros) {
            return total_bits;
        }
        return select0_impl(root, k);
    }

    void insert(std::size_t i, bool bit) {
        require(i <= total_bits, "insert index out of range");
        ensure_root();
        insert_impl(root, i, bit);
        finish_update();
    }

    void erase(std::size_t i) {
        require(i < total_bits, "erase index out of range");
        erase_impl(root, i);
        finish_update();
    }

    void set(std::size_t i, bool bit) {
        require(i < total_bits, "set index out of range");
        set_impl(root, i, bit);
        finish_update();
    }

    void flip(std::size_t i) {
        require(i < total_bits, "flip index out of range");
        flip_impl(root, i);
        finish_update();
    }

    void clear() noexcept {
        total_bits = 0;
        arity = parameter_arity(0);
        leaf_capacity = parameter_leaf_capacity(0);
        root = make_dynamic_leaf({}, 0);
    }

    std::size_t static_leaf_count() const {
        return count_kind(root.get(), static_leaf);
    }

    std::size_t dynamic_leaf_count() const {
        return count_kind(root.get(), dynamic_leaf);
    }

    std::size_t internal_node_count() const {
        return count_kind(root.get(), internal_node);
    }

    bool check_invariants() const {
        if (!root) {
            return total_bits == 0;
        }

        const auto summary = check_node_invariants(root.get());
        return summary.ok &&
               summary.bits == total_bits &&
               summary.bits == root->bit_count &&
               summary.ones == root->one_count;
    }

    struct invariant_summary {
        bool ok = true;
        std::size_t bits = 0;
        std::size_t ones = 0;
    };

    static std::size_t parameter_arity(std::size_t n) {
        const double guarded = static_cast<double>(std::max<std::size_t>(n, 4));
        const double value = std::ceil(std::sqrt(std::log2(guarded)));
        return std::max<std::size_t>(16, static_cast<std::size_t>(value));
    }

    static std::size_t parameter_leaf_capacity(std::size_t n) {
        const double guarded = static_cast<double>(std::max<std::size_t>(n, 4));
        const double log_n = std::max(2.0, std::log2(guarded));
        const double log_log_n = std::max(1.0, std::log2(log_n));
        const double raw = (log_n * log_n) / (16.0 * log_log_n);
        const auto rounded = static_cast<std::size_t>(std::ceil(raw));
        return std::max<std::size_t>(16, 16 * std::max<std::size_t>(1, rounded));
    }

    static void require(bool cond, const char* msg) {
        if (!cond) {
            throw std::out_of_range(msg);
        }
    }

    static std::size_t ceil_div(std::size_t x, std::size_t y) {
        return x == 0 ? 0 : 1 + (x - 1) / y;
    }

    static std::size_t word_count_for(std::size_t bit_count) {
        return ceil_div(bit_count, static_cast<std::size_t>(64));
    }

    static std::uint64_t tail_mask(std::size_t bit_count) {
        const std::size_t used = bit_count % 64;
        if (used == 0) {
            return ~std::uint64_t{0};
        }
        return (std::uint64_t{1} << used) - 1;
    }

    static bool dynamic_get_bit(const node& n, std::size_t i) {
        return ((n.words[i / 64] >> (i % 64)) & std::uint64_t{1}) != 0;
    }

    static void dynamic_set_bit(node& n, std::size_t i, bool bit) {
        const auto mask = std::uint64_t{1} << (i % 64);
        auto& word = n.words[i / 64];
        if (bit) {
            word |= mask;
        } else {
            word &= ~mask;
        }
    }

    static void mask_unused_tail_bits(node& n) {
        n.words.resize(word_count_for(n.bit_count));
        if (!n.words.empty()) {
            n.words.back() &= tail_mask(n.bit_count);
        }
    }

    static std::vector<std::uint64_t> pack_bits(const std::vector<unsigned char>& bits) {
        std::vector<std::uint64_t> words(word_count_for(bits.size()), 0);
        for (std::size_t i = 0; i < bits.size(); ++i) {
            if (bits[i] != 0) {
                words[i / 64] |= std::uint64_t{1} << (i % 64);
            }
        }
        return words;
    }

    static void unpack_dynamic_bits(const node& n, std::vector<unsigned char>& out) {
        out.reserve(out.size() + n.bit_count);
        for (std::size_t i = 0; i < n.bit_count; ++i) {
            out.push_back(static_cast<unsigned char>(dynamic_get_bit(n, i)));
        }
    }

    static std::size_t dynamic_popcount(const node& n) {
        if (n.bit_count == 0) {
            return 0;
        }

        std::size_t count = 0;
        const std::size_t words = word_count_for(n.bit_count);
        for (std::size_t i = 0; i < words; ++i) {
            std::uint64_t word = n.words[i];
            if (i + 1 == words) {
                word &= tail_mask(n.bit_count);
            }
            count += std::popcount(word);
        }
        return count;
    }

    void ensure_root() {
        if (!root) {
            root = make_dynamic_leaf({}, 0);
        }
    }

    std::size_t max_children() const {
        return 4 * arity;
    }

    std::size_t max_weight(std::size_t level) const {
        std::size_t weight = leaf_capacity;
        for (std::size_t i = 0; i < level; ++i) {
            if (weight > std::numeric_limits<std::size_t>::max() / arity) {
                return std::numeric_limits<std::size_t>::max();
            }
            weight *= arity;
        }
        return weight;
    }

    std::size_t min_weight(std::size_t level) const {
        return std::max<std::size_t>(1, max_weight(level) / 4);
    }

    std::size_t level_for_size(std::size_t bit_count) const {
        std::size_t level = 0;
        while (bit_count > max_weight(level)) {
            ++level;
        }
        return level;
    }

    std::unique_ptr<node> make_dynamic_leaf(const std::vector<unsigned char>& src, std::size_t level) const {
        auto out = std::make_unique<node>();
        out->kind = dynamic_leaf;
        out->level = level;
        out->bit_count = src.size();
        out->words = pack_bits(src);
        recompute_metadata(*out);
        return out;
    }

    std::unique_ptr<node> make_dynamic_leaf_range(
        const std::vector<unsigned char>& src,
        std::size_t begin,
        std::size_t count,
        std::size_t level) const {
        std::vector<unsigned char> bits;
        bits.reserve(count);
        bits.insert(bits.end(), src.begin() + static_cast<std::ptrdiff_t>(begin),
                    src.begin() + static_cast<std::ptrdiff_t>(begin + count));
        return make_dynamic_leaf(bits, level);
    }

    std::unique_ptr<node> make_static_leaf(const std::vector<unsigned char>& src, std::size_t level) const {
        auto out = std::make_unique<node>();
        out->kind = static_leaf;
        out->level = level;
        out->static_bits.build(src);
        recompute_metadata(*out);
        return out;
    }

    std::unique_ptr<node> make_static_leaf_range(
        const std::vector<unsigned char>& src,
        std::size_t begin,
        std::size_t count,
        std::size_t level) const {
        std::vector<unsigned char> bits;
        bits.reserve(count);
        bits.insert(bits.end(), src.begin() + static_cast<std::ptrdiff_t>(begin),
                    src.begin() + static_cast<std::ptrdiff_t>(begin + count));
        return make_static_leaf(bits, level);
    }

    std::unique_ptr<node> make_internal(std::size_t level) const {
        auto out = std::make_unique<node>();
        out->kind = internal_node;
        out->level = level;
        return out;
    }

    void recompute_metadata(node& n) const {
        if (n.kind == internal_node) {
            n.bit_count = 0;
            n.one_count = 0;
            for (const auto& child : n.children) {
                if (child) {
                    n.bit_count += child->bit_count;
                    n.one_count += child->one_count;
                }
            }
            n.words.clear();
            n.static_bits.clear();
            return;
        }

        if (n.kind == dynamic_leaf) {
            mask_unused_tail_bits(n);
            n.one_count = dynamic_popcount(n);
            n.static_bits.clear();
            n.children.clear();
        } else {
            n.bit_count = n.static_bits.size();
            n.one_count = n.static_bits.ones();
            n.words.clear();
            n.children.clear();
            n.queries = 0;
        }
    }

    void collect_bits(const node& n, std::vector<unsigned char>& out) const {
        if (n.kind == internal_node) {
            for (const auto& child : n.children) {
                if (child) {
                    collect_bits(*child, out);
                }
            }
            return;
        }
        if (n.kind == dynamic_leaf) {
            unpack_dynamic_bits(n, out);
        } else {
            n.static_bits.append_bits(out);
        }
    }

    std::vector<unsigned char> collect_bits_copy(const node& n) const {
        std::vector<unsigned char> bits;
        bits.reserve(n.bit_count);
        collect_bits(n, bits);
        return bits;
    }

    std::unique_ptr<node> build_subtree_exact_level(
        const std::vector<unsigned char>& bits,
        std::size_t begin,
        std::size_t count,
        std::size_t level) const {
        if (level == 0 || count == 0) {
            return make_dynamic_leaf_range(bits, begin, count, 0);
        }

        auto out = make_internal(level);
        const std::size_t child_level = level - 1;
        const std::size_t child_limit = std::max<std::size_t>(1, max_weight(child_level));
        std::size_t child_count = std::max<std::size_t>(1, ceil_div(count, child_limit));
        child_count = std::min<std::size_t>(child_count, max_children());

        const std::size_t base = count / child_count;
        std::size_t extra = count % child_count;
        std::size_t offset = begin;

        for (std::size_t i = 0; i < child_count; ++i) {
            const std::size_t part = base + (extra > 0 ? 1 : 0);
            if (extra > 0) {
                --extra;
            }
            out->children.push_back(build_subtree_exact_level(bits, offset, part, child_level));
            offset += part;
        }

        recompute_metadata(*out);
        return out;
    }

    std::unique_ptr<node> build_tree_from_bits(const std::vector<unsigned char>& bits) const {
        if (bits.empty()) {
            return make_dynamic_leaf({}, 0);
        }
        return build_subtree_exact_level(bits, 0, bits.size(), level_for_size(bits.size()));
    }

    void flatten_slot(std::unique_ptr<node>& slot) const {
        if (!slot || slot->kind != internal_node) {
            return;
        }
        const std::size_t level = slot->level;
        const auto bits = collect_bits_copy(*slot);
        slot = make_static_leaf(bits, level);
    }

    void materialize_static_leaf(std::unique_ptr<node>& slot) const {
        if (!slot || slot->kind != static_leaf) {
            return;
        }
        const std::size_t level = slot->level;
        std::vector<unsigned char> bits;
        slot->static_bits.append_bits(bits);
        slot = build_subtree_exact_level(bits, 0, bits.size(), level);
    }

    bool maybe_flatten_after_query(std::unique_ptr<node>& slot) const {
        if (!slot || slot->kind != internal_node || slot->bit_count == 0) {
            return false;
        }
        ++slot->queries;
        if (slot->queries >= slot->bit_count) {
            flatten_slot(slot);
            return true;
        }
        return false;
    }

    bool access_leaf(const node& n, std::size_t i) const {
        if (n.kind == dynamic_leaf) {
            return dynamic_get_bit(n, i);
        }
        return n.static_bits.access(i);
    }

    std::size_t rank1_leaf(const node& n, std::size_t i) const {
        if (n.kind == static_leaf) {
            return n.static_bits.rank1(i);
        }

        const std::size_t full_words = i / 64;
        const std::size_t tail_bits = i % 64;
        std::size_t count = 0;
        for (std::size_t word = 0; word < full_words; ++word) {
            count += std::popcount(n.words[word]);
        }
        if (tail_bits > 0) {
            const std::uint64_t mask = (std::uint64_t{1} << tail_bits) - 1;
            count += std::popcount(n.words[full_words] & mask);
        }
        return count;
    }

    std::size_t select1_leaf(const node& n, std::size_t k) const {
        if (n.kind == static_leaf) {
            return n.static_bits.select1(k);
        }

        const std::size_t words = word_count_for(n.bit_count);
        for (std::size_t word_index = 0; word_index < words; ++word_index) {
            std::uint64_t word = n.words[word_index];
            if (word_index + 1 == words) {
                word &= tail_mask(n.bit_count);
            }
            const std::size_t ones = std::popcount(word);
            if (k >= ones) {
                k -= ones;
                continue;
            }

            while (word != 0) {
                const std::size_t bit = std::countr_zero(word);
                if (k == 0) {
                    return word_index * 64 + bit;
                }
                --k;
                word &= word - 1;
            }
        }
        return n.bit_count;
    }

    std::size_t select0_leaf(const node& n, std::size_t k) const {
        if (n.kind == dynamic_leaf) {
            const std::size_t words = word_count_for(n.bit_count);
            for (std::size_t word_index = 0; word_index < words; ++word_index) {
                std::uint64_t word = ~n.words[word_index];
                if (word_index + 1 == words) {
                    word &= tail_mask(n.bit_count);
                }
                const std::size_t zeros = std::popcount(word);
                if (k >= zeros) {
                    k -= zeros;
                    continue;
                }

                while (word != 0) {
                    const std::size_t bit = std::countr_zero(word);
                    if (k == 0) {
                        return word_index * 64 + bit;
                    }
                    --k;
                    word &= word - 1;
                }
            }
            return n.bit_count;
        }

        return n.static_bits.select0(k);
    }

    std::size_t child_for_position(const node& n, std::size_t i) const {
        std::size_t prefix = 0;
        for (std::size_t child = 0; child < n.children.size(); ++child) {
            const std::size_t next = prefix + n.children[child]->bit_count;
            if (i < next) {
                return child;
            }
            prefix = next;
        }
        return n.children.empty() ? 0 : n.children.size() - 1;
    }

    std::pair<std::size_t, std::size_t> child_for_insert_position(const node& n, std::size_t i) const {
        if (n.children.empty()) {
            return {0, 0};
        }

        std::size_t prefix = 0;
        for (std::size_t child = 0; child < n.children.size(); ++child) {
            const std::size_t child_bits = n.children[child]->bit_count;
            if (i <= prefix + child_bits) {
                return {child, i - prefix};
            }
            prefix += child_bits;
        }

        const std::size_t child = n.children.size() - 1;
        return {child, n.children[child]->bit_count};
    }

    std::pair<std::size_t, std::size_t> child_for_one(const node& n, std::size_t k) const {
        std::size_t skipped_bits = 0;
        std::size_t skipped_ones = 0;
        for (std::size_t child = 0; child < n.children.size(); ++child) {
            const auto& c = n.children[child];
            if (k < skipped_ones + c->one_count) {
                return {child, skipped_bits};
            }
            skipped_bits += c->bit_count;
            skipped_ones += c->one_count;
        }
        return {n.children.size(), skipped_bits};
    }

    std::pair<std::size_t, std::size_t> child_for_zero(const node& n, std::size_t k) const {
        std::size_t skipped_bits = 0;
        std::size_t skipped_zeros = 0;
        for (std::size_t child = 0; child < n.children.size(); ++child) {
            const auto& c = n.children[child];
            const std::size_t child_zeros = c->bit_count - c->one_count;
            if (k < skipped_zeros + child_zeros) {
                return {child, skipped_bits};
            }
            skipped_bits += c->bit_count;
            skipped_zeros += child_zeros;
        }
        return {n.children.size(), skipped_bits};
    }

    bool access_impl(std::unique_ptr<node>& slot, std::size_t i) const {
        if (slot->kind != internal_node) {
            return access_leaf(*slot, i);
        }

        if (maybe_flatten_after_query(slot)) {
            return access_leaf(*slot, i);
        }

        const std::size_t child = child_for_position(*slot, i);
        std::size_t prefix = 0;
        for (std::size_t c = 0; c < child; ++c) {
            prefix += slot->children[c]->bit_count;
        }
        return access_impl(slot->children[child], i - prefix);
    }

    std::size_t rank1_impl(std::unique_ptr<node>& slot, std::size_t i) const {
        if (slot->kind != internal_node) {
            return rank1_leaf(*slot, i);
        }

        if (maybe_flatten_after_query(slot)) {
            return rank1_leaf(*slot, i);
        }

        std::size_t remaining = i;
        std::size_t result = 0;
        for (auto& child : slot->children) {
            if (remaining >= child->bit_count) {
                result += child->one_count;
                remaining -= child->bit_count;
            } else {
                return result + rank1_impl(child, remaining);
            }
        }
        return result;
    }

    std::size_t select1_impl(std::unique_ptr<node>& slot, std::size_t k) const {
        if (slot->kind != internal_node) {
            return select1_leaf(*slot, k);
        }

        if (maybe_flatten_after_query(slot)) {
            return select1_leaf(*slot, k);
        }

        const auto [child, skipped_bits] = child_for_one(*slot, k);
        if (child >= slot->children.size()) {
            return slot->bit_count;
        }

        std::size_t skipped_ones = 0;
        for (std::size_t c = 0; c < child; ++c) {
            skipped_ones += slot->children[c]->one_count;
        }
        return skipped_bits + select1_impl(slot->children[child], k - skipped_ones);
    }

    std::size_t select0_impl(std::unique_ptr<node>& slot, std::size_t k) const {
        if (slot->kind != internal_node) {
            return select0_leaf(*slot, k);
        }

        if (maybe_flatten_after_query(slot)) {
            return select0_leaf(*slot, k);
        }

        const auto [child, skipped_bits] = child_for_zero(*slot, k);
        if (child >= slot->children.size()) {
            return slot->bit_count;
        }

        std::size_t skipped_zeros = 0;
        for (std::size_t c = 0; c < child; ++c) {
            skipped_zeros += slot->children[c]->bit_count - slot->children[c]->one_count;
        }
        return skipped_bits + select0_impl(slot->children[child], k - skipped_zeros);
    }

    void replace_children(
        node& parent,
        std::size_t start,
        std::size_t count,
        std::vector<std::unique_ptr<node>> replacement) const {
        std::vector<std::unique_ptr<node>> next;
        next.reserve(parent.children.size() - count + replacement.size());

        for (std::size_t i = 0; i < start; ++i) {
            next.push_back(std::move(parent.children[i]));
        }
        for (auto& child : replacement) {
            next.push_back(std::move(child));
        }
        for (std::size_t i = start + count; i < parent.children.size(); ++i) {
            next.push_back(std::move(parent.children[i]));
        }

        parent.children = std::move(next);
        recompute_metadata(parent);
    }

    void split_child(node& parent, std::size_t child_index) const {
        auto bits = collect_bits_copy(*parent.children[child_index]);
        if (bits.size() <= 1) {
            return;
        }

        const std::size_t level = parent.children[child_index]->level;
        const std::size_t mid = bits.size() / 2;
        std::vector<std::unique_ptr<node>> replacement;
        replacement.push_back(build_subtree_exact_level(bits, 0, mid, level));
        replacement.push_back(build_subtree_exact_level(bits, mid, bits.size() - mid, level));
        replace_children(parent, child_index, 1, std::move(replacement));
    }

    void repair_underfull_child(node& parent, std::size_t child_index) const {
        if (parent.children.empty() || child_index >= parent.children.size()) {
            recompute_metadata(parent);
            return;
        }

        auto& child = parent.children[child_index];
        if (parent.children.size() == 1 || child->bit_count >= min_weight(child->level)) {
            recompute_metadata(parent);
            return;
        }

        const std::size_t sibling_index = child_index > 0 ? child_index - 1 : child_index + 1;
        const std::size_t start = std::min(child_index, sibling_index);
        const std::size_t level = parent.children[start]->level;

        std::vector<unsigned char> bits;
        bits.reserve(parent.children[start]->bit_count + parent.children[start + 1]->bit_count);
        collect_bits(*parent.children[start], bits);
        collect_bits(*parent.children[start + 1], bits);

        std::vector<std::unique_ptr<node>> replacement;
        if (bits.size() <= max_weight(level) || bits.size() <= 1) {
            replacement.push_back(build_subtree_exact_level(bits, 0, bits.size(), level));
        } else {
            const std::size_t mid = bits.size() / 2;
            replacement.push_back(build_subtree_exact_level(bits, 0, mid, level));
            replacement.push_back(build_subtree_exact_level(bits, mid, bits.size() - mid, level));
        }

        replace_children(parent, start, 2, std::move(replacement));
    }

    void repair_child_after_update(node& parent, std::size_t child_index) const {
        if (parent.children.empty() || child_index >= parent.children.size()) {
            recompute_metadata(parent);
            return;
        }

        if (parent.children[child_index]->bit_count > max_weight(parent.children[child_index]->level)) {
            split_child(parent, child_index);
        }

        if (child_index < parent.children.size()) {
            repair_underfull_child(parent, child_index);
        } else if (!parent.children.empty()) {
            repair_underfull_child(parent, parent.children.size() - 1);
        } else {
            recompute_metadata(parent);
        }

        if (parent.children.size() > max_children() && parent.bit_count > 0) {
            const std::size_t level = parent.level;
            auto bits = collect_bits_copy(parent);
            auto rebuilt = build_subtree_exact_level(bits, 0, bits.size(), level);
            parent.kind = rebuilt->kind;
            parent.level = rebuilt->level;
            parent.bit_count = rebuilt->bit_count;
            parent.one_count = rebuilt->one_count;
            parent.queries = rebuilt->queries;
            parent.children = std::move(rebuilt->children);
            parent.words = std::move(rebuilt->words);
            parent.static_bits = std::move(rebuilt->static_bits);
        } else {
            recompute_metadata(parent);
        }
    }

    void insert_impl(std::unique_ptr<node>& slot, std::size_t i, bool bit) const {
        if (slot->kind == static_leaf) {
            materialize_static_leaf(slot);
        }

        if (slot->kind == dynamic_leaf) {
            slot->queries = 0;
            const std::size_t old_size = slot->bit_count;
            ++slot->bit_count;
            slot->words.resize(word_count_for(slot->bit_count), 0);
            for (std::size_t pos = old_size; pos > i; --pos) {
                dynamic_set_bit(*slot, pos, dynamic_get_bit(*slot, pos - 1));
            }
            dynamic_set_bit(*slot, i, bit);
            mask_unused_tail_bits(*slot);
            recompute_metadata(*slot);
            return;
        }

        slot->queries = 0;
        if (slot->children.empty()) {
            std::vector<unsigned char> bits{static_cast<unsigned char>(bit)};
            slot->children.push_back(build_subtree_exact_level(bits, 0, 1, slot->level > 0 ? slot->level - 1 : 0));
            recompute_metadata(*slot);
            return;
        }

        auto [child, local] = child_for_insert_position(*slot, i);
        child = std::min(child, slot->children.size() - 1);
        local = std::min(local, slot->children[child]->bit_count);
        insert_impl(slot->children[child], local, bit);
        repair_child_after_update(*slot, child);
    }

    void erase_impl(std::unique_ptr<node>& slot, std::size_t i) const {
        if (slot->kind == static_leaf) {
            materialize_static_leaf(slot);
        }

        if (slot->kind == dynamic_leaf) {
            slot->queries = 0;
            const std::size_t old_size = slot->bit_count;
            for (std::size_t pos = i; pos + 1 < old_size; ++pos) {
                dynamic_set_bit(*slot, pos, dynamic_get_bit(*slot, pos + 1));
            }
            --slot->bit_count;
            mask_unused_tail_bits(*slot);
            recompute_metadata(*slot);
            return;
        }

        slot->queries = 0;
        const std::size_t child = child_for_position(*slot, i);
        std::size_t prefix = 0;
        for (std::size_t c = 0; c < child; ++c) {
            prefix += slot->children[c]->bit_count;
        }

        erase_impl(slot->children[child], i - prefix);
        repair_child_after_update(*slot, child);
    }

    void set_impl(std::unique_ptr<node>& slot, std::size_t i, bool bit) const {
        if (slot->kind == static_leaf) {
            materialize_static_leaf(slot);
        }

        if (slot->kind == dynamic_leaf) {
            slot->queries = 0;
            dynamic_set_bit(*slot, i, bit);
            recompute_metadata(*slot);
            return;
        }

        slot->queries = 0;
        const std::size_t child = child_for_position(*slot, i);
        std::size_t prefix = 0;
        for (std::size_t c = 0; c < child; ++c) {
            prefix += slot->children[c]->bit_count;
        }

        set_impl(slot->children[child], i - prefix, bit);
        repair_child_after_update(*slot, child);
    }

    void flip_impl(std::unique_ptr<node>& slot, std::size_t i) const {
        if (slot->kind == static_leaf) {
            materialize_static_leaf(slot);
        }

        if (slot->kind == dynamic_leaf) {
            slot->queries = 0;
            dynamic_set_bit(*slot, i, !dynamic_get_bit(*slot, i));
            recompute_metadata(*slot);
            return;
        }

        slot->queries = 0;
        const std::size_t child = child_for_position(*slot, i);
        std::size_t prefix = 0;
        for (std::size_t c = 0; c < child; ++c) {
            prefix += slot->children[c]->bit_count;
        }

        flip_impl(slot->children[child], i - prefix);
        repair_child_after_update(*slot, child);
    }

    void normalize_root() {
        ensure_root();
        recompute_metadata(*root);

        if (root->bit_count == 0) {
            root = make_dynamic_leaf({}, 0);
            return;
        }

        while (root->kind == internal_node && root->children.size() == 1) {
            root = std::move(root->children[0]);
            recompute_metadata(*root);
        }

        if (root->kind == internal_node && root->bit_count <= leaf_capacity) {
            auto bits = collect_bits_copy(*root);
            root = make_dynamic_leaf(bits, 0);
            return;
        }

        if (root->bit_count > max_weight(root->level) ||
            (root->kind == internal_node && root->children.size() > max_children())) {
            auto bits = collect_bits_copy(*root);
            root = build_tree_from_bits(bits);
        }

        recompute_metadata(*root);
    }

    void finish_update() {
        normalize_root();
        total_bits = root ? root->bit_count : 0;
    }

    static std::size_t count_kind(const node* n, node_kind kind) {
        if (n == nullptr) {
            return 0;
        }

        std::size_t count = n->kind == kind ? 1 : 0;
        for (const auto& child : n->children) {
            count += count_kind(child.get(), kind);
        }
        return count;
    }

    static invariant_summary check_node_invariants(const node* n) {
        if (n == nullptr) {
            return {false, 0, 0};
        }

        if (n->kind == dynamic_leaf) {
            const bool tail_ok = n->words.empty() || ((n->words.back() & ~tail_mask(n->bit_count)) == 0);
            const bool ok = n->children.empty() &&
                            n->static_bits.size() == 0 &&
                            n->words.size() == word_count_for(n->bit_count) &&
                            n->one_count == dynamic_popcount(*n) &&
                            tail_ok;
            return {ok, n->bit_count, n->one_count};
        }

        if (n->kind == static_leaf) {
            const bool ok = n->children.empty() &&
                            n->words.empty() &&
                            n->static_bits.size() == n->bit_count &&
                            n->static_bits.ones() == n->one_count;
            return {ok, n->bit_count, n->one_count};
        }

        std::size_t bits = 0;
        std::size_t ones = 0;
        bool ok = n->words.empty() && n->static_bits.size() == 0;
        for (const auto& child : n->children) {
            const auto child_summary = check_node_invariants(child.get());
            ok = ok && child_summary.ok;
            if (child && n->level > 0) {
                ok = ok && child->level + 1 == n->level;
            }
            bits += child_summary.bits;
            ones += child_summary.ones;
        }

        ok = ok && bits == n->bit_count && ones == n->one_count;
        return {ok, bits, ones};
    }
};

} // namespace Navarro25
