#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace BTreeDBV {

using namespace std;

template <size_t Fanout = 64, size_t LeafCapacityBits = (size_t{1} << 14)>
class BasicDynamicBitVector {
    static constexpr size_t W = 64;
    static constexpr size_t LEAF_WORDS = (LeafCapacityBits + W) / W;
    static constexpr size_t MIN_LEAF_BITS = (LeafCapacityBits + 1) / 2;
    static constexpr size_t MIN_CHILDREN = (Fanout + 1) / 2;

    enum class Kind { leaf, internal };

    struct Node {
        explicit Node(Kind k) : kind(k) {}
        virtual ~Node() = default;
        Kind kind;
        size_t bits = 0;
        size_t ones = 0;
    };

    struct Leaf : Node {
        Leaf() : Node(Kind::leaf) {}
        array<uint64_t, LEAF_WORDS> words{};
    };

    struct Internal : Node {
        Internal() : Node(Kind::internal) {}
        vector<unique_ptr<Node>> child;
        vector<size_t> bits_pref;
        vector<size_t> ones_pref;
    };

    struct Summary {
        bool ok = true;
        size_t bits = 0;
        size_t ones = 0;
        size_t depth = 0;
    };

public:
    static constexpr size_t fanout = Fanout;
    static constexpr size_t leaf_capacity_bits = LeafCapacityBits;

    BasicDynamicBitVector() : root_(make_unique<Leaf>()) {}

    size_t size() const noexcept { return root_->bits; }

    size_t count_bits() const noexcept {
        if (root_->bits == 0) {
            return 0;
        }

        const size_t total_width = bits_for_value(root_->bits);
        return count_bits(*root_, total_width);
    }

    bool access(size_t i) const {
        assert(i < size());
        Node* x = root_.get();
        while (x->kind == Kind::internal) {
            auto& in = *static_cast<Internal*>(x);
            size_t c = pick(in.bits_pref, i);
            i -= in.bits_pref[c];
            x = in.child[c].get();
        }
        return get(*static_cast<Leaf*>(x), i);
    }

    size_t rank1(size_t i) const {
        assert(i <= size());
        Node* x = root_.get();
        size_t ans = 0;
        while (x->kind == Kind::internal) {
            auto& in = *static_cast<Internal*>(x);
            if (i == in.bits) return ans + in.ones;
            size_t c = pick(in.bits_pref, i);
            ans += in.ones_pref[c];
            i -= in.bits_pref[c];
            x = in.child[c].get();
        }
        return ans + leaf_rank(*static_cast<Leaf*>(x), i);
    }

    size_t rank0(size_t i) const {
        assert(i <= size());
        return i - rank1(i);
    }

    size_t select1(size_t k) const noexcept {
        if (k >= root_->ones) return size();
        Node* x = root_.get();
        size_t ans = 0;
        while (x->kind == Kind::internal) {
            auto& in = *static_cast<Internal*>(x);
            size_t c = pick(in.ones_pref, k);
            ans += in.bits_pref[c];
            k -= in.ones_pref[c];
            x = in.child[c].get();
        }
        return ans + leaf_select(*static_cast<Leaf*>(x), k);
    }

    void set(size_t i, bool bit) {
        assert(i < size());
        vector<Internal*> path;
        Node* x = root_.get();
        while (x->kind == Kind::internal) {
            auto* in = static_cast<Internal*>(x);
            path.push_back(in);
            size_t c = pick(in->bits_pref, i);
            i -= in->bits_pref[c];
            x = in->child[c].get();
        }

        auto& leaf = *static_cast<Leaf*>(x);
        bool old = get(leaf, i);
        if (old == bit) return;
        put(leaf, i, bit);
        leaf.ones += bit ? 1 : -1;
        for (auto it = path.rbegin(); it != path.rend(); ++it) rebuild(**it);
    }

    void flip(size_t i) {
        assert(i < size());
        vector<Internal*> path;
        Node* x = root_.get();
        while (x->kind == Kind::internal) {
            auto* in = static_cast<Internal*>(x);
            path.push_back(in);
            size_t c = pick(in->bits_pref, i);
            i -= in->bits_pref[c];
            x = in->child[c].get();
        }

        auto& leaf = *static_cast<Leaf*>(x);
        bool old = get(leaf, i);
        put(leaf, i, !old);
        leaf.ones += old ? -1 : 1;
        for (auto it = path.rbegin(); it != path.rend(); ++it) rebuild(**it);
    }

    void insert(size_t i, bool bit) {
        assert(i <= size());
        vector<pair<Internal*, size_t>> path;
        Node* x = root_.get();
        while (x->kind == Kind::internal) {
            auto* in = static_cast<Internal*>(x);
            size_t c = pick(in->bits_pref, i);
            path.push_back({in, c});
            i -= in->bits_pref[c];
            x = in->child[c].get();
        }

        auto& leaf = *static_cast<Leaf*>(x);
        leaf_insert(leaf, i, bit);

        unique_ptr<Node> extra;
        if (leaf.bits > LeafCapacityBits) {
            Leaf snap = leaf;
            auto refill = [&](Leaf& out, size_t from, size_t len) {
                out.words.fill(0);
                out.bits = len;
                out.ones = 0;
                for (size_t j = 0; j < len; ++j) {
                    bool b = get(snap, from + j);
                    put(out, j, b);
                    out.ones += b;
                }
                trim(out);
            };
            size_t left = snap.bits / 2;
            auto right = make_unique<Leaf>();
            refill(leaf, 0, left);
            refill(*right, left, snap.bits - left);
            extra = move(right);
        }

        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            auto* in = it->first;
            size_t c = it->second;
            if (extra) in->child.insert(in->child.begin() + static_cast<ptrdiff_t>(c + 1), move(extra));
            rebuild(*in);
            if (in->child.size() <= Fanout) continue;

            auto right = make_unique<Internal>();
            size_t mid = in->child.size() / 2;
            right->child.insert(
                right->child.end(),
                make_move_iterator(in->child.begin() + static_cast<ptrdiff_t>(mid)),
                make_move_iterator(in->child.end()));
            in->child.erase(in->child.begin() + static_cast<ptrdiff_t>(mid), in->child.end());
            rebuild(*in);
            rebuild(*right);
            extra = move(right);
        }

        if (extra) {
            auto r = move(root_);
            auto nr = make_unique<Internal>();
            nr->child.push_back(move(r));
            nr->child.push_back(move(extra));
            rebuild(*nr);
            root_ = move(nr);
        }
    }

    void erase(size_t i) {
        assert(i < size());
        vector<pair<Internal*, size_t>> path;
        Node* x = root_.get();
        while (x->kind == Kind::internal) {
            auto* in = static_cast<Internal*>(x);
            size_t c = pick(in->bits_pref, i);
            path.push_back({in, c});
            i -= in->bits_pref[c];
            x = in->child[c].get();
        }

        leaf_erase(*static_cast<Leaf*>(x), i);

        auto small = [](const Node& n, bool root) {
            if (root) return false;
            if (n.kind == Kind::leaf) return n.bits < MIN_LEAF_BITS;
            return static_cast<const Internal&>(n).child.size() < MIN_CHILDREN;
        };
        auto can_lend = [](const Node& n) {
            if (n.kind == Kind::leaf) return n.bits > MIN_LEAF_BITS;
            return static_cast<const Internal&>(n).child.size() > MIN_CHILDREN;
        };
        auto merge = [&](Node& a, Node& b) {
            if (a.kind == Kind::leaf) {
                auto& x = static_cast<Leaf&>(a);
                auto& y = static_cast<Leaf&>(b);
                size_t base = x.bits;
                for (size_t j = 0; j < y.bits; ++j) put(x, base + j, get(y, j));
                x.bits += y.bits;
                x.ones += y.ones;
                trim(x);
            } 
            else {
                auto& x = static_cast<Internal&>(a);
                auto& y = static_cast<Internal&>(b);
                for (auto& ch : y.child) x.child.push_back(move(ch));
                y.child.clear();
                rebuild(x);
            }
        };

        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            auto* in = it->first;
            size_t c = it->second;

            if (c < in->child.size() && small(*in->child[c], false)) {
                if (c > 0 && can_lend(*in->child[c - 1])) {
                    if (in->child[c]->kind == Kind::leaf) {
                        auto& a = *static_cast<Leaf*>(in->child[c - 1].get());
                        auto& b = *static_cast<Leaf*>(in->child[c].get());
                        bool v = get(a, a.bits - 1);
                        leaf_erase(a, a.bits - 1);
                        leaf_insert(b, 0, v);
                    } else {
                        auto& a = *static_cast<Internal*>(in->child[c - 1].get());
                        auto& b = *static_cast<Internal*>(in->child[c].get());
                        auto moved = move(a.child.back());
                        a.child.pop_back();
                        b.child.insert(b.child.begin(), move(moved));
                        rebuild(a);
                        rebuild(b);
                    }
                } else if (c + 1 < in->child.size() && can_lend(*in->child[c + 1])) {
                    if (in->child[c]->kind == Kind::leaf) {
                        auto& a = *static_cast<Leaf*>(in->child[c].get());
                        auto& b = *static_cast<Leaf*>(in->child[c + 1].get());
                        bool v = get(b, 0);
                        leaf_erase(b, 0);
                        leaf_insert(a, a.bits, v);
                    } else {
                        auto& a = *static_cast<Internal*>(in->child[c].get());
                        auto& b = *static_cast<Internal*>(in->child[c + 1].get());
                        auto moved = move(b.child.front());
                        b.child.erase(b.child.begin());
                        a.child.push_back(move(moved));
                        rebuild(a);
                        rebuild(b);
                    }
                } else if (c > 0) {
                    merge(*in->child[c - 1], *in->child[c]);
                    in->child.erase(in->child.begin() + static_cast<ptrdiff_t>(c));
                } else if (c + 1 < in->child.size()) {
                    merge(*in->child[c], *in->child[c + 1]);
                    in->child.erase(in->child.begin() + static_cast<ptrdiff_t>(c + 1));
                }
            }

            rebuild(*in);
        }

        while (root_->kind == Kind::internal) {
            auto& in = *static_cast<Internal*>(root_.get());
            if (in.child.empty()) {
                root_ = make_unique<Leaf>();
                break;
            }
            if (in.child.size() != 1) break;
            root_ = move(in.child.front());
        }
    }

    void clear() { root_ = make_unique<Leaf>(); }

    bool check_invariants() const {
        Summary s = check(*root_, true, 0);
        return s.ok && s.bits == root_->bits && s.ones == root_->ones;
    }

private:
    unique_ptr<Node> root_;

    static uint64_t low_mask(size_t bits) {
        if (bits == 0) return 0;
        if (bits >= W) return ~uint64_t{0};
        return (uint64_t{1} << bits) - 1;
    }

    static size_t word_count(size_t bits) {
        return bits == 0 ? 0 : 1 + (bits - 1) / W;
    }

    static size_t ceil_log2(size_t x) {
        if (x <= 1) return 0;
        --x;
        size_t bits = 0;
        while (x != 0) {
            ++bits;
            x >>= 1;
        }
        return bits;
    }

    static size_t bits_for_value(size_t max_value) {
        return max<size_t>(1, ceil_log2(max_value + 1));
    }

    static bool get(const Leaf& x, size_t i) {
        return ((x.words[i / W] >> (i % W)) & uint64_t{1}) != 0;
    }

    static void put(Leaf& x, size_t i, bool bit) {
        auto& w = x.words[i / W];
        auto m = uint64_t{1} << (i % W);
        if (bit) w |= m;
        else w &= ~m;
    }

    static void trim(Leaf& x) {
        size_t wc = word_count(x.bits);
        if (wc == 0) {
            x.words.fill(0);
            return;
        }
        if (x.bits % W) x.words[wc - 1] &= low_mask(x.bits % W);
        for (size_t i = wc; i < LEAF_WORDS; ++i) x.words[i] = 0;
    }

    static size_t leaf_rank(const Leaf& x, size_t i) {
        size_t ans = 0;
        for (size_t w = 0; w < i / W; ++w) ans += popcount(x.words[w]);
        if (i % W) ans += popcount(x.words[i / W] & low_mask(i % W));
        return ans;
    }

    static size_t leaf_select(const Leaf& x, size_t k) {
        size_t wc = word_count(x.bits);
        for (size_t w = 0; w < wc; ++w) {
            uint64_t cur = x.words[w];
            if (w + 1 == wc && (x.bits % W)) cur &= low_mask(x.bits % W);
            size_t pc = popcount(cur);
            if (k < pc) {
                while (k--) cur &= cur - 1;
                return w * W + countr_zero(cur);
            }
            k -= pc;
        }
        return x.bits;
    }

    static void leaf_insert(Leaf& x, size_t pos, bool bit) {
        size_t old = x.bits++;
        for (size_t j = old; j > pos; --j) put(x, j, get(x, j - 1));
        put(x, pos, bit);
        x.ones += bit;
        trim(x);
    }

    static bool leaf_erase(Leaf& x, size_t pos) {
        bool old = get(x, pos);
        for (size_t j = pos; j + 1 < x.bits; ++j) put(x, j, get(x, j + 1));
        --x.bits;
        if (old) --x.ones;
        trim(x);
        return old;
    }

    static void rebuild(Internal& x) {
        x.bits_pref.assign(x.child.size() + 1, 0);
        x.ones_pref.assign(x.child.size() + 1, 0);
        for (size_t i = 0; i < x.child.size(); ++i) {
            x.bits_pref[i + 1] = x.bits_pref[i] + x.child[i]->bits;
            x.ones_pref[i + 1] = x.ones_pref[i] + x.child[i]->ones;
        }
        x.bits = x.bits_pref.back();
        x.ones = x.ones_pref.back();
    }

    static size_t pick(const vector<size_t>& pref, size_t x) {
        auto it = upper_bound(pref.begin() + 1, pref.end(), x);
        return it == pref.end() ? pref.size() - 2 : static_cast<size_t>(it - pref.begin() - 1);
    }

    size_t count_bits(const Node& x, size_t total_width) const {
        size_t bits = 1 + 2 * total_width;
        if (x.kind == Kind::leaf) {
            return bits + x.bits;
        }

        const auto& in = static_cast<const Internal&>(x);
        bits += in.bits_pref.size() * total_width;
        bits += in.ones_pref.size() * total_width;
        for (const auto& child : in.child) {
            bits += count_bits(*child, total_width);
        }
        return bits;
    }

    Summary check(const Node& x, bool root, size_t depth) const {
        if (x.kind == Kind::leaf) {
            auto& leaf = static_cast<const Leaf&>(x);
            bool ok = x.ones <= x.bits && x.bits <= LeafCapacityBits;
            if (!root) ok = ok && x.bits >= MIN_LEAF_BITS;
            ok = ok && leaf_rank(leaf, x.bits) == x.ones;
            return {ok, x.bits, x.ones, depth};
        }

        auto& in = static_cast<const Internal&>(x);
        bool ok = !in.child.empty() && in.child.size() <= Fanout;
        if (root) ok = ok && in.child.size() >= 2;
        else ok = ok && in.child.size() >= MIN_CHILDREN;
        ok = ok && in.bits_pref.size() == in.child.size() + 1 && in.ones_pref.size() == in.child.size() + 1;
        ok = ok && !in.bits_pref.empty() && in.bits_pref[0] == 0 && in.ones_pref[0] == 0;

        size_t bits = 0, ones = 0, d = 0;
        bool seen = false;
        for (size_t i = 0; i < in.child.size(); ++i) {
            ok = ok && in.bits_pref[i] == bits && in.ones_pref[i] == ones;
            Summary cur = check(*in.child[i], false, depth + 1);
            ok = ok && cur.ok;
            bits += cur.bits;
            ones += cur.ones;
            if (!seen) d = cur.depth, seen = true;
            else ok = ok && d == cur.depth;
        }
        ok = ok && in.bits_pref.back() == bits && in.ones_pref.back() == ones;
        ok = ok && x.bits == bits && x.ones == ones;
        return {ok, bits, ones, d};
    }
};

using Dcc22StyleDynamicBitVector = BasicDynamicBitVector<64, (size_t{1} << 14)>;
using PrezzaStyleDynamicBitVector = BasicDynamicBitVector<16, 8192>;
using DynamicBitVector = Dcc22StyleDynamicBitVector;

} // namespace BTreeDBV
