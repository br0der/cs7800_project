#pragma once

#include "dynamic_bitvector_concept.hpp"
#include "reference_model.hpp"

#include <cstdint>
#include <cstddef>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

namespace dbv {

struct TestConfig {
    size_t rounds = 5;
    size_t operations_per_round = 5000;
    size_t q = 1; // expected query/update ratio (q queries per 1 update)
    uint64_t seed = 0xC57800ULL;
    bool verbose = true;
};

template <typename BV>
void load_from_bits(BV& bv, const vector<bool>& bits) {
    static_assert(is_dynamic_bitvector_v<BV>,
                  "Non-dynamic bitvectors must provide dbv::load_from_bits(BV&, const vector<bool>&)");

    bv.clear();
    for (bool b : bits) {
        bv.insert(bv.size(), b);
    }
}

inline void load_oracle_from_bits(ReferenceBitVector& oracle, const vector<bool>& bits) {
    oracle.clear();
    for (bool b : bits) {
        oracle.insert(oracle.size(), b);
    }
}

inline void check(bool ok, const string& msg) {
    if (!ok) {
        throw runtime_error(msg);
    }
}

template <typename BV>
inline void run_random_queries_only(const BV& bv, const ReferenceBitVector& oracle, mt19937_64& rng, size_t count) {
    const size_t n = oracle.size();

    for (size_t i = 0; i < count; ++i) {
        if (n == 0) {
            bernoulli_distribution rank_or_select(0.5);
            if (rank_or_select(rng)) {
                const auto a = bv.rank1(0);
                const auto b = oracle.rank1(0);
                check(a == b, "rank1 query mismatch (empty)");
            } else {
                const auto a = bv.select1(0);
                const auto b = oracle.select1(0);
                check(a == b, "select1 query mismatch (empty)");
            }
            continue;
        }

        uniform_int_distribution<int> query_dist(4, 6);
        const int op = query_dist(rng);

        switch (op) {
            case 4: { // access query
                uniform_int_distribution<size_t> pos_dist(0, n - 1);
                const auto pos = pos_dist(rng);
                const bool a = bv.access(pos);
                const bool b = oracle.access(pos);
                check(a == b, "access query mismatch");
                break;
            }
            case 5: { // rank query
                uniform_int_distribution<size_t> pos_dist(0, n);
                const auto pos = pos_dist(rng);
                const auto a = bv.rank1(pos);
                const auto b = oracle.rank1(pos);
                check(a == b, "rank1 query mismatch");
                break;
            }
            case 6: { // select query
                const auto ones = oracle.rank1(n);
                uniform_int_distribution<size_t> k_dist(0, ones + 2);
                const auto k = k_dist(rng);
                const auto a = bv.select1(k);
                const auto b = oracle.select1(k);
                check(a == b, "select1 query mismatch");
                break;
            }
            default:
                throw logic_error("invalid query operation");
        }
    }
}

template <typename BV>
void assert_same_state(const BV& candidate, const ReferenceBitVector& oracle, const string& context) {
    ostringstream err;

    if (candidate.size() != oracle.size()) {
        err << context << " | size mismatch candidate=" << candidate.size() << " oracle=" << oracle.size();
        throw runtime_error(err.str());
    }

    const size_t n = oracle.size();

    for (size_t i = 0; i < n; ++i) {
        const bool a = candidate.access(i);
        const bool b = oracle.access(i);
        if (a != b) {
            err.str("");
            err << context << " | access mismatch at i=" << i << " candidate=" << a << " oracle=" << b;
            throw runtime_error(err.str());
        }
    }

    for (size_t i = 0; i <= n; ++i) {
        const auto r1 = candidate.rank1(i);
        const auto r2 = oracle.rank1(i);
        if (r1 != r2) {
            err.str("");
            err << context << " | rank1 mismatch at i=" << i << " candidate=" << r1 << " oracle=" << r2;
            throw runtime_error(err.str());
        }
    }

    const size_t ones = oracle.rank1(n);
    for (size_t k = 0; k < ones + 3; ++k) {
        const auto s1 = candidate.select1(k);
        const auto s2 = oracle.select1(k);
        if (s1 != s2) {
            err.str("");
            err << context << " | select1 mismatch at k=" << k << " candidate=" << s1 << " oracle=" << s2;
            throw runtime_error(err.str());
        }
    }
}

template <typename BV>
void deterministic_tests() {
    static_assert(is_dynamic_bitvector_v<BV>,
                  "deterministic_tests() requires a dynamic bitvector implementation");

    BV bv;
    ReferenceBitVector oracle;

    for (bool b : {false, true, true, false, true}) {
        bv.insert(bv.size(), b);
        oracle.insert(oracle.size(), b);
    }
    assert_same_state(bv, oracle, "after append sequence");

    bv.insert(0, true);
    oracle.insert(0, true);
    bv.insert(3, false);
    oracle.insert(3, false);
    assert_same_state(bv, oracle, "after middle/begin inserts");

    bv.set(2, true);
    oracle.set(2, true);
    bv.flip(4);
    oracle.flip(4);
    assert_same_state(bv, oracle, "after set/flip");

    bv.erase(0);
    oracle.erase(0);
    bv.erase(bv.size() - 1);
    oracle.erase(oracle.size() - 1);
    assert_same_state(bv, oracle, "after erase boundary");

    bv.clear();
    oracle.clear();
    assert_same_state(bv, oracle, "after clear");

    check(bv.select1(0) == 0, "select1 on empty should return size() sentinel");
}

template <typename BV>
void randomized_differential_tests(const TestConfig& cfg) {
    mt19937_64 rng(cfg.seed);

    if constexpr (!is_dynamic_bitvector_v<BV>) {
        for (size_t round = 0; round < cfg.rounds; ++round) {
            BV bv;
            ReferenceBitVector oracle;

            vector<bool> bits;
            bits.reserve(cfg.operations_per_round / 2 + 1);

            bernoulli_distribution bit_dist(0.5);
            const size_t n = cfg.operations_per_round / 2 + 1;
            for (size_t i = 0; i < n; ++i) {
                bits.push_back(bit_dist(rng));
            }

            load_from_bits(bv, bits);
            load_oracle_from_bits(oracle, bits);

            ostringstream context;
            context << "query-only round=" << round << " after build";
            assert_same_state(bv, oracle, context.str());

            run_random_queries_only(bv, oracle, rng, cfg.operations_per_round);

            context.str("");
            context << "query-only round=" << round << " after queries";
            assert_same_state(bv, oracle, context.str());

            if (cfg.verbose) {
                cout << "[ok] query-only round " << (round + 1) << "/" << cfg.rounds << '\n';
            }
        }
        return;
    }

    for (size_t round = 0; round < cfg.rounds; ++round) {
        BV bv;
        ReferenceBitVector oracle;

        for (size_t op_index = 0; op_index < cfg.operations_per_round; ++op_index) {
            const size_t n = oracle.size();
            const bool can_delete_or_update = n > 0;

            double query_probability = (double)cfg.q / (cfg.q + 1);
            bernoulli_distribution choose_query(query_probability);

            int op = -1;
            const bool should_query = choose_query(rng);

            if (should_query) {
                // queries: access, rank, select
                if (n == 0) {
                    // access is invalid on empty; use rank/select
                    bernoulli_distribution rank_or_select(0.5);
                    op = rank_or_select(rng) ? 5 : 6;
                } else {
                    uniform_int_distribution<int> query_dist(4, 6);
                    op = query_dist(rng);
                }
            } else {
                // updates: insert, erase, set, flip
                if (!can_delete_or_update) {
                    op = 0; // only insert possible
                } else {
                    uniform_int_distribution<int> update_dist(0, 3);
                    op = update_dist(rng);
                }
            }

            switch (op) {
                case 0: { // insert
                    uniform_int_distribution<size_t> pos_dist(0, n);
                    bernoulli_distribution bit_dist(0.5);
                    const auto pos = pos_dist(rng);
                    const bool bit = bit_dist(rng);
                    bv.insert(pos, bit);
                    oracle.insert(pos, bit);
                    break;
                }
                case 1: { // erase
                    uniform_int_distribution<size_t> pos_dist(0, n - 1);
                    const auto pos = pos_dist(rng);
                    bv.erase(pos);
                    oracle.erase(pos);
                    break;
                }
                case 2: { // set
                    uniform_int_distribution<size_t> pos_dist(0, n - 1);
                    bernoulli_distribution bit_dist(0.5);
                    const auto pos = pos_dist(rng);
                    const bool bit = bit_dist(rng);
                    bv.set(pos, bit);
                    oracle.set(pos, bit);
                    break;
                }
                case 3: { // flip
                    uniform_int_distribution<size_t> pos_dist(0, n - 1);
                    const auto pos = pos_dist(rng);
                    bv.flip(pos);
                    oracle.flip(pos);
                    break;
                }
                case 4: { // access query
                    uniform_int_distribution<size_t> pos_dist(0, n - 1);
                    const auto pos = pos_dist(rng);
                    const bool a = bv.access(pos);
                    const bool b = oracle.access(pos);
                    check(a == b, "access query mismatch");
                    break;
                }
                case 5: { // rank query
                    uniform_int_distribution<size_t> pos_dist(0, n);
                    const auto pos = pos_dist(rng);
                    const auto a = bv.rank1(pos);
                    const auto b = oracle.rank1(pos);
                    check(a == b, "rank1 query mismatch");
                    break;
                }
                case 6: { // select query
                    const auto ones = oracle.rank1(n);
                    uniform_int_distribution<size_t> k_dist(0, ones + 2);
                    const auto k = k_dist(rng);
                    const auto a = bv.select1(k);
                    const auto b = oracle.select1(k);
                    check(a == b, "select1 query mismatch");
                    break;
                }
                default:
                    throw logic_error("invalid random operation");
            }

            if (op_index % 127 == 0) {
                ostringstream context;
                context << "round=" << round << " op=" << op_index;
                assert_same_state(bv, oracle, context.str());
            }
        }

        ostringstream context;
        context << "end of round " << round;
        assert_same_state(bv, oracle, context.str());

        if (cfg.verbose) {
            cout << "[ok] round " << (round + 1) << "/" << cfg.rounds << '\n';
        }
    }
}

template <typename BV>
void run_suite(const string& impl_name, const TestConfig& cfg = {}) {
    static_assert(is_query_bitvector_v<BV>,
                  "BV must provide size/access/rank1/select1 to use this harness");

    if (cfg.verbose) {
        cout << "Testing implementation: " << impl_name << '\n';
    }

    if constexpr (is_dynamic_bitvector_v<BV>) {
        deterministic_tests<BV>();
        if (cfg.verbose) {
            cout << "[ok] deterministic dynamic tests\n";
        }
    } else if (cfg.verbose) {
        cout << "[info] non-dynamic bitvector: skipping dynamic deterministic tests\n";
    }

    randomized_differential_tests<BV>(cfg);

    if (cfg.verbose) {
        cout << "[ok] all tests passed for " << impl_name << "\n";
    }
}

} // namespace dbv
