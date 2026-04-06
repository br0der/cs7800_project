#pragma once

#include "dynamic_bitvector_concept.hpp"
#include "reference_model.hpp"

#include <cstdint>
#include <cstddef>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <array>
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

struct PerfConfig {
    size_t rounds = 3;
    size_t initial_size = 1 << 14;
    size_t operations = 200000;
    size_t q = 4; // used only for dynamic bitvectors
    uint64_t seed = 0xC57800ULL ^ 0x9E3779B97F4A7C15ULL;
    bool verbose = true;
};

enum class PerfOp : size_t {
    Insert = 0,
    Erase = 1,
    Set = 2,
    Flip = 3,
    Access = 4,
    Rank = 5,
    Select = 6,
    Count = 7,
};

struct PerfStats {
    array<uint64_t, static_cast<size_t>(PerfOp::Count)> counts{};
    array<long double, static_cast<size_t>(PerfOp::Count)> nanos{};
    uint64_t sink = 0;
};

inline const char* perf_op_name(PerfOp op) {
    switch (op) {
        case PerfOp::Insert: return "insert";
        case PerfOp::Erase: return "erase";
        case PerfOp::Set: return "set";
        case PerfOp::Flip: return "flip";
        case PerfOp::Access: return "access";
        case PerfOp::Rank: return "rank1";
        case PerfOp::Select: return "select1";
        default: return "unknown";
    }
}

inline void perf_record(PerfStats& stats, PerfOp op, const chrono::steady_clock::time_point& start,
                        const chrono::steady_clock::time_point& end) {
    const auto idx = static_cast<size_t>(op);
    stats.counts[idx] += 1;
    stats.nanos[idx] += static_cast<long double>(chrono::duration_cast<chrono::nanoseconds>(end - start).count());
}

template <typename BV>
void initialize_for_perf(BV& bv, const PerfConfig& cfg, mt19937_64& rng) {
    if constexpr (is_dynamic_bitvector_v<BV>) {
        bv.clear();
        bernoulli_distribution bit_dist(0.5);
        for (size_t i = 0; i < cfg.initial_size; ++i) {
            bv.insert(bv.size(), bit_dist(rng));
        }
    } else {
        vector<bool> bits;
        bits.reserve(cfg.initial_size);
        bernoulli_distribution bit_dist(0.5);
        for (size_t i = 0; i < cfg.initial_size; ++i) {
            bits.push_back(bit_dist(rng));
        }
        load_from_bits(bv, bits);
    }
}

template <typename BV>
PerfStats run_perf_round(const PerfConfig& cfg, uint64_t seed) {
    mt19937_64 rng(seed);
    BV bv;
    initialize_for_perf(bv, cfg, rng);

    PerfStats stats;

    for (size_t i = 0; i < cfg.operations; ++i) {
        const size_t n = bv.size();

        if constexpr (is_dynamic_bitvector_v<BV>) {
            const double query_probability = static_cast<double>(cfg.q) / static_cast<double>(cfg.q + 1);
            bernoulli_distribution choose_query(query_probability);
            int op = -1;

            if (choose_query(rng)) {
                if (n == 0) {
                    bernoulli_distribution rank_or_select(0.5);
                    op = rank_or_select(rng) ? 5 : 6;
                } else {
                    uniform_int_distribution<int> query_dist(4, 6);
                    op = query_dist(rng);
                }
            } else {
                if (n == 0) {
                    op = 0;
                } else {
                    uniform_int_distribution<int> update_dist(0, 3);
                    op = update_dist(rng);
                }
            }

            switch (op) {
                case 0: {
                    uniform_int_distribution<size_t> pos_dist(0, n);
                    bernoulli_distribution bit_dist(0.5);
                    const auto t0 = chrono::steady_clock::now();
                    bv.insert(pos_dist(rng), bit_dist(rng));
                    const auto t1 = chrono::steady_clock::now();
                    perf_record(stats, PerfOp::Insert, t0, t1);
                    break;
                }
                case 1: {
                    uniform_int_distribution<size_t> pos_dist(0, n - 1);
                    const auto t0 = chrono::steady_clock::now();
                    bv.erase(pos_dist(rng));
                    const auto t1 = chrono::steady_clock::now();
                    perf_record(stats, PerfOp::Erase, t0, t1);
                    break;
                }
                case 2: {
                    uniform_int_distribution<size_t> pos_dist(0, n - 1);
                    bernoulli_distribution bit_dist(0.5);
                    const auto t0 = chrono::steady_clock::now();
                    bv.set(pos_dist(rng), bit_dist(rng));
                    const auto t1 = chrono::steady_clock::now();
                    perf_record(stats, PerfOp::Set, t0, t1);
                    break;
                }
                case 3: {
                    uniform_int_distribution<size_t> pos_dist(0, n - 1);
                    const auto t0 = chrono::steady_clock::now();
                    bv.flip(pos_dist(rng));
                    const auto t1 = chrono::steady_clock::now();
                    perf_record(stats, PerfOp::Flip, t0, t1);
                    break;
                }
                case 4: {
                    uniform_int_distribution<size_t> pos_dist(0, n - 1);
                    const auto t0 = chrono::steady_clock::now();
                    const bool ans = bv.access(pos_dist(rng));
                    const auto t1 = chrono::steady_clock::now();
                    stats.sink ^= static_cast<uint64_t>(ans);
                    perf_record(stats, PerfOp::Access, t0, t1);
                    break;
                }
                case 5: {
                    uniform_int_distribution<size_t> pos_dist(0, n);
                    const auto t0 = chrono::steady_clock::now();
                    const auto ans = bv.rank1(pos_dist(rng));
                    const auto t1 = chrono::steady_clock::now();
                    stats.sink ^= static_cast<uint64_t>(ans);
                    perf_record(stats, PerfOp::Rank, t0, t1);
                    break;
                }
                case 6: {
                    uniform_int_distribution<size_t> k_dist(0, n + 2);
                    const auto t0 = chrono::steady_clock::now();
                    const auto ans = bv.select1(k_dist(rng));
                    const auto t1 = chrono::steady_clock::now();
                    stats.sink ^= static_cast<uint64_t>(ans);
                    perf_record(stats, PerfOp::Select, t0, t1);
                    break;
                }
                default:
                    throw logic_error("invalid benchmark operation");
            }
        } else {
            const int op = (n == 0)
                               ? (bernoulli_distribution(0.5)(rng) ? 5 : 6)
                               : uniform_int_distribution<int>(4, 6)(rng);

            switch (op) {
                case 4: {
                    uniform_int_distribution<size_t> pos_dist(0, n - 1);
                    const auto t0 = chrono::steady_clock::now();
                    const bool ans = bv.access(pos_dist(rng));
                    const auto t1 = chrono::steady_clock::now();
                    stats.sink ^= static_cast<uint64_t>(ans);
                    perf_record(stats, PerfOp::Access, t0, t1);
                    break;
                }
                case 5: {
                    uniform_int_distribution<size_t> pos_dist(0, n);
                    const auto t0 = chrono::steady_clock::now();
                    const auto ans = bv.rank1(pos_dist(rng));
                    const auto t1 = chrono::steady_clock::now();
                    stats.sink ^= static_cast<uint64_t>(ans);
                    perf_record(stats, PerfOp::Rank, t0, t1);
                    break;
                }
                case 6: {
                    uniform_int_distribution<size_t> k_dist(0, n + 2);
                    const auto t0 = chrono::steady_clock::now();
                    const auto ans = bv.select1(k_dist(rng));
                    const auto t1 = chrono::steady_clock::now();
                    stats.sink ^= static_cast<uint64_t>(ans);
                    perf_record(stats, PerfOp::Select, t0, t1);
                    break;
                }
                default:
                    throw logic_error("invalid benchmark operation");
            }
        }
    }

    return stats;
}

inline void merge_perf_stats(PerfStats& total, const PerfStats& add) {
    for (size_t i = 0; i < static_cast<size_t>(PerfOp::Count); ++i) {
        total.counts[i] += add.counts[i];
        total.nanos[i] += add.nanos[i];
    }
    total.sink ^= add.sink;
}

inline void print_perf_stats(const string& impl_name, const PerfConfig& cfg, const PerfStats& stats) {
    cout << "Performance (speed-only) for " << impl_name << '\n';
    cout << "rounds=" << cfg.rounds
         << " initial_size=" << cfg.initial_size
         << " operations/round=" << cfg.operations
         << " q=" << cfg.q << '\n';
    cout << left << setw(10) << "op"
         << right << setw(14) << "count"
         << setw(16) << "avg ns/op"
         << setw(16) << "Mops/s" << '\n';

    long double total_ns = 0.0L;
    uint64_t total_count = 0;

    for (size_t i = 0; i < static_cast<size_t>(PerfOp::Count); ++i) {
        const uint64_t c = stats.counts[i];
        if (c == 0) {
            continue;
        }

        const long double ns = stats.nanos[i];
        const long double avg_ns = ns / static_cast<long double>(c);
        const long double mops = (ns > 0.0L) ? (static_cast<long double>(c) / (ns / 1.0e9L) / 1.0e6L) : 0.0L;

        cout << left << setw(10) << perf_op_name(static_cast<PerfOp>(i))
             << right << setw(14) << c
             << setw(16) << fixed << setprecision(2) << avg_ns
             << setw(16) << fixed << setprecision(3) << mops << '\n';

        total_ns += ns;
        total_count += c;
    }

    const long double total_avg_ns = (total_count > 0) ? total_ns / static_cast<long double>(total_count) : 0.0L;
    const long double total_mops = (total_ns > 0.0L)
                                       ? (static_cast<long double>(total_count) / (total_ns / 1.0e9L) / 1.0e6L)
                                       : 0.0L;

    cout << left << setw(10) << "total"
         << right << setw(14) << total_count
         << setw(16) << fixed << setprecision(2) << total_avg_ns
         << setw(16) << fixed << setprecision(3) << total_mops << '\n';

    // Prevent compilers from optimizing out query results.
    cout << "benchmark sink=" << stats.sink << '\n';
}

template <typename BV>
void run_performance_suite(const string& impl_name, const PerfConfig& cfg = {}) {
    static_assert(is_query_bitvector_v<BV>,
                  "BV must provide size/access/rank1/select1 to run performance benchmarks");

    PerfStats total;
    for (size_t r = 0; r < cfg.rounds; ++r) {
        const auto round_stats = run_perf_round<BV>(cfg, cfg.seed + static_cast<uint64_t>(r));
        merge_perf_stats(total, round_stats);
    }

    if (cfg.verbose) {
        print_perf_stats(impl_name, cfg, total);
    }
}

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
    } else {
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

                if ((op_index & ((1<<7)-1)) == 0) {
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
