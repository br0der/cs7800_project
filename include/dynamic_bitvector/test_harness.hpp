#pragma once

#include "dynamic_bitvector_concept.hpp"
#include "reference_model.hpp"

#include <chrono>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace std;

namespace dbv {

// Very small, ad-hoc harness by design.
struct TestConfig {
    size_t rounds = 3;
    size_t operations_per_round = 2000;
    uint64_t seed = 0xC57800ULL;
};

struct PerfConfig {
    size_t rounds = 3;
    size_t initial_size = 1 << 23;
    size_t operations = 10000000;
    uint64_t seed = 0x9E3779B97F4A7C15ULL;
};

template <typename BV>
void load_from_bits(BV& bv, const vector<bool>& bits) {
    if constexpr (is_constructible_v<BV, const vector<bool>&>) {
        bv = BV(bits);
    } else {
        static_assert(is_dynamic_bitvector_v<BV>,
                      "For query-only bitvectors, provide dbv::load_from_bits(BV&, const vector<bool>&)");

        bv.clear();
        for (bool b : bits) {
            bv.insert(bv.size(), b);
        }
    }
}

inline void check(bool ok, const string& msg) {
    if (!ok) {
        throw runtime_error(msg);
    }
}

template <typename BV>
void assert_same_state(const BV& bv, const ReferenceBitVector& oracle, const string& context) {
    check(bv.size() == oracle.size(), context + " | size mismatch");

    const size_t n = oracle.size();
    for (size_t i = 0; i < n; ++i) {
        check(bv.access(i) == oracle.access(i), context + " | access mismatch");
    }

    for (size_t i = 0; i <= n; ++i) {
        check(bv.rank1(i) == oracle.rank1(i), context + " | rank mismatch");
    }

    const size_t ones = oracle.rank1(n);
    for (size_t k = 0; k < ones + 3; ++k) {
        check(bv.select1(k) == oracle.select1(k), context + " | select mismatch");
    }
}

template <typename BV>
void run_suite(const string& impl_name, const TestConfig& cfg = {}) {
    static_assert(is_query_bitvector_v<BV>, "BV must support size/access/rank1/select1");

    cout << "[correctness] " << impl_name << '\n';

    mt19937_64 rng(cfg.seed);

    auto light_check = [&](const BV& bv, const ReferenceBitVector& oracle) {
        check(bv.size() == oracle.size(), "size mismatch");
        const size_t n = oracle.size();
        const size_t head = (n <= 64) ? n : 64;
        for (size_t i = 0; i < head; ++i) {
            check(bv.access(i) == oracle.access(i), "access mismatch");
        }
        uniform_int_distribution<size_t> rank_idx(0, n);
        for (size_t t = 0; t < 64; ++t) {
            const size_t i = rank_idx(rng);
            check(bv.rank1(i) == oracle.rank1(i), "rank mismatch");
        }
        const size_t ones = oracle.rank1(n);
        uniform_int_distribution<size_t> k_dist(0, ones + 2);
        for (size_t t = 0; t < 64; ++t) {
            const size_t k = k_dist(rng);
            check(bv.select1(k) == oracle.select1(k), "select mismatch");
        }
    };

    for (size_t round = 0; round < cfg.rounds; ++round) {
        BV bv;
        ReferenceBitVector oracle;

        if constexpr (is_dynamic_bitvector_v<BV>) {
            // Tiny smoke sequence.
            bv.insert(0, true); oracle.insert(0, true);
            bv.insert(1, false); oracle.insert(1, false);
            bv.flip(1); oracle.flip(1);
            light_check(bv, oracle);

            for (size_t op = 0; op < cfg.operations_per_round; ++op) {
                const size_t n = oracle.size();
                const bool do_query = bernoulli_distribution(0.8)(rng);

                if (!do_query) {
                    if (n == 0) {
                        const bool b = bernoulli_distribution(0.5)(rng);
                        bv.insert(0, b); oracle.insert(0, b);
                    } else {
                        const int upd = uniform_int_distribution<int>(0, 3)(rng);
                        if (upd == 0) {
                            const size_t pos = uniform_int_distribution<size_t>(0, n)(rng);
                            const bool b = bernoulli_distribution(0.5)(rng);
                            bv.insert(pos, b); oracle.insert(pos, b);
                        } else if (upd == 1) {
                            const size_t pos = uniform_int_distribution<size_t>(0, n - 1)(rng);
                            bv.erase(pos); oracle.erase(pos);
                        } else if (upd == 2) {
                            const size_t pos = uniform_int_distribution<size_t>(0, n - 1)(rng);
                            const bool b = bernoulli_distribution(0.5)(rng);
                            bv.set(pos, b); oracle.set(pos, b);
                        } else {
                            const size_t pos = uniform_int_distribution<size_t>(0, n - 1)(rng);
                            bv.flip(pos); oracle.flip(pos);
                        }
                    }
                } else {
                    const size_t n2 = oracle.size();
                    if (n2 == 0) {
                        check(bv.rank1(0) == 0, "rank mismatch");
                        check(bv.select1(0) == bv.size(), "select mismatch");
                    } else {
                        const int qop = uniform_int_distribution<int>(0, 2)(rng);
                        if (qop == 0) {
                            const size_t i = uniform_int_distribution<size_t>(0, n2 - 1)(rng);
                            check(bv.access(i) == oracle.access(i), "access mismatch");
                        } else if (qop == 1) {
                            const size_t i = uniform_int_distribution<size_t>(0, n2)(rng);
                            check(bv.rank1(i) == oracle.rank1(i), "rank mismatch");
                        } else {
                            const size_t k = uniform_int_distribution<size_t>(0, oracle.rank1(n2) + 2)(rng);
                            check(bv.select1(k) == oracle.select1(k), "select mismatch");
                        }
                    }
                }

                if ((op % 256) == 0) {
                    light_check(bv, oracle);
                }
            }
        } else {
            // Query-only: build once, query many times.
            vector<bool> bits;
            bits.reserve(cfg.operations_per_round / 2 + 64);
            bernoulli_distribution bit_dist(0.5);
            for (size_t i = 0; i < cfg.operations_per_round / 2 + 64; ++i) {
                bits.push_back(bit_dist(rng));
            }

            load_from_bits(bv, bits);
            for (bool b : bits) {
                oracle.insert(oracle.size(), b);
            }

            for (size_t op = 0; op < cfg.operations_per_round; ++op) {
                const size_t n = oracle.size();
                const int qop = (n == 0) ? 1 : uniform_int_distribution<int>(0, 2)(rng);
                if (qop == 0) {
                    const size_t i = uniform_int_distribution<size_t>(0, n - 1)(rng);
                    check(bv.access(i) == oracle.access(i), "access mismatch");
                } else if (qop == 1) {
                    const size_t i = uniform_int_distribution<size_t>(0, n)(rng);
                    check(bv.rank1(i) == oracle.rank1(i), "rank mismatch");
                } else {
                    const size_t k = uniform_int_distribution<size_t>(0, oracle.rank1(n) + 2)(rng);
                    check(bv.select1(k) == oracle.select1(k), "select mismatch");
                }
            }

            light_check(bv, oracle);
        }

        cout << "  [ok] round " << (round + 1) << "/" << cfg.rounds << '\n';
    }
}

template <typename BV>
void run_performance_suite(const string& impl_name, const PerfConfig& cfg = {}) {
    static_assert(is_query_bitvector_v<BV>, "BV must support size/access/rank1/select1");

    using clock_t = chrono::steady_clock;
    enum OpId : size_t { OP_INSERT, OP_ERASE, OP_SET, OP_FLIP, OP_ACCESS, OP_RANK, OP_SELECT, OP_COUNT };
    const array<const char*, OP_COUNT> op_name = {"insert", "erase", "set", "flip", "access", "rank1", "select1"};

    uint64_t sink = 0;
    uint64_t total_ops = 0;
    const auto t_all_0 = clock_t::now();
    array<uint64_t, OP_COUNT> op_counts{};
    array<long double, OP_COUNT> op_ns{};

    auto record = [&](OpId op, const clock_t::time_point& t0, const clock_t::time_point& t1) {
        op_counts[op] += 1;
        op_ns[op] += static_cast<long double>(chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count());
    };

    for (size_t round = 0; round < cfg.rounds; ++round) {
        mt19937_64 rng(cfg.seed + static_cast<uint64_t>(round));
        BV bv;

        if constexpr (is_dynamic_bitvector_v<BV>) {
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

        const auto t0 = clock_t::now();
        for (size_t i = 0; i < cfg.operations; ++i) {
            const size_t n = bv.size();

            if constexpr (is_dynamic_bitvector_v<BV>) {
                const bool do_query = bernoulli_distribution(0.8)(rng);
                if (!do_query) {
                    if (n == 0) {
                        const auto a = clock_t::now();
                        bv.insert(0, false);
                        const auto b = clock_t::now();
                        record(OP_INSERT, a, b);
                    } else {
                        const int upd = uniform_int_distribution<int>(0, 3)(rng);
                        if (upd == 0) {
                            const size_t pos = uniform_int_distribution<size_t>(0, n)(rng);
                            const bool bit = bernoulli_distribution(0.5)(rng);
                            const auto a = clock_t::now();
                            bv.insert(pos, bit);
                            const auto b = clock_t::now();
                            record(OP_INSERT, a, b);
                        } else if (upd == 1) {
                            const size_t pos = uniform_int_distribution<size_t>(0, n - 1)(rng);
                            const auto a = clock_t::now();
                            bv.erase(pos);
                            const auto b = clock_t::now();
                            record(OP_ERASE, a, b);
                        } else if (upd == 2) {
                            const size_t pos = uniform_int_distribution<size_t>(0, n - 1)(rng);
                            const bool bit = bernoulli_distribution(0.5)(rng);
                            const auto a = clock_t::now();
                            bv.set(pos, bit);
                            const auto b = clock_t::now();
                            record(OP_SET, a, b);
                        } else {
                            const size_t pos = uniform_int_distribution<size_t>(0, n - 1)(rng);
                            const auto a = clock_t::now();
                            bv.flip(pos);
                            const auto b = clock_t::now();
                            record(OP_FLIP, a, b);
                        }
                    }
                } else {
                    if (n == 0) {
                        const auto a = clock_t::now();
                        sink ^= bv.rank1(0);
                        const auto b = clock_t::now();
                        record(OP_RANK, a, b);
                    } else {
                        const int qop = uniform_int_distribution<int>(0, 2)(rng);
                        if (qop == 0) {
                            const size_t pos = uniform_int_distribution<size_t>(0, n - 1)(rng);
                            const auto a = clock_t::now();
                            sink ^= static_cast<uint64_t>(bv.access(pos));
                            const auto b = clock_t::now();
                            record(OP_ACCESS, a, b);
                        } else if (qop == 1) {
                            const size_t pos = uniform_int_distribution<size_t>(0, n)(rng);
                            const auto a = clock_t::now();
                            sink ^= bv.rank1(pos);
                            const auto b = clock_t::now();
                            record(OP_RANK, a, b);
                        } else {
                            const size_t k = uniform_int_distribution<size_t>(0, n + 2)(rng);
                            const auto a = clock_t::now();
                            sink ^= bv.select1(k);
                            const auto b = clock_t::now();
                            record(OP_SELECT, a, b);
                        }
                    }
                }
            } else {
                if (n == 0) {
                    const auto a = clock_t::now();
                    sink ^= bv.rank1(0);
                    const auto b = clock_t::now();
                    record(OP_RANK, a, b);
                } else {
                    const int qop = uniform_int_distribution<int>(0, 2)(rng);
                    if (qop == 0) {
                        const size_t pos = uniform_int_distribution<size_t>(0, n - 1)(rng);
                        const auto a = clock_t::now();
                        sink ^= static_cast<uint64_t>(bv.access(pos));
                        const auto b = clock_t::now();
                        record(OP_ACCESS, a, b);
                    } else if (qop == 1) {
                        const size_t pos = uniform_int_distribution<size_t>(0, n)(rng);
                        const auto a = clock_t::now();
                        sink ^= bv.rank1(pos);
                        const auto b = clock_t::now();
                        record(OP_RANK, a, b);
                    } else {
                        const size_t k = uniform_int_distribution<size_t>(0, n + 2)(rng);
                        const auto a = clock_t::now();
                        sink ^= bv.select1(k);
                        const auto b = clock_t::now();
                        record(OP_SELECT, a, b);
                    }
                }
            }
            ++total_ops;
        }
        const auto t1 = clock_t::now();

        const auto ns = chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count();
        const double mops = (ns > 0)
            ? (static_cast<double>(cfg.operations) / (static_cast<double>(ns) / 1e9) / 1e6)
            : 0.0;
        cout << "[perf] " << impl_name << " round " << (round + 1)
             << ": " << mops << " Mops/s\n";
    }

    const auto t_all_1 = clock_t::now();
    const auto total_ns = chrono::duration_cast<chrono::nanoseconds>(t_all_1 - t_all_0).count();
    const double overall_mops = (total_ns > 0)
        ? (static_cast<double>(total_ops) / (static_cast<double>(total_ns) / 1e9) / 1e6)
        : 0.0;

    cout << "[perf] " << impl_name << " per-op:\n";
    for (size_t op = 0; op < OP_COUNT; ++op) {
        if (op_counts[op] == 0) {
            continue; // static/query-only naturally omits modifications
        }
        const long double avg_ns = op_ns[op] / static_cast<long double>(op_counts[op]);
        const long double mops = (op_ns[op] > 0.0L)
            ? (static_cast<long double>(op_counts[op]) / (op_ns[op] / 1.0e9L) / 1.0e6L)
            : 0.0L;
        cout << "  - " << op_name[op]
             << ": count=" << op_counts[op]
             << ", avg_ns=" << static_cast<double>(avg_ns)
             << ", Mops/s=" << static_cast<double>(mops) << '\n';
    }

    cout << "[perf] " << impl_name << " total: " << overall_mops
         << " Mops/s (sink=" << sink << ")\n";
}

} // namespace dbv
