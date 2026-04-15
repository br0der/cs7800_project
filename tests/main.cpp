#include "Navarro25/adaptive_dynamic_bitvector.hpp"
#include "btree_dynamic_bitvector/basic_btree_dynamic_bitvector.hpp"
#include "dynamic_bitvector/naive_dynamic_bitvector.hpp"
#include "dynamic_bitvector/static_bitvector.hpp"
#include "dynamic_bitvector/test_harness.hpp"

#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>

using namespace std;

namespace dbv {

template <>
struct is_dynamic_bitvector<RankSelectBitVector, void> : false_type {};

} // namespace dbv

namespace {

void navarro25_check_static_bits(const vector<unsigned char>& bits, const string& context) {
    Navarro25::StaticBitVector bv;
    dbv::ReferenceBitVector oracle;

    bv.build(bits);
    for (unsigned char bit : bits) {
        oracle.insert(oracle.size(), bit != 0);
    }

    dbv::check(bv.size() == oracle.size(), context + " | size mismatch");
    dbv::check(bv.ones() == oracle.rank1(oracle.size()), context + " | ones mismatch");

    for (size_t i = 0; i < bits.size(); ++i) {
        dbv::check(bv.access(i) == oracle.access(i), context + " | access mismatch");
    }

    for (size_t i = 0; i <= bits.size(); ++i) {
        dbv::check(bv.rank1(i) == oracle.rank1(i), context + " | rank1 mismatch");
        dbv::check(bv.rank0(i) == i - oracle.rank1(i), context + " | rank0 mismatch");
    }

    const size_t ones = oracle.rank1(oracle.size());
    for (size_t k = 0; k < ones + 3; ++k) {
        dbv::check(bv.select1(k) == oracle.select1(k), context + " | select1 mismatch");
    }

    vector<size_t> zero_positions;
    for (size_t i = 0; i < bits.size(); ++i) {
        if (bits[i] == 0) {
            zero_positions.push_back(i);
        }
    }
    for (size_t k = 0; k < zero_positions.size() + 3; ++k) {
        const size_t expected = k < zero_positions.size() ? zero_positions[k] : bits.size();
        dbv::check(bv.select0(k) == expected, context + " | select0 mismatch");
    }

    vector<unsigned char> round_trip;
    bv.append_bits(round_trip);
    dbv::check(round_trip == bits, context + " | append_bits round-trip mismatch");
}

void navarro25_static_bitvector_regression() {
    navarro25_check_static_bits({}, "Navarro25 static empty");
    navarro25_check_static_bits(vector<unsigned char>(130, 0), "Navarro25 static all zeros");
    navarro25_check_static_bits(vector<unsigned char>(130, 1), "Navarro25 static all ones");

    vector<unsigned char> alternating(200, 0);
    for (size_t i = 0; i < alternating.size(); ++i) {
        alternating[i] = static_cast<unsigned char>((i % 2) != 0);
    }
    navarro25_check_static_bits(alternating, "Navarro25 static alternating");

    vector<unsigned char> boundary_sparse(257, 0);
    for (const size_t pos : {0ULL, 1ULL, 62ULL, 63ULL, 64ULL, 65ULL, 127ULL, 128ULL, 255ULL, 256ULL}) {
        boundary_sparse[pos] = 1;
    }
    navarro25_check_static_bits(boundary_sparse, "Navarro25 static boundary sparse");

    vector<unsigned char> mixed(513, 0);
    for (size_t i = 0; i < mixed.size(); ++i) {
        mixed[i] = static_cast<unsigned char>((i % 3 == 0) || (i % 11 == 0) || (i == 511));
    }
    navarro25_check_static_bits(mixed, "Navarro25 static mixed");
}

void navarro25_adaptive_regression() {
    Navarro25::AdaptiveDynamicBitVector bv;
    dbv::ReferenceBitVector oracle;

    const auto check_boundaries = [&](const string& context) {
        dbv::assert_same_state(bv, oracle, context);

        const size_t positions[] = {0, 1, 62, 63, 64, 65, 127, 128};
        for (const size_t pos : positions) {
            if (pos < bv.size()) {
                dbv::check(bv.access(pos) == oracle.access(pos), context + " | access boundary mismatch");
            }
            if (pos <= bv.size()) {
                dbv::check(bv.rank1(pos) == oracle.rank1(pos), context + " | rank boundary mismatch");
            }
            if (pos < oracle.rank1(oracle.size()) + 3) {
                dbv::check(bv.select1(pos) == oracle.select1(pos), context + " | select boundary mismatch");
            }
        }
    };

    for (size_t i = 0; i < 200; ++i) {
        const bool bit = (i % 3 == 0) || (i % 7 == 0);
        bv.insert(bv.size(), bit);
        oracle.insert(oracle.size(), bit);
    }

    check_boundaries("Navarro25 after multi-leaf build");
    dbv::check(bv.check_invariants(), "Navarro25 invariant failure after multi-leaf build");

    const size_t insert_positions[] = {0, 63, 64};
    for (const size_t pos : insert_positions) {
        bv.insert(pos, true);
        oracle.insert(pos, true);
        check_boundaries("Navarro25 after packed-boundary insert");
        dbv::check(bv.check_invariants(), "Navarro25 invariant failure after packed-boundary insert");
    }

    bv.insert(bv.size(), false);
    oracle.insert(oracle.size(), false);
    check_boundaries("Navarro25 after packed-boundary append");
    dbv::check(bv.check_invariants(), "Navarro25 invariant failure after packed-boundary append");

    const size_t erase_positions[] = {0, 63, 64};
    for (const size_t pos : erase_positions) {
        bv.erase(pos);
        oracle.erase(pos);
        check_boundaries("Navarro25 after packed-boundary erase");
        dbv::check(bv.check_invariants(), "Navarro25 invariant failure after packed-boundary erase");
    }

    bv.erase(bv.size() - 1);
    oracle.erase(oracle.size() - 1);
    check_boundaries("Navarro25 after packed-boundary erase end");
    dbv::check(bv.check_invariants(), "Navarro25 invariant failure after packed-boundary erase end");

    const size_t ones = oracle.rank1(oracle.size());
    for (size_t i = 0; i < bv.size() + 10; ++i) {
        (void)bv.access(i % bv.size());
        (void)bv.rank1(i % (bv.size() + 1));
        (void)bv.select1(i % (ones + 1));
    }

    dbv::check(bv.static_leaf_count() > 0, "Navarro25 should create at least one static leaf");
    dbv::check(bv.check_invariants(), "Navarro25 invariant failure after query-triggered flattening");

    bv.set(5, true);
    oracle.set(5, true);
    dbv::check(bv.check_invariants(), "Navarro25 invariant failure after static materialization set");

    bv.flip(17);
    oracle.flip(17);
    dbv::check(bv.check_invariants(), "Navarro25 invariant failure after materialized flip");

    bv.insert(11, true);
    oracle.insert(11, true);
    dbv::check(bv.check_invariants(), "Navarro25 invariant failure after materialized insert repair");

    bv.erase(2);
    oracle.erase(2);
    dbv::check(bv.check_invariants(), "Navarro25 invariant failure after materialized erase repair");

    for (size_t i = 0; i < 20; ++i) {
        bv.erase(0);
        oracle.erase(0);
    }
    dbv::check(bv.check_invariants(), "Navarro25 invariant failure after materialized merge repair");

    check_boundaries("Navarro25 after update through static materialization");
}

void navarro25_static_heavy_adaptive_regression() {
    Navarro25::AdaptiveDynamicBitVector bv;
    dbv::ReferenceBitVector oracle;
    mt19937_64 rng(0x57A71C25ULL);
    bernoulli_distribution bit_dist(0.45);

    for (size_t i = 0; i < 384; ++i) {
        const bool bit = bit_dist(rng);
        bv.insert(bv.size(), bit);
        oracle.insert(oracle.size(), bit);
    }

    dbv::assert_same_state(bv, oracle, "Navarro25 static-heavy after build");
    dbv::check(bv.check_invariants(), "Navarro25 invariant failure in static-heavy after build");

    for (size_t i = 0; i < bv.size() * 3 + 64; ++i) {
        const size_t n = oracle.size();
        switch (i % 3) {
            case 0: {
                uniform_int_distribution<size_t> pos_dist(0, n - 1);
                const size_t pos = pos_dist(rng);
                dbv::check(bv.access(pos) == oracle.access(pos), "Navarro25 static-heavy access mismatch");
                break;
            }
            case 1: {
                uniform_int_distribution<size_t> pos_dist(0, n);
                const size_t pos = pos_dist(rng);
                dbv::check(bv.rank1(pos) == oracle.rank1(pos), "Navarro25 static-heavy rank mismatch");
                break;
            }
            default: {
                const size_t ones = oracle.rank1(n);
                uniform_int_distribution<size_t> k_dist(0, ones + 2);
                const size_t k = k_dist(rng);
                dbv::check(bv.select1(k) == oracle.select1(k), "Navarro25 static-heavy select mismatch");
                break;
            }
        }
    }

    dbv::check(bv.static_leaf_count() > 0, "Navarro25 static-heavy should create at least one static leaf");
    dbv::assert_same_state(bv, oracle, "Navarro25 static-heavy after flattening queries");
    dbv::check(bv.check_invariants(), "Navarro25 invariant failure in static-heavy after flattening queries");

    bernoulli_distribution choose_query(0.95);
    for (size_t op = 0; op < 1200; ++op) {
        const size_t n = oracle.size();
        if (choose_query(rng) && n > 0) {
            uniform_int_distribution<int> query_dist(0, 2);
            switch (query_dist(rng)) {
                case 0: {
                    uniform_int_distribution<size_t> pos_dist(0, n - 1);
                    const size_t pos = pos_dist(rng);
                    dbv::check(bv.access(pos) == oracle.access(pos), "Navarro25 static-heavy post-update access mismatch");
                    break;
                }
                case 1: {
                    uniform_int_distribution<size_t> pos_dist(0, n);
                    const size_t pos = pos_dist(rng);
                    dbv::check(bv.rank1(pos) == oracle.rank1(pos), "Navarro25 static-heavy post-update rank mismatch");
                    break;
                }
                default: {
                    const size_t ones = oracle.rank1(n);
                    uniform_int_distribution<size_t> k_dist(0, ones + 2);
                    const size_t k = k_dist(rng);
                    dbv::check(bv.select1(k) == oracle.select1(k), "Navarro25 static-heavy post-update select mismatch");
                    break;
                }
            }
        } else {
            uniform_int_distribution<int> update_dist(0, n == 0 ? 0 : 3);
            switch (update_dist(rng)) {
                case 0: {
                    uniform_int_distribution<size_t> pos_dist(0, n);
                    const size_t pos = pos_dist(rng);
                    const bool bit = bit_dist(rng);
                    bv.insert(pos, bit);
                    oracle.insert(pos, bit);
                    break;
                }
                case 1: {
                    uniform_int_distribution<size_t> pos_dist(0, n - 1);
                    const size_t pos = pos_dist(rng);
                    bv.erase(pos);
                    oracle.erase(pos);
                    break;
                }
                case 2: {
                    uniform_int_distribution<size_t> pos_dist(0, n - 1);
                    const size_t pos = pos_dist(rng);
                    const bool bit = bit_dist(rng);
                    bv.set(pos, bit);
                    oracle.set(pos, bit);
                    break;
                }
                default: {
                    uniform_int_distribution<size_t> pos_dist(0, n - 1);
                    const size_t pos = pos_dist(rng);
                    bv.flip(pos);
                    oracle.flip(pos);
                    break;
                }
            }
        }

        if ((op % 64) == 0) {
            dbv::assert_same_state(bv, oracle, "Navarro25 static-heavy randomized checkpoint");
            dbv::check(bv.check_invariants(), "Navarro25 invariant failure in static-heavy randomized checkpoint");
        }
    }

    dbv::assert_same_state(bv, oracle, "Navarro25 static-heavy randomized final");
    dbv::check(bv.check_invariants(), "Navarro25 invariant failure in static-heavy randomized final");
}

void btree_dynamic_bitvector_regression() {
    using TestBV = BTreeDBV::BasicDynamicBitVector<8, 256>;

    TestBV bv;
    dbv::ReferenceBitVector oracle;
    mt19937_64 rng(0xB7AEE245ULL);
    bernoulli_distribution bit_dist(0.5);

    for (size_t i = 0; i < 1024; ++i) {
        uniform_int_distribution<size_t> pos_dist(0, oracle.size());
        const size_t pos = pos_dist(rng);
        const bool bit = bit_dist(rng);
        bv.insert(pos, bit);
        oracle.insert(pos, bit);

        if ((i % 97) == 0) {
            dbv::assert_same_state(bv, oracle, "BTreeDBV split-build checkpoint");
            dbv::check(bv.check_invariants(), "BTreeDBV invariant failure during split-build checkpoint");
        }
    }

    dbv::assert_same_state(bv, oracle, "BTreeDBV after split-heavy build");
    dbv::check(bv.check_invariants(), "BTreeDBV invariant failure after split-heavy build");

    for (size_t op = 0; op < 3000; ++op) {
        const size_t n = oracle.size();
        const bool do_insert = (n == 0) || bernoulli_distribution(0.35)(rng);

        if (do_insert) {
            uniform_int_distribution<size_t> pos_dist(0, n);
            const size_t pos = pos_dist(rng);
            const bool bit = bit_dist(rng);
            bv.insert(pos, bit);
            oracle.insert(pos, bit);
        } else {
            switch (uniform_int_distribution<int>(0, 2)(rng)) {
                case 0: {
                    uniform_int_distribution<size_t> pos_dist(0, n - 1);
                    const size_t pos = pos_dist(rng);
                    bv.erase(pos);
                    oracle.erase(pos);
                    break;
                }
                case 1: {
                    uniform_int_distribution<size_t> pos_dist(0, n - 1);
                    const size_t pos = pos_dist(rng);
                    const bool bit = bit_dist(rng);
                    bv.set(pos, bit);
                    oracle.set(pos, bit);
                    break;
                }
                default: {
                    uniform_int_distribution<size_t> pos_dist(0, n - 1);
                    const size_t pos = pos_dist(rng);
                    bv.flip(pos);
                    oracle.flip(pos);
                    break;
                }
            }
        }

        if ((op % 64) == 0) {
            dbv::assert_same_state(bv, oracle, "BTreeDBV mixed checkpoint");
            dbv::check(bv.check_invariants(), "BTreeDBV invariant failure during mixed checkpoint");
        }
    }

    while (oracle.size() > 48) {
        uniform_int_distribution<size_t> pos_dist(0, oracle.size() - 1);
        const size_t pos = pos_dist(rng);
        bv.erase(pos);
        oracle.erase(pos);

        if ((oracle.size() % 53) == 0) {
            dbv::assert_same_state(bv, oracle, "BTreeDBV merge checkpoint");
            dbv::check(bv.check_invariants(), "BTreeDBV invariant failure during merge checkpoint");
        }
    }

    dbv::assert_same_state(bv, oracle, "BTreeDBV final regression");
    dbv::check(bv.check_invariants(), "BTreeDBV invariant failure in final regression");
}

} // namespace

namespace dbv {

template <>
struct is_dynamic_bitvector<RankSelectBitVector, void> : false_type {};

// Loader hook for query-only static bitvector.
inline void load_from_bits(RankSelectBitVector& bv, const vector<bool>& bits) {
    bv = RankSelectBitVector(bits);
}

} // namespace dbv

namespace Navarro25 {

inline void load_from_bits(StaticBitVector& bv, const vector<bool>& bits) {
    vector<unsigned char> packed_bits;
    packed_bits.reserve(bits.size());
    for (bool bit : bits) {
        packed_bits.push_back(static_cast<unsigned char>(bit));
    }
    bv.build(packed_bits);
}

} // namespace Navarro25

namespace {

struct CliOptions {
    string impl = "all";       // naive | static | btree | navarro-static | navarro | all
    string test = "both";      // correctness | performance | both
};

void print_help(const char* prog) {
    cout
        << "Usage: " << prog << " [options]\n"
        << "\n"
        << "Implementation selection:\n"
        << "  --impl naive|static|btree|navarro-static|navarro|all\n"
        << "                                  Choose implementation(s) to run (default: all)\n"
        << "\n"
        << "Test selection:\n"
        << "  --test correctness|performance|both   Choose test type(s) (default: both)\n"
        << "\n"
        << "  -h, --help                      Show this help\n";
}

CliOptions parse_args(int argc, char** argv) {
    CliOptions opt;

    for (int i = 1; i < argc; ++i) {
        const string arg = argv[i];

        auto need_value = [&](const string& name) -> string {
            if (i + 1 >= argc) {
                throw invalid_argument("Missing value for " + name);
            }
            return string(argv[++i]);
        };

        if (arg == "-h" || arg == "--help") {
            print_help(argv[0]);
            exit(0);
        } else if (arg == "--impl") {
            opt.impl = need_value(arg);
        } else if (arg == "--test") {
            opt.test = need_value(arg);
        }
    }

    if (!(opt.impl == "naive" || opt.impl == "static" || opt.impl == "btree" || opt.impl == "navarro-static" ||
          opt.impl == "navarro" || opt.impl == "all")) {
        throw invalid_argument("--impl must be one of: naive, static, btree, navarro-static, navarro, all");
    }
    if (!(opt.test == "correctness" || opt.test == "performance" || opt.test == "both")) {
        throw invalid_argument("--test must be one of: correctness, performance, both");
    }

    return opt;
}

template <typename BV>
void run_extra_correctness_checks() {
    if constexpr (is_same_v<BV, Navarro25::StaticBitVector>) {
        navarro25_static_bitvector_regression();
    } else if constexpr (is_same_v<BV, Navarro25::AdaptiveDynamicBitVector>) {
        navarro25_adaptive_regression();
        navarro25_static_heavy_adaptive_regression();
    } else if constexpr (is_same_v<BV, BTreeDBV::DynamicBitVector>) {
        btree_dynamic_bitvector_regression();
    }
}

template <typename BV>
void run_selected(const string& name, const CliOptions& opt) {
    dbv::TestConfig cfg;

    dbv::PerfConfig perf_cfg;

    const bool run_correctness = (opt.test == "correctness" || opt.test == "both");
    const bool run_performance = (opt.test == "performance" || opt.test == "both");

    if (run_correctness) {
        dbv::run_suite<BV>(name, cfg);
        run_extra_correctness_checks<BV>();
    }
    if (run_performance) {
        dbv::run_performance_suite<BV>(name, perf_cfg);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions opt = parse_args(argc, argv);

        // if (opt.impl == "naive" || opt.impl == "all") {
        //     run_selected<dbv::NaiveDynamicBitVector>("NaiveDynamicBitVector", opt);
        // }
        if (opt.impl == "static" || opt.impl == "all") {
            run_selected<dbv::RankSelectBitVector>("RankSelectBitVector", opt);
        }
        if (opt.impl == "btree" || opt.impl == "all") {
            run_selected<BTreeDBV::DynamicBitVector>("BTreeDBV::DynamicBitVector", opt);
        }
        if (opt.impl == "navarro-static" || opt.impl == "navarro" || opt.impl == "all") {
            run_selected<Navarro25::StaticBitVector>("Navarro25::StaticBitVector", opt);
        }
        if (opt.impl == "navarro" || opt.impl == "all") {
            run_selected<Navarro25::AdaptiveDynamicBitVector>("Navarro25::AdaptiveDynamicBitVector", opt);
        }

        return 0;
    } catch (const exception& ex) {
        cerr << "Error: " << ex.what() << '\n';
        cerr << "Use --help for usage.\n";
        return 1;
    }
}
