#include "dynamic_bitvector/naive_dynamic_bitvector.hpp"
#include "dynamic_bitvector/static_bitvector.hpp"
#include "dynamic_bitvector/test_harness.hpp"

#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace std;

namespace dbv {

// Loader hook for query-only static bitvector.
inline void load_from_bits(RankSelectBitVector& bv, const vector<bool>& bits) {
    bv = RankSelectBitVector(bits);
}

} // namespace dbv

namespace {

struct CliOptions {
    string impl = "all";       // naive | static | all
    string test = "both";      // correctness | performance | both

    size_t rounds = 6;
    size_t ops_per_round = 4000;
    size_t q = 4;
    uint64_t seed = 0xB17B17ULL;

    size_t perf_rounds = 3;
    size_t perf_initial_size = 1 << 14;
    size_t perf_ops = 200000;
    size_t perf_q = 4;
    uint64_t perf_seed = 0xB17B17ULL ^ 0xABCDEFULL;

    bool verbose = true;
};

void print_help(const char* prog) {
    cout
        << "Usage: " << prog << " [options]\n"
        << "\n"
        << "Implementation selection:\n"
        << "  --impl naive|static|all         Choose implementation(s) to run (default: all)\n"
        << "\n"
        << "Test selection:\n"
        << "  --test correctness|performance|both   Choose test type(s) (default: both)\n"
        << "\n"
        << "Correctness config:\n"
        << "  --rounds N                      Randomized correctness rounds (default: 6)\n"
        << "  --ops-per-round N              Randomized correctness operations per round (default: 4000)\n"
        << "  --q N                           Queries-per-update ratio for correctness (default: 4)\n"
        << "  --seed N                        Correctness RNG seed (default: 0xB17B17)\n"
        << "\n"
        << "Performance config:\n"
        << "  --perf-rounds N                 Performance rounds (default: 3)\n"
        << "  --perf-ops N                    Timed operations per performance round (default: 200000)\n"
        << "  --initial-size N                Initial size for performance runs (default: 16384)\n"
        << "  --perf-q N                      Queries-per-update ratio for dynamic perf (default: 4)\n"
        << "  --perf-seed N                   Performance RNG seed\n"
        << "\n"
        << "Other:\n"
        << "  --quiet                         Reduce output\n"
        << "  -h, --help                      Show this help\n";
}

uint64_t parse_u64(const string& s, const string& flag_name) {
    try {
        size_t idx = 0;
        const uint64_t v = stoull(s, &idx, 0);
        if (idx != s.size()) {
            throw invalid_argument("trailing characters");
        }
        return v;
    } catch (const exception&) {
        throw invalid_argument("Invalid value for " + flag_name + ": " + s);
    }
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
        } else if (arg == "--rounds") {
            opt.rounds = static_cast<size_t>(parse_u64(need_value(arg), arg));
        } else if (arg == "--ops-per-round") {
            opt.ops_per_round = static_cast<size_t>(parse_u64(need_value(arg), arg));
        } else if (arg == "--q") {
            opt.q = static_cast<size_t>(parse_u64(need_value(arg), arg));
        } else if (arg == "--seed") {
            opt.seed = parse_u64(need_value(arg), arg);
        } else if (arg == "--perf-rounds") {
            opt.perf_rounds = static_cast<size_t>(parse_u64(need_value(arg), arg));
        } else if (arg == "--perf-ops") {
            opt.perf_ops = static_cast<size_t>(parse_u64(need_value(arg), arg));
        } else if (arg == "--initial-size") {
            opt.perf_initial_size = static_cast<size_t>(parse_u64(need_value(arg), arg));
        } else if (arg == "--perf-q") {
            opt.perf_q = static_cast<size_t>(parse_u64(need_value(arg), arg));
        } else if (arg == "--perf-seed") {
            opt.perf_seed = parse_u64(need_value(arg), arg);
        } else if (arg == "--quiet") {
            opt.verbose = false;
        } else {
            throw invalid_argument("Unknown argument: " + arg);
        }
    }

    if (!(opt.impl == "naive" || opt.impl == "static" || opt.impl == "all")) {
        throw invalid_argument("--impl must be one of: naive, static, all");
    }
    if (!(opt.test == "correctness" || opt.test == "performance" || opt.test == "both")) {
        throw invalid_argument("--test must be one of: correctness, performance, both");
    }

    return opt;
}

template <typename BV>
void run_selected(const string& name, const CliOptions& opt) {
    dbv::TestConfig cfg;
    cfg.rounds = opt.rounds;
    cfg.operations_per_round = opt.ops_per_round;
    cfg.q = opt.q;
    cfg.seed = opt.seed;
    cfg.verbose = opt.verbose;

    dbv::PerfConfig perf_cfg;
    perf_cfg.rounds = opt.perf_rounds;
    perf_cfg.initial_size = opt.perf_initial_size;
    perf_cfg.operations = opt.perf_ops;
    perf_cfg.q = opt.perf_q;
    perf_cfg.seed = opt.perf_seed;
    perf_cfg.verbose = opt.verbose;

    const bool run_correctness = (opt.test == "correctness" || opt.test == "both");
    const bool run_performance = (opt.test == "performance" || opt.test == "both");

    if (run_correctness) {
        dbv::run_suite<BV>(name, cfg);
    }
    if (run_performance) {
        dbv::run_performance_suite<BV>(name, perf_cfg);
    }
}

} // namespace

int main(int argc, char** argv) {
    CliOptions opt;

    try {
        opt = parse_args(argc, argv);

        if (opt.impl == "naive" || opt.impl == "all") {
            run_selected<dbv::NaiveDynamicBitVector>("NaiveDynamicBitVector", opt);
        }
        if (opt.impl == "static" || opt.impl == "all") {
            run_selected<dbv::RankSelectBitVector>("RankSelectBitVector", opt);
        }

        return 0;
    } catch (const exception& ex) {
        cerr << "Error: " << ex.what() << '\n';
        cerr << "Use --help for usage.\n";
        return 1;
    }
}
