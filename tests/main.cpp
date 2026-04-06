#include "dynamic_bitvector/naive_dynamic_bitvector.hpp"
#include "dynamic_bitvector/static_bitvector.hpp"
#include "dynamic_bitvector/test_harness.hpp"

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

    return opt;
}

template <typename BV>
void run_selected(const string& name, const CliOptions& opt) {
    dbv::TestConfig cfg;

    dbv::PerfConfig perf_cfg;

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

    opt = parse_args(argc, argv);

    if (opt.impl == "naive" || opt.impl == "all") {
        run_selected<dbv::NaiveDynamicBitVector>("NaiveDynamicBitVector", opt);
    }
    if (opt.impl == "static" || opt.impl == "all") {
        run_selected<dbv::RankSelectBitVector>("RankSelectBitVector", opt);
    }

    return 0;
}
