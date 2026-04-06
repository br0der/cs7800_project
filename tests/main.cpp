#include "dynamic_bitvector/naive_dynamic_bitvector.hpp"
#include "dynamic_bitvector/test_harness.hpp"

#include <exception>
#include <iostream>

using namespace std;

int main() {
    dbv::TestConfig cfg;
    cfg.rounds = 6;
    cfg.operations_per_round = 4000;
    cfg.q = 4;
    cfg.seed = 0xB17B17ULL;

    try {
        dbv::run_suite<dbv::NaiveDynamicBitVector>("NaiveDynamicBitVector", cfg);

        // Add additional implementations from recent papers by creating an adapter
        // class (dynamic or query-only) and calling run_suite again.
        // Example:
        // dbv::run_suite<MyPaperBitVectorAdapter>("MyPaperBitVectorAdapter", cfg);

        return 0;
    } catch (const exception& ex) {
        cerr << "Test failure: " << ex.what() << '\n';
        return 1;
    }
}
