# Dynamic Bitvector Testing Suite (C++)

This project provides a basic differential testing suite for dynamic bitvector implementations (for example, implementations inspired by recent succinct data structure papers), and also supports query-only (non-dynamic) bitvectors.

## What is included

- A minimal dynamic bitvector interface contract (compile-time trait check).
- A reference oracle model (`ReferenceBitVector`) with simple, trusted behavior.
- A baseline implementation (`NaiveDynamicBitVector`) to validate the harness.
- Deterministic tests and randomized operation-sequence tests.

### Tuning query pressure

`TestConfig` now includes `q`, the expected query-per-update ratio used by the randomized tester:

- `q = 0`: mostly updates
- `q = 1`: roughly 1 query per update
- `q = 4`: roughly 4 queries per update

Set it in [tests/main.cpp](tests/main.cpp):

```cpp
cfg.q = 4;
```

## Operations covered

The harness validates these operations:

- `size()`
- `access(i)`
- `rank1(i)` where `i` is a prefix endpoint in `[0, size()]` and rank is over `[0, i)`
- `select1(k)` returning position of the `k`-th one (0-indexed), or `size()` if absent
- `insert(i, bit)`
- `erase(i)`
- `set(i, bit)`
- `flip(i)`
- `clear()`

## Build and run

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Or run directly:

```bash
./build/bitvector_tests
```

## Adapting a paper implementation

1. Create an adapter class that exposes the expected methods and semantics.
2. For dynamic adapters, satisfy `dbv::is_dynamic_bitvector_v<T>`.
3. For query-only adapters, satisfy `dbv::is_query_bitvector_v<T>` and provide a loader:

```cpp
namespace dbv {
void load_from_bits(MyQueryOnlyAdapter& bv, const std::vector<bool>& bits);
}
```

4. Add a call in [tests/main.cpp](tests/main.cpp):

```cpp
dbv::run_suite<MyPaperBitVectorAdapter>("MyPaperBitVectorAdapter", cfg);
```

## Suggested adapter pattern

If your implementation has different method names, wrap it.

Dynamic example:

```cpp
class MyPaperBitVectorAdapter {
public:
	std::size_t size() const;
	bool access(std::size_t i) const;
	std::size_t rank1(std::size_t i) const;
	std::size_t select1(std::size_t k) const;
	void insert(std::size_t i, bool b);
	void erase(std::size_t i);
	void set(std::size_t i, bool b);
	void flip(std::size_t i);
	void clear();
};
```

Query-only example:

```cpp
class MyQueryOnlyAdapter {
public:
	std::size_t size() const;
	bool access(std::size_t i) const;
	std::size_t rank1(std::size_t i) const;
	std::size_t select1(std::size_t k) const;
};

namespace dbv {
void load_from_bits(MyQueryOnlyAdapter& bv, const std::vector<bool>& bits) {
	// Rebuild/initialize your static structure from bits.
}
}
```

## Notes

- This suite is intentionally simple and extensible.
- For research-grade evaluation, add larger seeds, adversarial traces, and timing/memory instrumentation.
