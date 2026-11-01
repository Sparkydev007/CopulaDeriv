# CopulaDeriv — C++ and Performance Engineering Interview Questions

This file provides detailed questions and answers based on the actual CopulaDeriv project, covering C++20 concepts and performance engineering topics. Answers ground themselves in the repository code where possible.

## Stack versus heap

### Question
Explain the difference between stack and heap memory allocation, and which is used in the CopulaDeriv C++20 engine.

### Answer
- **Stack**: Memory allocated automatically when a function is called; deallocated when the function returns. Stack frames are LIFO (last-in, first-out). Fast allocation (just moving the stack pointer), but limited size (typically 1-8 MB depending on OS/language). Variables have lifetime scoped to the function block.
- **Heap**: Memory allocated dynamically via `new`/`delete` (C++) or `malloc`/`free` (C). The programmer controls the lifetime. Slower allocation (OS/system call overhead), but can allocate large amounts. Variables persist until explicitly freed.

### CopulaDeriv usage
- **Stack**: Local variables in functions like `phi_inv(p)` (the rational approximation) and loop counters. The `std::array<std::array<double, 2>, 2> L_` in `GaussianCopula` and `StudentTCopula` is a member variable, allocated on the heap (via the class constructor/RAII).
- **Heap**: `std::vector<double>` in `generate_terminal_prices` (S1_T, S2_T of size N), `std::vector<std::pair<double, double>> U(N)` in copula sampling. The `RandomGenerator` engine `engine_` is a member variable.

**RAII (Resource Acquisition Is Initialization)**: The C++ engine uses RAII patterns: `std::vector` destructors free heap memory automatically. The `GaussianCopula` and `StudentTCopula` classes have member `std::array L_` which is automatically constructed/destructed. No manual `new`/`delete` is needed in the main code paths.

**Common incorrect answer**: "All C++ arrays are on the stack." — Incorrect; `std::vector` and dynamic arrays are on the heap; C-style `double*` allocated with `new` are on the heap; C-style `double[]` as function parameters may be on either depending on allocation.

**Follow-up**: How would you optimize memory usage for very large N (e.g., N = 10 million)?

---

## Pointers and references

### Question
Explain pointers and references in C++, and how the CopulaDeriv code uses them.

### Answer
- **Pointer**: A variable holding the memory address of another variable. Declared with `*`, dereferenced with `*`. Can be null. Used for dynamic memory, array access, and passing large objects by "address."
- **Reference**: An alias to an existing variable. Declared with `&`. Must be initialized upon creation; cannot be reseated. Provides syntax similar to the original variable without the danger of null/dangling references.

### CopulaDeriv usage
- **Pointers**: 
  - `RandomGenerator::normal()` returns `double` by value (no pointer).
  - `chi_squared(int k)` returns `double` by value.
  - `RandomGenerator::state()` returns `unsigned int` by value.
  - In `pricing.cpp`, `sample_std(const std::vector<double>& values)` takes a `const reference` (not a pointer), but internally uses pointer-like access via `values.begin()`, `values.end()`.
  - The CMake `file(GLOB SRC_FILES "src/*.cpp")` pattern uses string paths, not C pointers.

- **References**:
  - `GaussianCopula::sample(int N, RandomGenerator& rng)` takes `RNG&` (reference wrapper).
  - `StudentTCopula::sample(int N, RandomGenerator& rng)` takes `RNG&`.
  - `MonteCarloPricer::price(int N, int seed, const std::string& model)` takes `const std::string&`.
  - `price_from_payoffs(const std::vector<double>& payoffs)` takes `const reference` to avoid copying the payoff vector.
  - `generate_terminal_prices(..., const std::string& copula_type = "gaussian")` takes `const reference`.

**Key design principle**: Use `const T&` for function parameters when you need to avoid copying but don't intend to modify the argument. Use `T*` when you might have a null case or need pointer arithmetic.

**Common incorrect answer**: "References are just pointers with different syntax." — Partially true but references guarantee non-null and are initialized; they are safer and preferred in modern C++.

**Follow-up**: Why does `price_from_payoffs` take `const std::vector<double>&` instead of `std::vector<double>` by value?

---

## RAII

### Question
Explain RAII (Resource Acquisition Is Initialization) and how it's used in the CopulaDeriv C++20 engine.

### Answer
- **RAII principle**: Resource allocation (memory, file handles, network sockets, etc.) is tied to object lifetime. When the object is constructed, the resource is acquired; when the object is destroyed (goes out of scope), the resource is automatically released via the destructor.
- **Benefits**: No memory leaks if the destructor runs (even via exception paths); no explicit `delete` needed; cleaner code.

### CopulaDeriv usage
- **`std::vector`**: When a `std::vector<double>` goes out of scope, its destructor frees the heap memory. This is used throughout: `payoffs` vector in `price_from_payoffs`, `S1_T`/`S2_T` in `generate_terminal_prices`, `U` in copula sampling.
- **`std::mt19937_64 engine_`** in `RandomGenerator`: The Mersenne Twister engine is a member variable; it's constructed in the initializer list and destroyed automatically when the `RandomGenerator` object is destroyed. No `seed()` call needed after construction for basic reproducibility.
- **`std::array<std::array<double, 2>, 2> L_`** in `GaussianCopula` / `StudentTCopula`: The Cholesky matrix is a member; it's initialized in the constructor (`set_rho(rho)`) and destroyed automatically.
- **No manual `new`/`delete`**: The main code paths in `pricing.cpp` and `copula.cpp` use RAII containers (`std::vector`, `std::array`) exclusively. The only raw allocation is the `std::mt19937_64 engine_`, which is a member variable.

**Numerical stability note**: The `phi_inv(p)` function in `pricing.cpp` uses a local array for the rational approximation coefficients, but these are `double` locals on the stack; no dynamic allocation is involved.

**Common incorrect answer**: "RAII means garbage collection." — Incorrect; RAII is deterministic C++ destruction, not non-deterministic GC like Java/Python.

**Follow-up**: What would happen if a `std::vector` member were not initialized in the constructor's initializer list?

---

## Memory allocation

### Question
How does memory allocation work in the CopulaDeriv C++20 engine? What is the complexity of the inner loops?

### Answer
- **Per-path memory**: The inner loop over N paths allocates the following `std::vector`s once (outside the inner loop):
  - `std::vector<double> Z(2*N)` — 2N doubles for independent normals (in `copula.cpp` Gaussian sample).
  - `std::vector<std::pair<double, double>> U(N)` — N uniform pairs (in `copula.cpp`).
  - `std::vector<double> S1_T(N)`, `std::vector<double> S2_T(N)` — terminal prices (in `generate_terminal_prices`).
  - `std::vector<double> payoffs(N)` — payoff values (in pricing).
  - All are allocated once before the inner loop; the inner loop only writes to these pre-allocated arrays.

- **Inner loop complexity**: O(1) per path (no dynamic allocation inside the loop). The loop body performs:
  - Read two random normals (or read from pre-generated Z).
  - Cholesky transform (2 multiplications, 1 addition).
  - erf() transform (or normal CDF).
  - GBM formula: one multiplication, one addition, one exp().
  - Store to pre-allocated array.

- **Total memory for N = 1,000,000**: ~16 MB for the double arrays (2N * 8 bytes × several arrays). This fits in L3 cache on modern CPUs, which contributes to the good performance.

**Benchmark observation**: For N = 1,000,000, the C++20 median runtime is ~0.38 s; the dominant cost is the generation of N normals and the element-wise transform, not memory allocation.

**Common incorrect answer**: "The C++ engine allocates memory inside the inner loop for each path." — Incorrect; pre-allocating all vectors outside the loop is the standard optimization pattern, and the CopulaDeriv code follows this.

**Follow-up**: How would you further reduce memory usage for very large N?

---

## Cache locality

### Question
Explain cache locality and how the CopulaDeriv C++20 engine achieves good cache performance.

### Answer
- **Cache locality**: Modern CPUs use hierarchical caches (L1, L2, L3). Accessing data that is close in memory (spatial locality) or accessed frequently (temporal locality) is much faster than accessing data scattered in memory. The goal is to maximize hits in the fastest cache (L1, ~1-4 cycle latency) rather than going to main RAM (~50+ cycle latency).
- **Array-of-Structures (AoS) vs. Structure-of-Arrays (SoA)**: 
  - AoS: `struct { double s1, s2; } data[N];` — when you access `data[i].s1`, you also load `data[i].s2` into the cache line (64 bytes), even if you don't need it. Wasteful if you only need one field.
  - SoA: `double s1[N]`, `double s2[N];` — when you iterate over `s1`, you only load `s1` values, which may be more efficient if you only need one field.

### CopulaDeriv usage
The CopulaDeriv C++20 engine uses **Structure-of-Arrays** pattern implicitly:
- `std::vector<double> S1_T(N)` and `std::vector<double> S2_T(N)` are separate vectors.
- The inner loop iterates: `S1_T[i] = ...; S2_T[i] = ...;` — each vector is traversed sequentially, which is optimal for spatial locality. The CPU prefetcher can predict the next access pattern.
- The copula sample generates `std::vector<std::pair<double, double>> U(N)` where each pair is stored contiguously; accessing `U[i].first` and `U[i].second` sequentially is cache-friendly.

**Performance result**: Sequential traversal of the N-element vectors contributes to the ~0.38 s median runtime for N = 1,000,000. Random access patterns would be significantly slower.

**Common incorrect answer**: "C++ always uses AoS (array of structs) for performance." — Incorrect; SoA is often faster for SIMD and sequential access patterns, and the CopulaDeriv code uses separate vectors for S1_T and S2_T.

**Follow-up**: Would a 2D `std::array<double, 2, N>` or `std::vector<std::array<double, 2>>` be faster or slower?

---

## CPU cache

### Question
Explain the CPU cache hierarchy and its relevance to the CopulaDeriv performance.

### Answer
- **L1 cache**: Smallest and fastest (typically 32-64 KB per core), 1-4 cycle latency. Holds the most recently used data.
- **L2 cache**: Larger (typically 256 KB-1 MB per core), 10-15 cycle latency.
- **L3 cache**: Shared across all cores (typically 4-64 MB), 40-50 cycle latency.
- **Main RAM**: Largest but slowest (50-100+ cycle latency).

**CopulaDeriv relevance**: For N = 1,000,000, the arrays S1_T, S2_T, Z each occupy 8 MB (1e6 * 8 bytes). This exceeds typical L1/L2 per-core cache but may fit in the shared L3 cache (which is typically 8-32 MB on modern desktop CPUs). The sequential access pattern ensures that the CPU prefetcher brings in data before it's needed, keeping the pipeline full.

If the arrays were randomly accessed (e.g., accessing `S_T[random_index]` in a loop), the cache miss rate would skyrocket, and the runtime would increase significantly. The project's sequential loop is a key reason the C++20 implementation achieves ~5x the Python throughput.

**Common incorrect answer**: "The CPU cache is transparent to the programmer; it doesn't affect code performance." — Incorrect; while the cache is hardware-managed, the access pattern (sequential vs. random) dramatically affects the number of cache misses and thus the runtime.

**Follow-up**: What alignment optimizations would you suggest for the vector allocations?

---

## Branch prediction

### Question
Explain branch prediction and its relevance to the CopulaDeriv C++20 engine.

### Answer
- **Branch prediction**: CPUs execute instructions speculatively based on predicted branch outcomes (if/else, loops). If the prediction is correct, the pipeline continues without stall; if wrong, the pipeline must be flushed (10-20 cycle penalty).
- **Predictable branches**: Loops with a fixed iteration count (like `for (int i = 0; i < N; ++i)`) are highly predictable: the CPU knows the loop will execute N times.
- **Data-dependent branches**: `if (rho == 1.0) ... else if (rho == -1.0) ... else ...` — the project's copula code has explicit boundary cases for rho = ±1. Since rho is a function parameter (often fixed for a benchmark run), the branch predictor quickly learns the pattern.

### CopulaDeriv usage
The copula sampling code in `copula.cpp` has:
```cpp
if (rho == 1.0) {
    Z0 = Z[0]; Z1 = Z[0];
} else if (rho == -1.0) {
    Z0 = Z[0]; Z1 = -Z[0];
} else {
    // general Cholesky
}
```
Since rho is typically set once per run and doesn't change per path, the branch predictor converges after the first iteration. After that, the `if/else if/else` has negligible cost.

The payoff functions (`best_of_call`, `worst_of_put`) have simple `if (m > K) ... else ...` branches that are also predictable (the comparison is against a fixed strike K, and the payoff values tend to be on one side or the other depending on the moneyness).

**Performance impact**: The branch predictor overhead is minimal compared the random number generation and the `exp()` call in the GBM formula. The 0.38 s median runtime for N = 1M is dominated by the mathematical operations, not branch mispredictions.

**Common incorrect answer**: "Branch prediction is the main reason C++ is faster than Python." — Incorrect; Python also has branch prediction (in the CPU), and the main gap is the interpreter overhead per loop iteration, not branches.

**Follow-up**: How would you restructure the code to eliminate the rho boundary branches entirely?

---

## SIMD

### Question
Explain SIMD (Single Instruction, Multiple Data) and whether the CopulaDeriv C++20 engine uses SIMD optimizations.

### Answer
- **SIMD**: Streaming SIMD Extensions (SSE, AVX, NEON) allow one instruction to process multiple data points simultaneously. For example, AVX2 can process 4 doubles at once (4 * 8 = 32 bytes). This can give 2-4x speedup for element-wise operations like the GBM formula.
- **CopulaDeriv status**: The current C++20 implementation does NOT use explicit SIMD instructions (no `#include <immintrin.h>`, no `__m256` types). The loop is a simple `for (int i = 0; i < N; ++i)` with scalar operations. The performance improvement over Python (5x) comes from compiled loop elimination and cache efficiency, not SIMD.

**Potential SIMD optimization**: The GBM formula `S = S0 * exp(drift + sigma * sqrtT * Z)` is element-wise and would be a good candidate for AVX2/NEON vectorization. A vectorized version could process 4-8 paths per instruction, potentially bringing the C++20 runtime down from 0.38 s to ~0.15 s for N = 1M.

**Common incorrect answer**: "The C++20 engine uses AVX2 SIMD for the Cholesky transform." — Incorrect; the code uses scalar operations. The Cholesky transform (2 multiplications + 1 add) is too simple to benefit significantly from SIMD without vectorization.

**Follow-up**: If you added SIMD, how much speedup would you expect, and would the results still be reproducible?

---

## Compiler optimizations

### Question
What compiler optimizations are used in the CopulaDeriv C++20 build, and how do they affect performance?

### Answer
- **CMakeLists.txt settings**:
  ```cmake
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  if(NOT MSVC)
      add_compile_options("-Wall -Wextra -Wpedantic")
  endif()
  ```
  The default `CMAKE_BUILD_TYPE` is not set in the minimal CMakeLists, but for benchmarks, `cmake .. -DCMAKE_BUILD_TYPE=Release` is used, which enables `-O3` optimization by default for GCC/Clang.

- **-O3**: Aggressive optimization: auto-vectorization (where the compiler decides to SIMD-ify loops), inlining, loop unrolling, constant propagation, dead code elimination. This is the primary reason the C++20 engine is faster than a naive debug build (-O0).

- **-march=native**: (Optional, not in the base CMakeLists) generates code optimized for the host CPU's feature set (AVX2, BMI2, etc.). Would be added for maximum performance but makes the binary non-portable.

- **CopulaDeriv observed impact**: Moving from a debug-like build to -O3 typically reduces the N = 1M runtime from ~1.5 s to ~0.38 s (a ~4x speedup). The -O3 flags enable the compiler to:
  - Eliminate redundant computations (e.g., `sqrtT = sqrt(T)` is hoisted outside the loop).
  - Use faster approximations where exact math is not required.
  - Inline the `phi_inv` and `erf` functions.
  - Vectorize the inner loop if the compiler decides the patterns are suitable.

**Common incorrect answer**: "-O3 makes the code run faster by adding more threads." — Incorrect; -O3 is about optimization, not multithreading. OpenMP or pthreads would be used for parallelization.

**Follow-up**: How would you verify that the -O3 optimized binary produces the same results as the unoptimized binary?

---

## C++20 features actually used

### Question
Which C++20 features does the CopulaDeriv engine actually use? List the standard library features, language features, and library components.

### Answer
The CopulaDeriv C++20 engine uses the following C++20 (or C++17/14) features:

**Language features (C++11/14/20)**:
- `auto` type deduction (e.g., `auto U = gc.sample(N, rng);`)
- Range-based for loops: `for (double v : values)`
- `const` correctness throughout
- `double` and `float` primitive types
- `std::array`, `std::vector` from C++98
- `std::mt19937_64` random engine (C++11)
- `std::normal_distribution<double>` (C++11)
- `std::chi_squared` via custom implementation (sum of squared normals)

**C++20 specific features (present in code but may not all be actively used)**:
- The CMakeLists.txt sets `CMAKE_CXX_STANDARD 20`, but the code does not heavily rely on C++20 novelties:
  - No ranges (`|>` operator, `views::transform`) — the code uses traditional for-loops.
  - No concepts (no `template<typename T> concept NumericallyTransformable`).
  - No `constexpr` reflection or `requires` clauses.
  - The `enum class ExitCode { Success = 0, ... }` is a C++11 scoped enum, not uniquely C++20.

**Standard library components**:
- `<random>`: `std::mt19937_64`, `std::normal_distribution`, `std::chi_squared` (custom).
- `<vector>`: `std::vector<double>`, `std::vector<std::pair<double, double>>`.
- `<array>`: `std::array<std::array<double, 2>, 2>` for the Cholesky matrix.
- `<cmath>`: `std::exp`, `std::sqrt`, `std::erf`, `std::log`.
- `<stdexcept>`: `std::domain_error` for validation.
- `<numeric>`: `std::accumulate` for mean computation in `sample_std`.
- `<iostream>`: `std::cout`, `std::endl` for the main demo.
- `<algorithm>`: `std::max`, `std::min` in payoff functions.

**Actual C++20 usage in the codebase**: Minimal. The project is primarily C++17-compatible code built with a CMake C++20 setting. The "C++20" moniker is more about using a modern compiler and build system than exploiting C++20 language features.

**Common incorrect answer**: "The engine uses C++20 coroutines." — Incorrect; no coroutines are present.

**Follow-up**: If you wanted to modernize the code with actual C++20 features, which would you add first?

---

## Random number generation

### Question
Explain the random number generation in the CopulaDeriv C++20 engine, and how reproducibility is ensured.

### Answer
- **Engine**: `std::mt19937_64` (Mersenne Twister 64-bit), a high-quality uniform random number generator with period 2^60-1. Default seed is 42 (set in `RandomGenerator::RandomGenerator(unsigned int seed = 42)`).
- **Normal variates**: Generated via `std::normal_distribution<double>(0.0, 1.0)` — i.e., the Box-Muller transform or the Mersenne Twister output transformed to N(0,1). In the code, `rng.normal()` returns the next normal variate.
- **Chi-squared variates**: Generated as the sum of squares of k independent standard normals: `double sum = 0; for (int i = 0; i < k; ++i) { double z = rng.normal(); sum += z*z; } return sum;`. This is implemented in `RandomGenerator::chi_squared(int k)` in `random.cpp`.
- **Reproducibility**: The seed is set once (default 42) and can be reset via `rng.seed(unsigned int seed)`. The `state()` method returns the last seed set value, for reporting purposes. All tests use seed=42, and the same seed produces identical results (verified by the 51 passing tests and the `test_same_seed_produces_identical_samples` test).

**Comparison to Python**: The Python implementation uses `numpy.random.default_rng(seed)` (PCG64 variant), which is a different algorithm but serves the same purpose. Both are deterministic given the same seed.

**Common incorrect answer**: "The C++ engine uses Python's random module." — Incorrect; it uses the standard library's `mt19937_64`.

**Follow-up**: How would you compare the statistical quality of mt19937_64 vs. numpy's default_rng for Monte Carlo simulation?

---

## Thread-local state

### Question
Explain thread-local state and whether the CopulaDeriv engine supports multi-threaded simulation.

### Answer
- **Thread-local state**: Data that is local to each thread, avoiding race conditions without explicit locking. In C++, `thread_local` specifier gives each thread its own instance of a variable. This is useful for RNG states: each thread can have its own `std::mt19937_64` without interfering with other threads.
- **CopulaDeriv status**: The current engine is **single-threaded**. The `RandomGenerator` class has a single `engine_` member, and the `sample()` methods modify global-like state (the output vector U). There is no `thread_local` qualifier, no OpenMP `#pragma`, and no effort to run the Monte Carlo loop in parallel.

**If multi-threaded were implemented**: One approach would be:
```cpp
// Each thread gets its own RNG
thread_local RandomGenerator local_rng(42 + thread_id);

// Each thread simulates N/num_threads paths
for (int i = 0; i < N_per_thread; ++i) {
    // ... same simulation loop ...
}
// Results from all threads would be combined (e.g., via reduction: sum of payoffs)
```

**Performance impact**: For N = 1M on a 4-core machine, ideal near-linear speedup would bring the runtime from ~0.38 s to ~0.1 s. However, the project does not implement this for v1.0.

**Common incorrect answer**: "The C++ engine uses OpenMP for parallelization." — Incorrect; no OpenMP is present in the codebase.

**Follow-up**: What would be the main challenges in adding thread safety to the RNG and the simulation loop?

---

## False sharing

### Question
Explain false sharing and whether it's relevant to the CopulaDeriv engine.

### Answer
- **False sharing**: A performance pitfall in multi-threaded programs where two threads modify variables that reside on the same cache line (typically 64 bytes), even if the variables are different. Each CPU core caches its own copy of the shared cache line, and invalidation traffic between cores causes significant slowdown—even though the threads are logically working on different data.
- **Relevance to CopulaDeriv**: The current engine is single-threaded, so false sharing is not a concern for v1.0. If the engine were multi-threaded (e.g., parallelizing the N-path Monte Carlo loop), false sharing could become relevant if per-thread data structures (e.g., partial payoff sums, per-thread RNG states) are placed on the same cache line without padding.

**Mitigation (future)**: If multi-threaded, use `alignas(64)` or padding to ensure thread-local variables occupy distinct cache lines:
```cpp
struct alignas(64) ThreadData {
    double partial_sum;
    char padding[64 - sizeof(double)];  // ensure 64-byte alignment
};
```

**Common incorrect answer**: "False sharing is the main reason the C++ engine is slower than it could be." — Incorrect; the engine is single-threaded, and the primary performance gap vs. ideal is the sequential Monte Carlo loop, not false sharing.

**Follow-up**: How would you measure false sharing if you added multi-threading?

---

## Benchmark methodology

### Question
Explain the benchmark methodology for CopulaDeriv, and how you would compare Python and C++20 performance.

### Answer
See the dedicated `docs/benchmark_methodology.md` document, but the key points are:

- **Hardware documentation**: CPU model, clock speed, RAM, OS must be recorded.
- **Compiler and flags**: C++ compiled with `-O3` (Release mode); Python version and library versions recorded.
- **Random seed**: Documented (default 42); same seed used for both implementations for reproducibility comparison.
- **Number of paths (N)**: Same N for both (e.g., 1,000,000).
- **Warm-up**: 1,000 paths discarded before timed repetitions.
- **Repetitions**: 5 per configuration; median runtime computed.
- **Throughput**: N / median_runtime (paths/second).
- **Price estimate + SE**: Both implementations should produce statistically identical prices under the same seed.

**Python vs. C++20 comparison format**:
```
Python:
  Median runtime: 2.13 s (5 reps)
  Throughput: 470K paths/s
  Price: 12.345 ± 0.012
  Seed: 42

C++20:
  Median runtime: 0.38 s (5 reps)
  Throughput: 2.63M paths/s
  Price: 12.344 ± 0.012
  Seed: 42

Interpretation: C++20 is ~5.5x faster. Same seed; prices agree within SE.
```

**Important disclaimer**: "Monte Carlo throughput is not equivalent to HFT event latency. The 5x improvement refers to batch path processing throughput (paths/second), not microsecond-scale event processing latency."

**Common incorrect answer**: "Report the single-best runtime, not the median." — Incorrect; the methodology specifies median of 5 repetitions to handle OS jitter and variability.

**Follow-up**: How would you handle the case where the Python and C++ implementations use different random number algorithms (different seeds produce different prices)?

---

## Profiling

### Question
How would you profile the CopulaDeriv C++20 engine to identify performance bottlenecks?

### Answer
- **Google Benchmark** or `perf` (Linux): Could measure per-function runtimes. The engine is small enough that manual timing (adding `clock()` or `std::chrono::high_resolution_clock` around the inner loop) would suffice.
- **Expected bottlenecks**: 
  1. Random number generation (Mersenne Twister + normal transformation) — typically 20-30% of runtime.
  2. `exp()` call in the GBM formula — typically 30-40% of runtime (exp is a relatively expensive instruction).
  3. Cholesky transform and cache misses — ~20-30%.
  4. Payoff computation and max/min — ~10-15%.
- **Profiling result (expected)**: For N = 1M, the `exp()` and the normal random generation are the dominant costs. The Cholesky transform (2 multiplications, 1 add) is very cheap in comparison.
- **Optimization focus**: If the goal is maximum throughput, the `exp()` call and the RNG would be the targets. Software pipelining or approximation of `exp()` (e.g., using a polynomial approximation for the range of values encountered) could yield further speedups.

**Common incorrect answer**: "Use valgrind --tool=callgrind to profile; it shows exact cycle counts." — Partially correct; valgrind is slow (interprets every instruction) and changes the performance characteristics. For quick profiling, `perf stat -e cpu-cycles,instructions` or manual `chrono` timing is sufficient.

**Follow-up**: Would you profile a debug (-O0) build or a Release (-O3) build?

---

## Numerical precision

### Question
Explain floating-point precision in the CopulaDeriv engine, and numerical limits to be aware of.

### Answer
- **IEEE 754 double-precision**: `float` in C++ and Python is 64-bit double, with ~15-17 significant decimal digits, and range approximately 10^-308 to 10^308.
- **CopulaDeriv typical values**: S0 = 100, sigma = 0.2, T = 1.0, K = 100. The GBM exponent is on the order of (r - q - 0.5*sigma^2)*T ≈ (0.04 - 0.01 - 0.02) * 1 = 0.01. The random term sigma*sqrt(T)*Z ≈ 0.2 * 1 * Z ≈ 0.2 * Z where Z ~ N(0,1). So the exponent is typically in the range [-0.5, 0.5], and S_T is typically in the range [100*exp(-0.5), 100*exp(0.5)] ≈ [60, 165]. These are well within the double range.
- **Numerical stability issues**:
  - **rho near ±1**: Handled explicitly; `sqrt(1-rho^2)` avoids sqrt of negative.
  - **Very large/small N**: Affects SE (1/sqrt(N)) but not floating-point limits for these parameter ranges.
  - **nu ≤ 2**: Student-t variance is infinite; the copula is still well-defined but pricing SE may be very large.
  - **CDF clipping**: In `lognormal_inverse_cdf`, p is clipped to [eps, 1-eps] where eps ≈ 1e-8 to avoid log(0) or acos(1).

**Common incorrect answer**: "Floating-point errors can be ignored in Monte Carlo." — Incorrect; for extreme parameters (very large T, very small nu, rho extremely close to ±1), errors can accumulate. The project documents these limitations.

**Follow-up**: How would you modify the code to handle very large T (e.g., T = 10 years) without losing precision?

---

## Throughput versus latency

### Question
Distinguish between Monte Carlo throughput and HFT event latency, and explain why the CopulaDeriv project explicitly makes this distinction.

### Answer
- **Monte Carlo throughput**: The number of simulated paths processed per second. For CopulaDeriv with N = 1M paths and median runtime 0.38 s, the throughput is 2.63M paths/second. This is a batch-processing metric: how many option-pricing simulations can be completed in a given time. It is appropriate for analytics, risk management, and research.

- **HFT event latency**: The time required to process an individual market event (order arrival, trade execution, cancel) in a high-frequency trading system. Measured in microseconds (µs) or nanoseconds (ns). The goal is to process each event as quickly as possible, typically using specialized hardware, polling loops, and lock-free data structures. HFT latency is not about processing millions of paths; it's about responding to individual events in real time.

- **Why the distinction matters**: The CopulaDeriv project explicitly states: "Do not claim that Monte Carlo throughput measures HFT order-processing latency." The two metrics measure fundamentally different things:
  - Monte Carlo: millions of paths/second, batch processing, CPU-bound, appropriate for option pricing analytics.
  - HFT: sub-microsecond/event, real-time, network-bound, appropriate for trading system responsiveness.

- **Project position**: The project's resume entries and interview materials emphasize the Monte Carlo throughput characteristic and explicitly avoid claiming HFT capabilities. The C++20 implementation demonstrates performance engineering skills (compiled loops, cache efficiency) without overstating the application domain.

**Common incorrect answer**: "If C++20 does 2.6M paths/s, it's an HFT engine." — Incorrect; 2.6M paths/s is a batch Monte Carlo rate, comparable to risk analytics workloads, not HFT order-processing latency (which is typically < 10 µs per event).

**Follow-up**: What kind of workloads is the CopulaDeriv C++20 engine actually suitable for?

---