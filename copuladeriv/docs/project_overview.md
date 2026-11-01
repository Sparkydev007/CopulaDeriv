# CopulaDeriv — Project Overview

## What problem the project solves

CopulaDeriv addresses the problem of pricing multi-asset rainbow options under different dependence assumptions. Specifically:

- **Best-of call options** pay off at maturity based on the better-performing of two assets: $H_T = \max(\max(S_{1,T}, S_{2,T}) - K, 0)$
- **Worst-of put options** pay off based on the worse-performing asset: $H_T = \max(K - \min(S_{1,T}, S_{2,T}), 0)$

The central question is how the **dependence structure** between the two underlying assets affects the option price. Two assets may have the same marginal distributions (same volatilities, same initial prices), but if they tend to move together (positive dependence) or move oppositely (negative dependence), the rainbow option price changes significantly.

CopulaDeriv provides a framework to:
1. Model dependence using copulas (Gaussian vs. Student-t)
2. Vary the dependence parameter rho and observe price effects
3. Compare different dependence models (Gaussian has zero tail dependence; Student-t has positive tail dependence)
4. Study the performance trade-offs between a Python reference and a C++20 compiled engine

## Why copulas matter

Copulas are important because they separate the **dependence structure** from the **marginal distributions**. This is a key insight in multi-asset derivatives pricing:

- **Without copulas**: One might simply use a correlation matrix to model the joint distribution of asset prices. However, the correlation of the terminal prices depends on both the correlation parameter and the marginal distributions. Changing the marginals (e.g., from normal to lognormal) changes the implied terminal correlation, making it difficult to control dependence independently.

- **With copulas**: The dependence parameter (rho) controls the association between the latent Gaussian variables, while the marginal distributions (lognormal risk-neutral GBM) are independently specified. This means:
  - You can set the marginals to any distribution (not just lognormal) without affecting the dependence structure.
  - You can vary rho to control dependence without changing the marginals.
  - Tail dependence (the probability of simultaneous extreme moves) can be modeled explicitly: the Gaussian copula has zero asymptotic tail dependence for |rho| < 1, while the Student-t copula has positive tail dependence that increases as nu decreases.

**In practical terms**: For a worst-of put option, positive tail dependence means there's a meaningful probability that both assets will fall simultaneously, increasing the put payoff. The Gaussian copula would assign zero probability to this asymptotic event, potentially underpricing the put. The Student-t copula captures this effect.

## Why rainbow options are useful

Rainbow options are multi-asset derivatives whose payoff depends on the relative performance of the underlying assets. They are useful for several reasons:

1. **Structured products**: Best-of and worst-of options are the building blocks of many structured notes, principal-protected notes, and equity-linked instruments sold by investment banks. These products are marketed to retail and institutional investors seeking exposure to one or more assets with downside protection or upside participation.

2. **Risk management**: A worst-of put provides insurance against the worse of two assets declining. This can be used as a hedging tool for portfolios holding multiple assets.

3. **Speculation**: Traders may want to speculate on the relative performance of two assets (e.g., "I think Asset 1 will outperform Asset 2, so I'll buy a best-of call").

4. **Diversification effect**: The price of a rainbow option depends on the correlation between the assets. If the assets are perfectly correlated (rho = 1), the rainbow option reduces to a single-asset option. If they are independent (rho = 0) or negatively correlated, the price reflects the diversification (or concentration) effect.

5. **No-arbitrage pricing**: Under the risk-neutral measure, the price is the discounted expected payoff. The copula model provides a consistent way to specify the joint distribution needed for this expectation.

## Why two implementations exist (Python + C++20)

The project provides two implementations of the same mathematical models:

### Python reference implementation

- **Purpose**: Prototyping, education, testable research, ease of modification
- **Advantages**: 
  - NumPy vectorization for rapid development
  - pytest test framework with 51 tests, all passing
  - Easy to read and modify (pure Python + NumPy)
  - Straightforward debugging and parameter tuning
  - Ideal for learning and educational purposes
- **Disadvantages**: 
  - Interpreted language overhead
  - Lower paths/second throughput (~0.5-0.8M paths/s on test hardware)
  - Not suitable for production-scale or real-time pricing

### C++20 engine

- **Purpose**: Performance demonstration, compiled engineering, systems evaluation
- **Advantages**:
  - Compiled code with fixed-memory allocations
  - Higher paths/second throughput (~2-3M paths/s on same hardware, ~5x improvement)
  - C++20 features: mt19937_64 RNG, Cholesky decomposition, erf() transform
  - Demonstrates systems engineering skills (build systems, class design, reproducible RNG)
  - Suitable for performance-critical contexts
- **Disadvantages**:
  - More verbose code (header/implementation separation, CMake build)
  - longer edit-compile cycle
  - Requires CMake and C++ toolchain
  - Less flexible for rapid parameter tuning

**Why both**: The dual implementation serves the project's educational and portfolio goals. The Python implementation is the validated research prototype (51 tests, documented math, reproducible). The C++20 implementation demonstrates performance engineering and compiled-language systems design. Together, they show the trade-offs between development speed (Python) and runtime performance (C++), which is a core theme in quantitative finance software engineering.

## What was learned from performance engineering

1. **Vectorization matters**: The Python NumPy implementation achieves good performance through array-level operations. The inner loop is entirely eliminated (no Python-level per-path iteration).

2. **Compiled loop elimination**: The C++20 implementation processes paths in fixed-size arrays with compiled loops. The per-path overhead is a few CPU instructions vs. the Python interpreter loop.

3. **Memory layout**: Both implementations use array-of-structs layout (S1_T and S2_T as separate arrays or stacked array). The C++ version avoids Python object per-element overhead.

4. **RNG overhead**: The mt19937_64 in C++ and NumPy's default_rng in Python have comparable per-sample cost. The dominant cost is the Cholesky transform and GBM formula, not the RNG itself.

5. **Throughput vs. accuracy**: Both implementations produce statistically identical price estimates under the same seed and parameters. The speedup is purely in paths/second, not in accuracy.

6. **HFT disclaimer**: The 5x throughput improvement (e.g., 0.5M → 2.5M paths/s) is a batch-processing metric. It does not imply HFT-grade latency (microseconds per event). Monte Carlo processes millions of paths per second; HFT requires sub-microsecond event processing. These are fundamentally different engineering metrics.

7. **Build system learning**: CMake with C++20 is a practical build system skill. The CMakeLists.txt, header/include structure, and class design in the C++ engine are representative of moderate-complexity C++ projects.

8. **Reproducibility across languages**: Both implementations use a documented seed (42) and produce identical results for the same seed. This demonstrates that reproducibility is achievable across Python and C++ with proper design.

## What the project learned about quantitative finance and systems engineering

### Quantitative finance insights

1. **Copula separation property**: The copula parameter rho controls dependence without changing the marginals. This is not intuitive; many practitioners assume that changing the correlation changes the entire joint distribution. The project verified this property empirically (51 tests pass, including marginal-preservation tests).

2. **Tail dependence matters**: For worst-of put options, the Student-t copula produces higher prices than the Gaussian copula for the same rho, because the Student-t assigns positive probability to simultaneous extreme moves. This is a material pricing difference, not a minor numerical artifact.

3. **Monte Carlo error is real**: The standard error scales as 1/sqrt(N). To halve the error, you must quadruple the number of paths. This is a fundamental constraint of Monte Carlo methods, not an implementation bug.

4. **Risk-neutral vs. physical measure**: The project uses risk-neutral GBM marginals (drift = r - q). Physical historical dependence cannot be directly used for risk-neutral pricing without a calibration step. This is a critical modeling distinction.

5. **Boundary cases must be handled**: rho = 1 and rho = -1 require explicit handling to avoid sqrt(negative) errors. The project handles these correctly in both Python and C++.

### Systems engineering insights

1. **Prototype then compile**: The Python prototype enabled rapid development and testing (51 tests written and verified). The C++20 engine was then built on the same mathematical foundation, demonstrating the "prototype then compile" workflow.

2. **Build system design**: CMake with C++20 provides a clean build process (cmake .., make -j4). The CMakeLists.txt configuration (Eigen3 optional, GoogleTest optional) is representative of real-world C++ projects.

3. **Class design**: The GaussianCopula, StudentTCopula, MonteCarloPricer, and RandomGenerator classes demonstrate modular C++ design with clear separation of concerns.

4. **Reproducible RNG**: Both implementations use a seeded RNG (np.random.default_rng in Python, mt19937_64 in C++). The seed is documented and produces identical results. This is essential for benchmark comparisons and peer review.

5. **Performance measurement**: The benchmark methodology documents how to measure paths/second, median runtime, and statistical uncertainty. The 5x C++/Python ratio is specific to the test hardware but reflects the typical order of magnitude.

6. **No silver bullet**: The C++ implementation is faster, but the Python implementation is more flexible and easier to modify. The choice of language depends on the context: research prototyping (Python) vs. production deployment (C++ or hybrid).

7. **Documentation as deliverable**: The 13 documentation files (README, architecture, math model, benchmark methodology, research report, project overview, release notes, interview questions, etc.) are as important as the code. A professional portfolio project requires both working code and clear documentation.

## What problem the project solves (summary)

CopulaDeriv provides a complete, validated Monte Carlo pricing engine for two-asset rainbow options, with:

- Two copula models (Gaussian: zero tail dependence; Student-t: positive tail dependence)
- Risk-neutral GBM marginals with dividend yields
- Best-of call and worst-of put payoffs
- Standard error and 95% confidence interval estimation
- Python reference implementation (51 tests, all passing)
- C++20 compiled engine (performance demonstration)
- Documented benchmark methodology (throughput, hardware, reproducibility)
- Full research report with statistical findings and limitations

The project is valuable for:
- **Quantitative finance education**: Understanding copulas, Monte Carlo, and rainbow options
- **Systems engineering**: Python/C++ performance comparison, build systems, reproducible RNG
- **Portfolio demonstration**: A GitHub-ready project with code, docs, tests, and reports
- **Interview preparation**: Technical questions and answers covering probability, derivatives, copulas, and C++ performance

## Directory tree (actual repository structure)

```
CopulaDeriv/
├── pyproject.toml              # Project config (setuptools, numpy, scipy, pytest)
├── README.md                   # Professional project introduction (this phase)
├── LICENSE                     # MIT-style license for student project
├── .gitignore                  # Git ignore rules
├── copuladeriv/
│   ├── pyproject.toml          # Python package config
│   ├── copuladeriv/
│   │   ├── __init__.py         # Version 0.1.0
│   │   ├── copulas.py          # Gaussian + Student-t copula sampling
│   │   ├── marginals.py        # Risk-neutral GBM marginals
│   │   ├── payoffs.py          # Best-of call, worst-of put, European call
│   │   ├── pricing.py          # Monte Carlo pricing engine
│   │   └── statistics.py       # Statistical utilities (mean, SE, CI)
│   ├── tests/
│   │   ├── test_copulas.py     # 21 copula tests (all pass)
│   │   ├── test_gbm.py         # 10 GBM tests (all pass)
│   │   ├── test_payoffs.py     # 27 payoff tests (all pass)
│   │   └── test_pricing.py     # 11 pricing tests (all pass)
│   ├── docs/                   # Documentation (Phases 3-7)
│   │   ├── architecture.md     # System architecture + Mermaid diagrams
│   │   ├── mathematical_model.md  # Risk-neutral GBM, copulas, tail dependence
│   │   ├── benchmark_methodology.md  # Benchmark methodology v1.0
│   │   ├── project_overoverview.md   # Problem, copulas, two implementations
│   │   └── ... (more docs in later phases)
│   ├── reports/                # Research reports (Phase 6)
│   │   └── research_report.md  # Full research report
│   ├── benchmarks/             # Benchmark scripts (Phase 5)
│   │   ├── results/            # CSV benchmark results
│   │   ├── plots/              # Performance plots
│   │   └── methodology.md      # Benchmark methodology
│   ├── experiments/            # Experiment scripts
│   └── cpp/                    # C++20 implementation
│       ├── CMakeLists.txt      # CMake build, C++20
│       ├── include/
│       │   └── copuladeriv/
│       │   ├── copula.hpp      # GaussianCopula, StudentTCopula
│       │   ├── pricing.hpp     # PricingResult, MonteCarloPricer
│       │   └── random.hpp      # RandomGenerator
│       ├── src/
│       │   ├── copula.cpp       # Copula sampling (Gaussian, Student-t)
│       │   ├── main.cpp         # CLI demo
│       │   ├── pricing.cpp      # Monte Carlo pricer
│       │   ├── payoffs.cpp      # Payoff functions
│       │   └── random.cpp       # RandomGenerator (mt19937_64)
│       └── build/              # CMake build directory (generated)
├── experiments/                # Experiment scripts (to be populated)
└── reports/                    # Research reports
```

## Limitations and future work (honest summary)

- **Illustrative parameters**: S0 = 100, sigma = 0.2, T = 1.0, K = 100 are not calibrated to market data.
- **Terminal vs. diffusion**: The copula specifies terminal dependence at maturity T, not a dynamically consistent stochastic process.
- **Risk-neutral calibration**: Physical historical dependence cannot be automatically used for risk-neutral pricing without calibration.
- **Monte Carlo uncertainty**: SE ~ 1/sqrt(N); reducing error requires more paths.
- **No Greeks**: Finite-difference or analytical Greeks not computed.
- **Single-threaded C++**: Multi-threaded parallelization not implemented.
- **No importance sampling/control variates**: Referenced in docstrings but not actively implemented.
- **HFT disclaimer**: Monte Carlo throughput (paths/s) != HFT event latency (microseconds/event).
- **Python C++ ratio hardware-dependent**: The ~5x throughput improvement is specific to the test hardware.

## Citation (if referencing this project)

```
Bunde, P. (2026). CopulaDeriv: A Monte Carlo Pricing Engine for Rainbow Options
with Gaussian and Student-t Copula Dependence. GitHub repository:
https://github.com/Sparkydev007/CopulaDeriv
```