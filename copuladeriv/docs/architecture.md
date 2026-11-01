# CopulaDeriv — Architecture Documentation

## 1. High-level system architecture

The project has two implementations that share the same mathematical models:

### Python reference implementation

- **Location**: `copuladeriv/python/copuladeriv/`
- **Entry point**: `copuladeriv.__init__`
- **Package structure**:
  - `copulas.py` — Gaussian and Student-t copula sampling
  - `marginals.py` — Risk-neutral GBM terminal distributions
  - `payoffs.py` — Best-of call, worst-of put, European call payoffs
  - `pricing.py` — Monte Carlo pricing engine
  - `statistics.py` — Statistical utilities (mean, SE, CI)
  - `tests/` — 51 unit and statistical tests
- **Random number generation**: NumPy `default_rng` with reproducible seed
- **Statistical distributions**: SciPy (norm.cdf, norm.ppf, t.cdf, t.ppf, chi-square)
- **Build/dependency**: `pyproject.toml` with numpy, scipy, pytest, matplotlib dependencies

### C++20 engine

- **Location**: `copuladeriv/cpp/`
- **Build system**: CMake 3.14+ with C++20 standard
- **Source files** (5 .cpp files, 3 .hpp headers):
  - `src/main.cpp` — CLI entry point, basic test demo
  - `src/copula.cpp` — Gaussian and Student-t copula sampling
  - `src/pricing.cpp` — Monte Carlo pricer (best-of call, worst-of put)
  - `src/payoffs.cpp` — Payoff function implementations
  - `src/random.cpp` — Random generator (mt19937_64, chi-squared)
- **Headers** (3 .hpp files):
  - `include/copuladeriv/copula.hpp` — GaussianCopula, StudentTCopula class definitions
  - `include/copuladeriv/pricing.hpp` — PricingResult struct, MonteCarloPricer class, payoff structs (BestOfCall, WorstOfPut, EuropeanCall)
  - `include/copuladeriv/random.hpp` — RandomGenerator class, generate_normals()
- **Dependencies**: Eigen3 (optional, for matrix operations), GoogleTest (optional, configured in CMakeLists.txt)
- **Random number generation**: std::mt19937_64 with reproducible seed

### Shared mathematical model (Python and C++)

Both implementations share the same core formulas:

- **Risk-neutral GBM terminal**: S_i(T) = S_i(0) exp((r - q_i - 0.5*sigma_i^2)*T + sigma_i*sqrt(T)*Z_i)
- **Gaussian copula**: Cholesky transformation of independent normals, then U = Phi(Z)
- **Student-t copula**: Shared chi-square scale per observation, Cholesky transformation, Student-t CDF transform
- **Payoffs**: Best-of call (max(max(S1,S2) - K, 0)), worst-of put (max(K - min(S1,S2), 0))
- **Pricing**: Discounted mean payoff with standard error and 95% CI

## 2. Python reference architecture

```
copuladeriv.python package
│
├── __init__.py         # Package metadata, version "0.1.0"
│
├── copulas.py          # Core copula implementations
│   │
│   │   gaussian_copula_sample(rho, N, seed, method)
│   │   │   └─ method: "cholesky" (default) or "direct"
│   │   │   └─ Returns U ~ Gaussian_copula(rho): shape (N, 2), Uniform(0,1) marginals
│   │
│   │   student_t_copula_sample(rho, nu, N, seed, method)
│   │   │   └─ method: "shared_chi_square" (default) or "independent"
│   │   │   └─ Critical: shared V ~ chi-squared(nu) per observation
│   │   │   └─ Returns U ~ StudentT_copula(rho, nu): shape (N, 2), Uniform(0,1) marginals
│   │
│   │   empirical_correlation(U)      # Pearson correlation of uniform columns
│   │   │
│   │   approximate_pearson_rho(gaussian_rho)  # Reference with documentation of distinction
│   │   │
│   │   student_t_tail_dependence_theoretical(nu, rho)  # lambda_U, lambda_L
│   │   │
│   │   gaussian_tail_dependence(nu, rho)  # (0, 0) for |rho| < 1
│   │
│   │   empirical_tail_dependence_upper(U, threshold)  # empirical lambda_U
│   │   │
│   │   empirical_tail_dependence_low(U, threshold)    # empirical lambda_L
│   │
├── marginals.py        # Risk-neutral marginal distributions
│   │
│   │   risk_neutral_lognormal_terminal(S0, r, q, sigma, T, Z)  # S_i(T) = S0*exp((r-q-0.5*sigma^2)*T + sigma*sqrt(T)*Z)
│   │
│   │   discount_factor(r, T)  # exp(-r*T)
│   │
│   │   forward_price(S0, r, q, sigma, T)  # S0*exp((r-q)*T)
│   │
│   │   lognormal_inverse_cdf(p, S0, r, q, sigma)  # Stable quantile function
│   │
│   │   lognormal_moment(S0, r, q, sigma, T, k)  # E[S^k]
│   │
│   │   expected_terminal_mean(S0, r, q, sigma, T)  # E[S] = S0*exp((r-q)*T)
│   │
│   │   expected_terminal_second_moment(S0, r, q, sigma, T)  # E[S^2]
│   │
│   │   expected_terminal_variance(S0, r, q, sigma, T)  # Var[S]
│   │
│   │   sample_std(values)  # ddof=1 sample standard deviation
│   │
├── payoffs.py          # Option payoff functions
│   │
│   │   best_of_call(S1_T, S2_T, K)  # max(max(S1,S2) - K, 0)
│   │
│   │   worst_of_put(S1_T, S2_T, K)  # max(K - min(S1,S2), 0)
│   │
│   │   european_call(S_T, K)  # max(S_T - K, 0)
│   │
│   │   simulate_terminal_prices(S1_0, S2_0, r, q1, q2, sigma1, sigma2, T, rho, N, seed)  # Cholesky GBM
│   │
├── pricing.py          # Monte Carlo pricing engine
│   │
│   │   monte_carlo_best_of_call_price(...)  # Gaussian copula pricing
│   │
│   │   monte_carlo_worst_of_put_price(...)  # Gaussian copula pricing
│   │
│   │   monte_carlo_best_of_call_price_copula(...)  # Gaussian copula variant
│   │
│   │   monte_carlo_worst_of_put_price_copula(...)  # Gaussian copula variant
│   │
│   │   monte_carlo_best_of_call_price_student_t(...)  # Student-t copula pricing
│   │
│   │   monte_carlo_worst_of_put_price_student_t(...)  # Student-t copula pricing
│   │
│   │   black_scholes_european_call(S0, K, r, q, sigma, T)  # Analytical benchmark
│   │
│   │   price_se_ci(price, se, confidence)  # CI from price and SE
│   │
│   │   bs_agreement_check(mc_price, bs_price, mc_se, tolerance_multiplier)  # BS comparison
│   │
├── statistics.py       # Statistical utilities
│   │
│   │   price_se_ci(price, se, confidence)  # CI: price ± z*se
│   │
│   │   sample_mean(values)  # np.mean
│   │
│   │   sample_std(values)  # np.std(ddof=1)
│   │
│   │   price_convergence(prices_by_N, se_by_N)  # Convergence summary
│   │
│   │   bs_agreement_check(...)  # Already in pricing.py; duplicate reference
│   │
└── __main__.py or cli entry point  # Not yet implemented
```

## 3. C++20 engine architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MonteCarloPricer class                       │
│        ┌─────────────────────┐  ┌─────────────────────────┐   │
│        │  Parameters         │  │  Members (copulas)      │   │
│        ├─────────────────────┤  ├─────────────────────────┤   │
│        │ S1_0, S2_0        │  │  GaussianCopula copula_ │   │
│        │ r, q1, q2         │  │  StudentTCopula student_│   │
│        │ sigma1, sigma2    │  │                          │   │
│        │ T, rho, nu, K     │  │                          │   │
│        └─────────────────────┘  └─────────────────────────┘   │
│               │                       │                     │
│       ┌─────────────────┐     ┌─────────────────┐             │
│       │  validate()     │     │  set_rho(rho)   │             │
│       │  check params   │     │  update rho     │             │
│       └─────────────────┘     └─────────────────┘             │
│               │                       │                     │
│    ┌──────────▼──────────┐   ┌──────────▼──────────┐   │
│    │  generate_terminal_prices()  │  │  price(N, seed, model) │
│    │  ── copula.sample(N, rng)  │   │  ── generate_terminal_prices │
│    │  ── phi_inv(u)               │   │  ── payoff computation │
│    │  ── Cholesky transform       │   │  ── price_from_payoffs │
│    │  ── GBM formula              │   │  │                     │
│    └────────────────────────────┘   └────────────────────────────┘   │
│                │                       │                     │
│    ┌──────────▼───────────────┐ ┌─────────▼───────────────┐ │
│    │  price_worst_of_put()      │ │  price_from_payoffs()   │ │
│    │  ── same pipeline          │ │  ── discounted stats    │ │
│    └────────────────────────────┘ └────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Key C++20 functions

- **GaussianCopula::sample(N, rng)**: Generate N uniform pairs via Cholesky + erf() transform
- **StudentTCopula::sample(N, rng)**: Generate N uniform pairs with shared chi-square scale
- **student_t_tail_dependence(rho, nu)**: Return (lambda_L, lambda_U) = (lambda, lambda)
- **MonteCarloPricer::price(N, seed, model)**: Full pricing pipeline
  - model = "gaussian": Gaussian copula
  - model = "student_t": Student-t copula
- **generate_terminal_prices(rng, N, copula_type, S1_0, S2_0, r, q1, q2, sigma1, sigma2, T, rho)**: Generate S1_T, S2_T pair
  - Falls back to GaussianCopula if student_t not fully integrated
- **phi_inv(p)**: Rational approximation of inverse standard normal CDF (probit), accurate to ~1e-7
- **price_from_payoffs(payoffs, r, T)**: Discounted mean + SE + 95% CI

### Random generator

- **RandomGenerator** (class, random.hpp):
  - std::mt19937_64 engine with reproducible seed (default 42)
  - normal() → N(0,1) variate
  - chi_squared(k) → chi-squared(k) as sum of k squared normals
  - seed(unsigned int) → reset seed
  - state() → last seed set value

## 4. Random-number generation

### Python

- **numpy.random.default_rng(seed)** — Modern numpy RNG (PCG64 variant)
- **Seed=42** used consistently across all tests for reproducibility
- **Vectorized**: All RNG acts on arrays of size N; no Python-level loops in inner loops

### C++20

- **std::mt19937_64** — Mersenne Twister 64-bit engine
- **RandomGenerator** wrapper class with seed() for reproducibility
- **normal_distribution<double>(0.0, 1.0)** → N(0,1) variates
- **chi-squared generation**: Sum of squares of k independent standard normals
- **State reporting**: `state()` returns the last seed set value (for reproducibility logging)

## 5. Copula sampling

### Gaussian copula (both Python and C++)

**Construction** (per observation):

1. Draw Z ~ N(0, I_2) — two independent standard normals
2. Apply Cholesky: X = L @ Z where L = [[1, 0], [rho, sqrt(1-rho^2)]]
   - Z_0 = z_0 (first normal becomes first asset shock)
   - Z_1 = rho*z_0 + sqrt(1-rho^2)*z_1 (second normal becomes correlated shock)
3. Transform to uniforms: U = Phi(X) = 0.5 * (1 + erf(x / sqrt(2)))

**Boundary cases**:
- rho = 1: Z_1 = Z_0 (perfect positive dependence)
- rho = -1: Z_1 = -Z_0 (perfect negative dependence)
- |rho| < 1: General Cholesky case

### Student-t copula (both Python and C++)

**Critical implementation detail**: A SINGLE chi-squared variate V ~ chi-squared(nu) is drawn per observation and shared by BOTH assets. This is what creates the intended multivariate Student-t dependence.

**Construction** (per observation i):

1. Draw V_i ~ chi-squared(nu) — ONE per observation, shared by both assets
2. Generate Z_i = [z0, z1] i.i.d. ~ N(0,1)
3. Apply Cholesky: X_i = L @ Z_i (correlated normals with latent correlation rho)
4. Scale by shared mixture: Y_i = X_i / sqrt(V_i / nu)
   - This is the scale-mixture-of-Gaussians representation
   - X_i / sqrt(V_i/nu) ~ multivariate Student-t with nu df
5. Transform to uniforms: U_i = T_nu(Y_i) (univariate Student-t CDF)

**Why shared scale matters**: Using independent W1, W2 per asset would destroy the multivariate Student-t property. The shared V creates the correlation induced by the shared scale mixture.

**Boundary cases**:
- rho = 1: Z_1 = Z_0 (perfect positive latent correlation)
- rho = -1: Z_1 = -Z_0 (perfect negative latent correlation)
- nu → ∞: Converges to Gaussian copula (lambda_U = lambda_L → 0)

### Comparison: Gaussian vs Student-t

| Aspect | Gaussian copula | Student-t copula |
|--------|----------------|------------------|
| Scale model | No scale mixture | One shared V ~ chi-squared(nu) per observation |
| Dependence source | Latent correlation rho only | Latent rho + shared scale V |
| Tail dependence | lambda_U = lambda_L = 0 for |rho| < 1 | lambda_U = lambda_L > 0 for finite nu |
| Limiting case | — | As nu → ∞, converges to Gaussian |
| Variance | Finite for all rho | Finite only for nu > 2 |
| Empirical correlation | Matches rho within sampling error | Matches rho within sampling error |

## 6. Marginal transformation

Both implementations follow the same two-step process:

1. **Copula sampling**: Generate U = (U1, U2) from the copula (Uniform(0,1) marginals, with desired dependence)
2. **Risk-neutral marginal transformation**: Z_i = Phi^{-1}(U_i) (inverse normal/CDF transform), then
   S_i(T) = S_i(0) * exp((r - q_i - 0.5*sigma_i^2)*T + sigma_i*sqrt(T)*Z_i)

The key insight: **the copula parameter rho controls dependence, not the marginals**. Changing rho alters the joint distribution of (Z1, Z2) while each Z_i remains standard normal marginally. The risk-neutral lognormal marginals are unchanged when rho varies.

In code (Python):
```python
# Step 1: copula sample
U = gaussian_copula_sample(rho, N, seed=seed)

# Step 2: transform to normal scores
Z1 = scipy_stats.norm.ppf(U[:, 0])  # = Phi^{-1}(U1)
Z2 = scipy_stats.norm.ppf(U[:, 1])  # = Phi^{-1}(U2)

# Step 3: apply risk-neutral GBM
S1_T = S1_0 * np.exp((r - q1 - 0.5 * sigma1**2) * T + sigma1 * np.sqrt(T) * Z1)
S2_T = S2_0 * np.exp((r - q2 - 0.5 * sigma2**2) * T + sigma2 * np.sqrt(T) * Z2)
```

## 7. Payoff calculation

### Best-of call

```python
def best_of_call(S1_T, S2_T, K):
    """H_T = max(max(S1_T, S2_T) - K, 0)"""
    return np.maximum(np.maximum(S1_T, S2_T) - K, 0.0)
```

### Worst-of put

```python
def worst_of_put(S1_T, S2_T, K):
    """H_T = max(K - min(S1_T, S2_T), 0)"""
    return np.maximum(K - np.minimum(S1_T, S2_T), 0.0)
```

### Pipeline flow

Terminal prices → payoff computation → discounted mean → standard error → confidence interval

Python:
```python
payoffs = best_of_call(S1_T, S2_T, K)
price = np.exp(-r * T) * np.mean(payoffs)
se = np.exp(-r * T) * sample_std(payoffs) / np.sqrt(N)
ci_low, ci_high = price_se_ci(price, se, confidence=0.95)
```

C++:
```cpp
auto payoffs = best_of_call(S1_T, S2_T, K);
auto result = price_from_payoffs(payoffs, r, T);
// result.price, result.se, result.ci_low, result.ci_high, result.num_scenarios
```

## 8. Statistical estimation

### Monte Carlo price estimator

$$\hat{P} = e^{-rT} \cdot \frac{1}{N} \sum_{i=1}^N \text{payoff}_i$$

### Standard error

$$\text{SE} = e^{-rT} \cdot \frac{s}{\sqrt{N}}$$

where s = sample standard deviation of the payoffs (ddof=1).

### 95% confidence interval

$$\hat{P} \pm 1.96 \cdot \text{SE}$$

(using normal approximation; exact coverage depends on payoff distribution)

### Convergence

The Monte Carlo error decreases as 1/sqrt(N). Doubling N reduces the SE by factor sqrt(2). To halve the SE, quadruple N.

### Black-Scholes agreement

The MC price should be within z*SE of the analytical BS price (default z = 3.0, plus small numerical tolerance 1e-10).

## 9. Benchmark execution

### Pipeline

1. Configure parameters (S0, K, T, r, q, sigma, rho, N, seed, model)
2. Generate random numbers (RNG with documented seed)
3. Sample copula (Gaussian or Student-t)
4. Transform to normal scores (Phi^{-1})
5. Apply risk-neutral GBM terminal formula
6. Compute payoffs (best-of call or worst-of put)
7. Discount and compute statistics (mean, SE, CI)
8. Record runtime, paths/second, and price estimate

### Data flow diagram

```mermaid
flowchart LR
    A[Configuration] --> B[RNG with seed]
    B --> C[Copula sampling]
    C --> D[Normal score transform]
    D --> E[GBM terminal prices]
    E --> F[Payoff computation]
    F --> G[Discounted statistics]
    G --> H[Benchmark results]
```

## 10. Data flow

The end-to-end data flow for a single Monte Carlo pricing run:

```
1. Set parameters: S1_0, S2_0, r, q1, q2, sigma1, sigma2, T, K, rho, nu, N, seed
2. Initialize RNG with seed → reproducible Z_i draws
3. Sample copula → U_ij ~ Uniform(0,1) with desired dependence
4. Z_i = Phi^{-1}(U_i) → correlated standard normals
5. S_i(T) = S_i(0) * exp((r - q_i - 0.5*sigma_i^2)*T + sigma_i*sqrt(T)*Z_i) → terminal prices
6. Payoff_i = payoff(S1_T[i], S2_T[i], K) → per-path payoff
7. Discounted payoff = exp(-r*T) * Payoff_i
8. price = mean(discounted payoffs)
9. SE = exp(-r*T) * std(discounted payoffs) / sqrt(N)
10. CI = price ± 1.96 * SE
```

The same flow applies to both Python and C++ implementations, with the C++ version using compiled loops and fixed memory allocations for step 3 and step 5.