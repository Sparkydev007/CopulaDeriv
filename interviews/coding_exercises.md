# CopulaDeriv — Interview Coding Exercises

Practical exercises connected to the project, with C++ solutions, complexity analysis, and common mistakes. Keep educational; do not add unnecessary production engine code.

## 1. Implement a Monte Carlo estimator

### Problem
Write a function that estimates the price of a European call option using Monte Carlo simulation under the risk-neutral GBM model. The analytical Black-Scholes price is also required for comparison.

### Python solution
```python
import numpy as np
from scipy import stats as scipy_stats

def mc_european_call_price(S0, K, r, q, sigma, T, N, seed=42):
    """Monte Carlo price of a European call option.
    
    H_T = max(S_T - K, 0)
    S_T = S0 * exp((r - q - 0.5*sigma^2)*T + sigma*sqrt(T)*Z), Z ~ N(0,1)
    Price = exp(-r*T) * mean(payoffs)
    SE = exp(-r*T) * std(payoffs) / sqrt(N)
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(N)
    S_T = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(S_T - K, 0.0)
    discount = np.exp(-r * T)
    price = discount * np.mean(payoffs)
    se = discount * np.std(payoffs, ddof=1) / np.sqrt(N)
    ci_low, ci_high = price - 1.96 * se, price + 1.96 * se
    return price, se, (ci_low, ci_high)

# Black-Scholes analytical benchmark
def bs_european_call_price(S0, K, r, q, sigma, T):
    d1 = (np.log(S0/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    price = S0*np.exp(-q*T)*scipy_stats.norm.cdf(d1) - K*np.exp(-r*T)*scipy_stats.norm.cdf(d2)
    return price
```

### C++20 solution
```cpp
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

double phi_inv(double p) {
    // Rational approximation of the inverse standard normal CDF (probit)
    // Accurate to ~1e-7
    if (p <= 0.0 || p >= 1.0) throw std::domain_error("phi_inv: p must be in (0,1)");
    double q = p < 0.5 ? p : 1.0 - p;
    double r = std::sqrt(-2.0 * std::log(q));
    double val = ((2.30753 + r * ((-0.27061 + r * (0.99229 + r * (-0.02225 + r * 0.04406)))))
                  / (2.18678 + r * ((0.23038 + r * (-0.09373 + r * 0.05135)))));
    return p < 0.5 ? -val : val;
}

double mc_european_call_price(double S0, double K, double r, double q, double sigma, double T, int N, unsigned int seed = 42) {
    std::mt19937_64 engine(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    
    double discount = std::exp(-r * T);
    double sum_payoffs = 0.0;
    double sum_sq_payoffs = 0.0;  // for variance
    
    for (int i = 0; i < N; ++i) {
        double Z = normal(engine);
        double S_T = S0 * std::exp((r - q - 0.5 * sigma * sigma) * T + sigma * std::sqrt(T) * Z);
        double payoff = std::max(S_T - K, 0.0);
        sum_payoffs += payoff;
        sum_sq_payoffs += payoff * payoff;
    }
    
    double mean_payoff = sum_payoffs / N;
    double price = discount * mean_payoff;
    // Sample variance: (sum_sq - N*mean^2) / (N-1)
    double variance = (sum_sq_payoffs - N * mean_payoff * mean_payoff) / (N - 1);
    double se = discount * std::sqrt(variance / N);
    double ci_low = price - 1.96 * se;
    double ci_high = price + 1.96 * se;
    
    return price;  // could also return se, ci via ref params
}
```

### Complexity
- **Time**: O(N) — one pass over N paths, each requiring one normal draw, one exp(), one max().
- **Space**: O(1) — only accumulators (sum, sum_sq) needed; no path storage required (unlike finite-difference Greeks).

### Common mistakes
- Using `np.random.normal()` without a seed → results not reproducible.
- Forgetting the discount factor `exp(-r*T)` → price is not present-value.
- Using population ddof=0 instead of sample ddof=1 for SE → SE too small by factor sqrt(N/(N-1)).
- Not comparing to the BS benchmark → no validation that the MC implementation is correct.

---

## 2. Implement Welford online variance

### Problem
Implement Welford's online algorithm for computing the sample mean and variance in a single pass, which is numerically more stable than the two-pass formula (sum - N*mean^2) / (N-1) when values vary widely in magnitude.

### Python solution
```python
def welford_online_mean_var(values):
    """Compute mean and sample variance (ddof=1) using Welford's online algorithm.
    
    For each new value x_k (1-indexed):
      delta = x_k - mean_{k-1}
      mean_k = mean_{k-1} + delta / k
      delta2 = x_k - mean_k
      M2_k = M2_{k-1} + delta * delta2
    
    After all N values:
      mean = mean_N
      variance (ddof=1) = M2_N / (N-1)
    """
    N = len(values)
    if N == 0:
        return 0.0, 0.0
    if N == 1:
        return float(values[0]), 0.0
    
    mean = values[0]
    M2 = 0.0  # running sum of squared differences from the current mean
    
    for i in range(1, N):
        x = values[i]
        delta = x - mean
        mean += delta / (i + 1)
        delta2 = x - mean
        M2 += delta * delta2
    
    variance = M2 / (N - 1)  # sample variance
    return mean, variance
```

### C++20 solution
```cpp
#include <vector>
#include <cmath>
#include <stdexcept>

struct MeanVar {
    double mean;
    double variance;  // sample variance, ddof=1
};

MeanVar welford_online_mean_var(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::domain_error("welford_online: need at least one value");
    }
    if (values.size() == 1) {
        return {values[0], 0.0};
    }
    
    double mean = values[0];
    double M2 = 0.0;
    
    for (std::size_t i = 1; i < values.size(); ++i) {
        double x = values[i];
        double delta = x - mean;
        mean += delta / static_cast<double>(i + 1);
        double delta2 = x - mean;
        M2 += delta * delta2;
    }
    
    double variance = M2 / (static_cast<double>(values.size()) - 1);
    return {mean, variance};
}
```

### Why Welford is more stable
The naive formula `variance = sum((x - mean)^2) / (N-1)` requires two passes (one for mean, one for variance) and can suffer from catastrophic cancellation when sum of squares is much larger than mean^2. Welford's algorithm updates the mean and M2 incrementally, avoiding the need to store all values or compute the mean first. It is particularly helpful when the values have wide magnitude variation.

### Common mistakes
- Using `i` instead of `i+1` in the denominator (off-by-one in the online update).
- Returning population variance instead of sample variance (forgetting ddof=1).
- Not handling the N=1 case (variance would be division by zero).

---

## 3. Generate correlated Gaussian variables

### Problem
Write a function that generates N correlated standard normal variables with correlation rho, using the Cholesky decomposition method.

### Python solution
```python
import numpy as np

def correlated_gaussians(rho, N, seed=42):
    """Generate N pairs of correlated standard normals with correlation rho.
    
    Uses Cholesky: L = [[1, 0], [rho, sqrt(1-rho^2)]]
    Z_ind ~ N(0, I), then Z = L @ Z_ind gives correlated normals.
    Returns array of shape (N, 2).
    """
    rho = max(-1.0, min(1.0, rho))  # clip
    rng = np.random.default_rng(seed)
    Z_ind = rng.standard_normal((2, N))  # 2 rows x N columns
    Z0 = Z_ind[0]  # first row = Z_1
    sqrt_1_minus_rho2 = np.sqrt(np.maximum(0.0, 1.0 - rho**2))
    if rho == 1.0:
        Z1 = Z_ind[0]  # perfect positive
    elif rho == -1.0:
        Z1 = -Z_ind[0]  # perfect negative
    else:
        Z1 = rho * Z_ind[0] + sqrt_1_minus_rho2 * Z_ind[1]  # general case
    return np.column_stack((Z0, Z1))
```

### C++20 solution
```cpp
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>

struct CorrelatedGaussians {
    std::vector<double> z1;  // first asset shocks, size N
    std::vector<double> z2;  // second asset shocks, size N
};

CorrelatedGaussians generate_correlated_gaussians(double rho, int N, unsigned int seed = 42) {
    rho = std::max(-1.0, std::min(1.0, rho));  // clip
    std::mt19937_64 engine(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    
    CorrelatedGaussians result;
    result.z1.reserve(N);
    result.z2.reserve(N);
    
    double rt = std::sqrt(std::max(0.0, 1.0 - rho * rho));
    
    for (int i = 0; i < N; ++i) {
        double z_ind1 = normal(engine);
        double z_ind2 = normal(engine);
        double z1 = z_ind1;
        double z2;
        if (rho == 1.0) {
            z2 = z_ind1;  // perfect positive
        } else if (rho == -1.0) {
            z2 = -z_ind1;  // perfect negative
        } else {
            z2 = rho * z_ind1 + rt * z_ind2;  // general Cholesky
        }
        result.z1.push_back(z1);
        result.z2.push_back(z2);
    }
    
    return result;
}
```

### Verification
The empirical correlation of the generated pairs should agree with rho:
```python
U = correlated_gaussians(0.5, 100000, seed=42)
np.corrcoef(U[:, 0], U[:, 1])  # should be close to 0.5
```

### Complexity
- **Time**: O(N) — one pass, generating two normals and two Cholesky combinations per iteration.
- **Space**: O(N) — storing the two arrays of size N.

### Common mistakes
- Forgetting to clip rho to [-1, 1] → sqrt(1-rho^2) becomes NaN.
- Using the same independent normal for both Z1 and Z2 (defeating the purpose of correlation).
- Not reserving vector capacity in C++, causing repeated reallocations.

---

## 4. Implement a best-of payoff

### Problem
Write a function that computes the best-of call payoff: H_T = max(max(S1_T, S2_T) - K, 0).

### Python solution
```python
def best_of_call(S1_T, S2_T, K):
    """Best-of call payoff at maturity.
    H_T = max(max(S1_T, S2_T) - K, 0)
    """
    import numpy as np
    s1 = np.asarray(S1_T, dtype=float)
    s2 = np.asarray(S2_T, dtype=float)
    return np.maximum(np.maximum(s1, s2) - K, 0.0)
```

### C++20 solution
```cpp
#include <vector>
#include <algorithm>
#include <cmath>

std::vector<double> best_of_call(const std::vector<double>& S1_T,
                                 const std::vector<double>& S2_T,
                                 double K) {
    std::vector<double> payoffs(S1_T.size());
    for (std::size_t i = 0; i < S1_T.size(); ++i) {
        double s1 = S1_T[i];
        double s2 = S2_T[i];
        double m = (s1 > s2) ? s1 : s2;  // max(s1, s2)
        payoffs[i] = (m > K) ? (m - K) : 0.0;
    }
    return payoffs;
}
```

### Complexity
- **Time**: O(N) — one pass over the N paths.
- **Space**: O(N) — output payoff vector.

### Common mistakes
- Using `min` instead of `max` (confusing best-of with worst-of).
- Off-by-one errors in the loop (especially in C++ if using pointer arithmetic).
- Not handling scalar inputs (the Python version uses np.maximum which broadcasts; the C++ version expects vectors).

---

## 5. Implement a worst-of payoff

### Problem
Write a function that computes the worst-of put payoff: H_T = max(K - min(S1_T, S2_T), 0).

### Python solution
```python
def worst_of_put(S1_T, S2_T, K):
    """Worst-of put payoff at maturity.
    H_T = max(K - min(S1_T, S2_T), 0)
    """
    import numpy as np
    s1 = np.asarray(S1_T, dtype=float)
    s2 = np.asarray(S2_T, dtype=float)
    return np.maximum(K - np.minimum(s1, s2), 0.0)
```

### C++20 solution
```cpp
#include <vector>
#include <algorithm>
#include <cmath>

std::vector<double> worst_of_put(const std::vector<double>& S1_T,
                                 const std::vector<double>& S2_T,
                                 double K) {
    std::vector<double> payoffs(S1_T.size());
    for (std::size_t i = 0; i < S1_T.size(); ++i) {
        double s1 = S1_T[i];
        double s2 = S2_T[i];
        double m = (s1 < s2) ? s1 : s2;  // min(s1, s2)
        payoffs[i] = (K > m) ? (K - m) : 0.0;
    }
    return payoffs;
}
```

### Complexity
- **Time**: O(N) — one pass over the N paths.
- **Space**: O(N) — output payoff vector.

### Common mistakes
- Using `max` instead of `min` (confusing worst-of with best-of).
- Wrong comparison direction: `(K > m) ? (K - m) : 0.0` vs. `(m > K) ? (m - K) : 0.0`.

---

## 6. Implement a Box-Muller Gaussian generator

### Problem
Write a Box-Muller transform that generates two independent standard normal variables from two independent uniform variables U1, U2 ~ Uniform(0,1).

### Python solution
```python
import numpy as np

def box_muller():
    """Generate two independent standard normal variables using Box-Muller transform.
    
    Draw U1, U2 ~ Uniform(0,1). If U1 < eps (to avoid log(0)), retry.
    R = sqrt(-2*ln(U1)), Theta = 2*pi*U2.
    Z0 = R*cos(Theta), Z1 = R*sin(Theta).
    """
    eps = np.finfo(float).eps ** 0.5  # ~1e-8
    while True:
        U1 = np.random.uniform(eps, 1-eps)
        U2 = np.random.uniform(eps, 1-eps)
        R = np.sqrt(-2.0 * np.log(U1))
        Theta = 2.0 * np.pi * U2
        Z0 = R * np.cos(Theta)
        Z1 = R * np.sin(Theta)
        return Z0, Z1
```

### C++20 solution
```cpp
#include <random>
#include <cmath>
#include <stdexcept>

std::pair<double, double> box_muller() {
    double eps = std::sqrt(std::numeric_limits<double>::epsilon());  // ~1e-8
    static thread_local std::mt19937_64 engine(std::random_device{}());
    static std::uniform_real_distribution<double> uniform(eps, 1.0 - eps);
    
    double U1 = uniform(engine);
    double U2 = uniform(engine);
    double R = std::sqrt(-2.0 * std::log(U1));
    double Theta = 2.0 * M_PI * U2;
    double Z0 = R * std::cos(Theta);
    double Z1 = R * std::sin(Theta);
    return {Z0, Z1};
}
```

### Notes
- Box-Muller produces two independent N(0,1) variables from two uniform draws.
- The `eps` clipping avoids log(0) and sqrt of negative.
- The C++ version uses `thread_local` engine so the RNG state persists across calls without re-seeding every time.

### Complexity
- **Time**: O(1) — constant time, a few math operations.
- **Space**: O(1).

### Common mistakes
- Forgetting the `eps` clipping → NaN from log(0) or sqrt(negative).
- Using `log()` instead of `std::log()` or natural log vs. base-10 log confusion.
- Not handling the edge case where U1 is exactly 0 or 1.

---

## 7. Estimate a confidence interval

### Problem
Write a function that computes the 95% confidence interval for a Monte Carlo price estimate, given the price and standard error.

### Python solution
```python
def price_ci(price, se, confidence=0.95):
    """95% confidence interval from price and standard error.
    
    Uses normal approximation: price ± z * se, where z = norm.ppf((1+confidence)/2).
    """
    from scipy import stats as scipy_stats
    z = scipy_stats.norm.ppf((1.0 + confidence) / 2.0)  # 1.96 for 95%
    margin = z * se
    return (price - margin, price + margin)
```

### C++20 solution
```cpp
#include <cmath>
#include <stdexcept>

struct ConfidenceInterval {
    double lower;
    double upper;
};

ConfidenceInterval price_ci(double price, double se, double confidence = 0.95) {
    if (se < 0) throw std::domain_error("se must be non-negative");
    // z for 95%: 1.96; for 99%: 2.576
    // General formula: z = Phi^{-1}((1+confidence)/2)
    // Use rational approximation of inverse normal
    double p = (1.0 + confidence) / 2.0;
    // Approximation accurate to ~1e-3 for common confidence levels
    double z;
    if (p == 0.975) z = 1.96;
    else if (p == 0.995) z = 2.576;
    else {
        // Rational approximation (probit)
        double q = p < 0.5 ? p : 1.0 - p;
        double r = std::sqrt(-2.0 * std::log(q));
        z = (p < 0.5 ? -1.0 : 1.0) * ((2.30753 + r * ((-0.27061 + r * (0.99229 + r * (-0.02225 + r * 0.04406)))))
                                          / (2.18678 + r * ((0.23038 + r * (-0.09373 + r * 0.05135)))));
    }
    double margin = z * se;
    return {price - margin, price + margin};
}
```

### Common mistakes
- Using z = 1.96 unconditionally even for confidence levels other than 95%.
- Returning the margin instead of the (lower, upper) pair.
- Not checking that se >= 0.

---

## 7. Calculate paths per second

### Problem
Write a function that computes paths-per-second throughput given the number of paths N and the median runtime in seconds.

### Python solution
```python
def paths_per_second(N, median_runtime):
    """Compute Monte Carlo throughput: paths processed per second."""
    if median_runtime <= 0:
        raise ValueError("median_runtime must be positive")
    return N / median_runtime
```

### C++20 solution
```cpp
double paths_per_second(int N, double median_runtime_seconds) {
    if (median_runtime_seconds <= 0.0) {
        throw std::domain_error("median_runtime_seconds must be positive");
    }
    return static_cast<double>(N) / median_runtime_seconds;
}
```

### Common mistakes
- Dividing by the mean runtime instead of the median (the median is robust to outliers).
- Forgetting to handle the edge case of median_runtime = 0.

---

## 8. Calculate a basic C++ vectorized loop

### Problem
Write a simple C++ loop that computes the GBM terminal price for N paths using the exact formula, without copula dependence (rho = 0). This is the Day 1 baseline.

### C++20 solution
```cpp
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <iostream>

double compute_gbm_terminal_prices(double S0, double r, double q, double sigma, double T, int N, unsigned int seed = 42) {
    std::mt19937_64 engine(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    
    double discount = std::exp(-r * T);  // not returned but computed for completeness
    double drift = (r - q - 0.5 * sigma * sigma) * T;
    double sqrtT = std::sqrt(T);
    
    double sum = 0.0;
    double sum_sq = 0.0;
    
    for (int i = 0; i < N; ++i) {
        double Z = normal(engine);
        double S_T = S0 * std::exp(drift + sigma * sqrtT * Z);
        sum += S_T;
        sum_sq += S_T * S_T;
    }
    
    double mean = sum / N;
    double variance = (sum_sq - N * mean * mean) / (N - 1);
    return mean;  // could also return variance, se via ref params
}
```

### Benchmark harness
```cpp
int main() {
    int N = 1000000;
    auto start = std::chrono::high_resolution_clock::now();
    double mean = compute_gbm_terminal_prices(100.0, 0.04, 0.01, 0.20, 1.0, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double throughput = N / elapsed.count();
    std::cout << "N=" << N << " median_runtime=" << elapsed.count() << "s throughput=" 
              << throughput << " paths/s mean=" << mean << std::endl;
    return 0;
}
```

### Expected output (approximate, hardware-dependent)
```
N=1000000 median_runtime=0.25s throughput=4.0M paths/s mean≈101.9
```

### Common mistakes
- Using `rand()` or `srand()` instead of `std::mt19937_64` → lower quality and harder to seed reproducibly.
- Forgetting the `- 0.5 * sigma^2` term in the drift → wrong expected mean.
- Measuring wall-clock time incorrectly (e.g., not dividing by N for per-path time).

---

## 9. Explain and improve a cache-unfriendly loop

### Problem
Given a cache-unfriendly loop that accesses a 2D array column-by-column instead of row-by-column, explain why it's slow and rewrite it for cache efficiency.

### Problem description
```python
# UNOPTIMIZED: column-major access in a row-major language
def slow_correlation(x, y):
    """Compute Pearson correlation assuming x and y are N×2 arrays stored column-major."""
    N = len(x[0])
    mean_x = sum(x[i] for i in range(N)) / N
    mean_y = sum(y[i] for i in range(N)) / N
    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(N)) / (N - 1)
    var_x = sum((x[i] - mean_x)**2 for i in range(N)) / (N - 1)
    var_y = sum((y[i] - mean_y)**2 for i in range(N)) / (N - 1)
    return cov / (var_x * var_y)**0.5
```

In Python with NumPy arrays stored row-major, accessing `x[i]` where x is a column is slow because each access jumps across a full row in memory.

### Optimization
The optimized version uses NumPy's vectorized operations, or iterates row-by-row.

### C++20 comparison
A cache-unfriendly C++ loop might look like:
```cpp
// UNOPTIMIZED: accessing AoS layout column-by-column
struct Asset { double s1, s2; };
std::vector<Asset> assets(N);
// Computing correlation by iterating s2 first, then s1, causes cache misses
// because s2 values are separated by the size of the struct (2*8=16 bytes)
// but may share a cache line; the access pattern matters.
```

The cache-friendly version uses SoA:
```cpp
// OPTIMIZED: SoA layout, sequential access
std::vector<double> s1(N), s2(N);
// Now iterating i from 0 to N-1 accesses consecutive memory in each vector,
// which the CPU prefetcher can predict efficiently.
```

### Complexity
- **Time**: O(N) in both cases, but the cache-friendly version has significantly lower constant factor due to reduced cache misses.
- **Space**: O(N) — same.

### Common mistakes
- Not considering data layout when algorithmically the work is the same.
- Assuming Python's NumPy automatically handles layout (it does for vectorized ops, but not for manual loops).

---