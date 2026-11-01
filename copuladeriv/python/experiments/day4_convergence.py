"""Day 4: Monte Carlo convergence experiment.

Tests that the Monte Carlo estimator converges as the number of paths increases,
that SE ∝ 1/sqrt(N), and that confidence intervals have the correct coverage.
"""

import numpy as np
from scipy import stats as scipy_stats
from pathlib import Path

from copuladeriv import pricing
from copuladeriv.copulas import gaussian_copula_sample, student_t_copula_sample


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
S1_0 = 100.0
S2_0 = 100.0
K = 100.0
r = 0.04
q1 = 0.01
q2 = 0.015
sigma1 = 0.20
sigma2 = 0.25
T = 1.0

# Path counts to test
N_values = [10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]

# Correlation and nu configurations
rho_values = [-0.5, 0.0, 0.5]
nu_values = [4, 6, 10, 20]

seed = 42

# ---------------------------------------------------------------------------
# Helper: price best-of call given terminal prices
# ---------------------------------------------------------------------------
def price_best_of_call(S1_T, S2_T):
    from copuladeriv.payoffs import best_of_call
    payoffs_arr = best_of_call(S1_T, S2_T, K)
    price = np.exp(-r * T) * np.mean(payoffs_arr)
    se = np.exp(-r * T) * np.std(payoffs_arr) / np.sqrt(len(payoffs_arr))
    ci_low, ci_high = pricing.price_se_ci(price, se, confidence=0.95)
    return price, se, (ci_low, ci_high)


# ---------------------------------------------------------------------------
# Helper: price worst-of put given terminal prices
# ---------------------------------------------------------------------------
def price_worst_of_put(S1_T, S2_T):
    from copuladeriv.payoffs import worst_of_put
    payoffs_arr = worst_of_put(S1_T, S2_T, K)
    price = np.exp(-r * T) * np.mean(payoffs_arr)
    se = np.exp(-r * T) * np.std(payoffs_arr) / np.sqrt(len(payoffs_arr))
    ci_low, ci_high = pricing.price_se_ci(price, se, confidence=0.95)
    return price, se, (ci_low, ci_high)


# ---------------------------------------------------------------------------
# Simulate terminal prices under Gaussian copula
# ---------------------------------------------------------------------------
def simulate_gaussian_terminal(rho, N):
    U = gaussian_copula_sample(rho=rho, N=N, seed=seed)
    Z1 = scipy_stats.norm.ppf(U[:, 0])
    Z2 = scipy_stats.norm.ppf(U[:, 1])
    disc1 = (r - q1 - 0.5 * sigma1 ** 2) * T
    disc2 = (r - q2 - 0.5 * sigma2 ** 2) * T
    sqrtT = np.sqrt(T)
    S1_T = S1_0 * np.exp(disc1 + sigma1 * sqrtT * Z1)
    S2_T = S2_0 * np.exp(disc2 + sigma2 * sqrtT * Z2)
    return S1_T, S2_T


# ---------------------------------------------------------------------------
# Simulate terminal prices under Student-t copula
# ---------------------------------------------------------------------------
def simulate_student_t_terminal(rho, nu, N):
    U = student_t_copula_sample(rho=rho, nu=nu, N=N, seed=seed)
    Z1 = scipy_stats.norm.ppf(U[:, 0])
    Z2 = scipy_stats.norm.ppf(U[:, 1])
    disc1 = (r - q1 - 0.5 * sigma1 ** 2) * T
    disc2 = (r - q2 - 0.5 * sigma2 ** 2) * T
    sqrtT = np.sqrt(T)
    S1_T = S1_0 * np.exp(disc1 + sigma1 * sqrtT * Z1)
    S2_T = S2_0 * np.exp(disc2 + sigma2 * sqrtT * Z2)
    return S1_T, S2_T


# ---------------------------------------------------------------------------
# Run convergence experiment
# ---------------------------------------------------------------------------
results = []

print("=" * 80)
print("Day 4: Monte Carlo Convergence Experiment")
print("=" * 80)
print(f"Parameters: S1_0={S1_0}, S2_0={S2_0}, K={K}, r={r}, q1={q1}, q2={q2}")
print(f"sigma1={sigma1}, sigma2={sigma2}, T={T}, seed={seed}")
print(f"Path counts: {N_values}")
print()

# Storage for convergence analysis
all_results = []

for rho in rho_values:
    print(f"\n--- rho = {rho:.2f} ---")

    for nu in nu_values:
        copula_label = f"Student-t(nu={nu})" if nu is not None else "Gaussian"

        print(f"  {copula_label}:", end=" ")

        for N in N_values:
            # Simulate terminal prices
            if copula_label == "Gaussian":
                S1_T, S2_T = simulate_gaussian_terminal(rho, N)
            else:
                S1_T, S2_T = simulate_student_t_terminal(rho, nu, N)

            # Price best-of call
            call_price, call_se, call_ci = price_best_of_call(S1_T, S2_T)

            # Price worst-of put
            put_price, put_se, put_ci = price_worst_of_put(S1_T, S2_T)

            # Store result
            all_results.append({
                "copula": copula_label,
                "rho": rho,
                "nu": nu,
                "N": N,
                "call_price": call_price,
                "call_se": call_se,
                "call_ci_low": call_ci[0],
                "call_ci_high": call_ci[1],
                "put_price": put_price,
                "put_se": put_se,
                "put_ci_low": put_ci[0],
                "put_ci_high": put_ci[1],
            })

            # Print progress (only first and last N for readability)
            if N == N_values[0] or N == N_values[-1]:
                print(f"N={N:>7d} call={call_price:.4f}({call_se:.4f}) put={put_price:.4f}({put_se:.4f})", end="  ")
            elif N == N_values[len(N_values) // 2]:
                print(f"N={N:>7d} call={call_price:.4f}({call_se:.4f}) put={put_price:.4f}({put_se:.4f})", end="  ")

        print()

# ---------------------------------------------------------------------------
# Convergence analysis: SE ∝ 1/sqrt(N)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("Convergence Analysis: Standard Error vs. 1/sqrt(N)")
print("=" * 80)

# For best-of call with rho=0, Student-t nu=10
print("\nBest-of call convergence (rho=0, Student-t nu=10):")
print(f"{'N':>10}{'SE':>10}{'1/sqrt(N)':>12}{'SE * sqrt(N)':>14}")
se_values = []
isqrt_values = []
for r in all_results:
    if r["copula"] == "Student-t(nu=10)" and r["rho"] == 0.0:
        se_values.append(r["call_se"])
        isqrt_values.append(1.0 / np.sqrt(r["N"]))
        print(f"{r['N']:>10}{r['call_se']:>10.6f}{1.0/np.sqrt(r['N']):>12.6f}{r['call_se'] * np.sqrt(r['N']):>14.6f}")

# Compute the mean of SE * sqrt(N), which should be approximately constant
if se_values:
    mean_se_sqrt_n = np.mean([se * np.sqrt(n) for se, n in zip(se_values, [r["N"] for r in all_results if r["copula"] == "Student-t(nu=10)" and r["rho"] == 0.0])])
    print(f"\nMean SE * sqrt(N): {mean_se_sqrt_n:.6f} (should be approximately constant)")

print("\nPut convergence (rho=0, Student-t nu=10):")
print(f"{'N':>10}{'SE':>10}{'1/sqrt(N)':>12}{'SE * sqrt(N)':>14}")
se_put_values = []
for r in all_results:
    if r["copula"] == "Student-t(nu=10)" and r["rho"] == 0.0:
        se_put_values.append(r["put_se"])
print(f"{'':>10}{np.mean(se_put_values):>10.6f}")

# Check convergence slope via linear regression on log-log plot
print("\nLog-log regression (SE vs N):")
valid_se = []
valid_N = []
for r in all_results:
    if r["copula"] == "Gaussian" and r["rho"] == 0.0:
        valid_se.append(r["call_se"])
        valid_N.append(r["N"])

# Linear regression on log10
import numpy.polynomial.polynomial as P
if len(valid_N) > 1:
    log_N = np.log10(valid_N)
    log_SE = np.log10(valid_se)
    # Fit linear: log_SE = a + b * log_N
    coeffs = P.polyfit(log_N, log_SE, 1)
    slope = coeffs[1]
    print(f"  Fitted slope (should be ~ -0.5): {slope:.4f}")
    print(f"  Intercept: {coeffs[0]:.4f}")
    print(f"  R-squared: {1 - np.sum((log_SE - (coeffs[0] + slope * log_N))**2) / np.sum((log_SE - np.mean(log_SE))**2):.4f}")

# ---------------------------------------------------------------------------
# Save results CSV
# ---------------------------------------------------------------------------
results_dir = Path("copuladeriv/python/experiments/results")
results_dir.mkdir(parents=True, exist_ok=True)

csv_path = results_dir / "day4_convergence.csv"
with open(csv_path, "w") as f:
    f.write("copula,rho,nu,N,call_price,call_se,call_ci_low,call_ci_high,put_price,put_se,put_ci_low,put_ci_high\n")
    for r in all_results:
        f.write(f"{r['copula']},{r['rho']},{r['nu']},{r['N']},{r['call_price']:.8f},{r['call_se']:.8f},{r['call_ci_low']:.8f},{r['call_ci_high']:.8f},{r['put_price']:.8f},{r['put_se']:.8f},{r['put_ci_low']:.8f},{r['put_ci_high']:.8f}\n")

# ---------------------------------------------------------------------------
# Generate convergence plot
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

# Plot SE vs N on log-log scale for Gaussian rho=0
for copula_label, color, rho_val, nu_val in [
    ("Gaussian", "blue", 0.0, None),
    ("Student-t(nu=5)", "red", 0.0, 5),
    ("Student-t(nu=10)", "green", 0.0, 10),
    ("Student-t(nu=20)", "orange", 0.0, 20),
]:
    se_by_N = {}
    for r in all_results:
        if r["copula"] == copula_label and r["rho"] == rho_val:
            N = r["N"]
            if N not in se_by_N:
                se_by_N[N] = r["call_se"]
    Ns = sorted(se_by_N.keys())
    ses = [se_by_N[N] for N in Ns]
    plt.plot(Ns, ses, "o-", label=f"{copula_label} (rho={rho_val}, nu={nu_val})", color=color)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Number of paths N")
plt.ylabel("Standard Error (SE)")
plt.title("Monte Carlo Convergence: SE vs. Path Count")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.7)

plot_path = results_dir / "day4_convergence.png"
plt.savefig(plot_path)
plt.close()
print(f"\nConvergence plot saved to: {plot_path}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("Convergence Summary")
print("=" * 80)
print(f"Results CSV: {csv_path}")
print(f"Plot: {plot_path}")
print("\nKey check: SE * sqrt(N) should be approximately constant across N.")
print("If the convergence is correct (SE ∝ 1/sqrt(N)), then SE * sqrt(N)")
print("should not vary significantly with N.")
print("\nExperiment complete!")