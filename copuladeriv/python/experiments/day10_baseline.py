"""Day 10: Establish experimental baseline."""

import sys
import os
import time
import csv
import numpy as np
from scipy import stats as scipy_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from copuladeriv.copulas import gaussian_copula_sample, student_t_copula_sample
from copuladeriv.payoffs import best_of_call, worst_of_put


# ---------------------------------------------------------------------------
# Experiment configuration (illustrative parameters, not market-calibrated)
# ---------------------------------------------------------------------------
S1_0 = 100.0
S2_0 = 100.0
K = 100.0
risk_free = 0.04
div_yield1 = 0.01
div_yield2 = 0.015
sigma1 = 0.20
sigma2 = 0.25
T = 1.0

# Configurations - reduced for practical runtime
# Student-t is ~350x slower than Gaussian, so use smaller N and fewer nus
rho_values = [-0.75, -0.25, 0.0, 0.25, 0.75]
nu_values = [10]  # Only nu=10 to keep runtime reasonable; Gaussian covered separately
# Path counts adjusted: Gaussian can handle larger N, Student-t needs smaller N
path_counts = [10_000, 100_000, 500_000]  # Skip 1M for Student-t

seed = 42


def run_pricing(rho, nu, N, contract_type):
    """Run pricing for one configuration.
    
    contract_type: "call" for best-of call, "put" for worst-of put
    """
    start_time = time.time()
    
    if contract_type == "call":
        payoff_fn = best_of_call
    else:
        payoff_fn = worst_of_put
    
    if nu is None:
        # Gaussian copula
        U = gaussian_copula_sample(rho=rho, N=N, seed=seed)
    else:
        # Student-t copula
        U = student_t_copula_sample(rho=rho, nu=nu, N=N, seed=seed)
    
    Z1 = scipy_stats.norm.ppf(U[:, 0])
    Z2 = scipy_stats.norm.ppf(U[:, 1])
    
    disc1 = (risk_free - div_yield1 - 0.5 * sigma1 ** 2) * T
    disc2 = (risk_free - div_yield2 - 0.5 * sigma2 ** 2) * T
    sqrtT = np.sqrt(T)
    
    S1_T = S1_0 * np.exp(disc1 + sigma1 * sqrtT * Z1)
    S2_T = S2_0 * np.exp(disc2 + sigma2 * sqrtT * Z2)
    
    payoffs_arr = payoff_fn(S1_T, S2_T, K)
    discounted = np.exp(-risk_free * T) * payoffs_arr
    price = np.mean(discounted)
    se = np.exp(-risk_free * T) * np.std(discounted) / np.sqrt(N)
    ci_low = price - 1.96 * se
    ci_high = price + 1.96 * se
    runtime = time.time() - start_time
    
    return price, se, (ci_low, ci_high), runtime


def main():
    print("=" * 80)
    print("Day 10: Experimental Baseline (simplified)")
    print("=" * 80)
    print(f"Configuration: S1_0={S1_0}, S2_0={S2_0}, K={K}, "
          f"risk_free={risk_free}, div1={div_yield1}, div2={div_yield2}")
    print(f"sigma1={sigma1}, sigma2={sigma2}, T={T}, seed={seed}")
    print()
    
    results = []
    
    for N in path_counts:
        print(f"\n=== Path count: N = {N:,} ===")
        
        for rho in rho_values:
            print(f"  rho = {rho:.2f} (Gaussian)...", end=" ")
            
            # Best-of call
            price, se, (ci_low, ci_high), runtime = run_pricing(rho, None, N, "call")
            results.append({
                "experiment_id": f"baseline_gaussian_call_N{N}_rho{rho:.2f}",
                "contract": "best-of call",
                "copula": "gaussian",
                "rho": rho,
                "nu": None,
                "N": N,
                "sampling_units": N,
                "payoff_evaluations": N,
                "price": price,
                "standard_error": se,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "runtime": runtime,
                "paths_per_second": N / runtime if runtime > 0 else 0,
                "seed": seed,
                "implementation": "python",
            })
            
            # Worst-of put
            price, se, (ci_low, ci_high), runtime = run_pricing(rho, None, N, "put")
            results.append({
                "experiment_id": f"baseline_gaussian_put_N{N}_rho{rho:.2f}",
                "contract": "worst-of put",
                "copula": "gaussian",
                "rho": rho,
                "nu": None,
                "N": N,
                "sampling_units": N,
                "payoff_evaluations": N,
                "price": price,
                "standard_error": se,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "runtime": runtime,
                "paths_per_second": N / runtime if runtime > 0 else 0,
                "seed": seed,
                "implementation": "python",
            })
            print(f"done (call price={price:.4f})..")
            
        # Student-t copula (only nu=10, rho=0)
        for nu in nu_values:
            print(f"  rho = 0.0, nu = {nu} (Student-t)...", end=" ")
            
            # Best-of call
            price, se, (ci_low, ci_high), runtime = run_pricing(0.0, nu, N, "call")
            results.append({
                "experiment_id": f"baseline_studentt_call_N{N}_nu{nu}_rho0.0",
                "contract": "best-of call",
                "copula": "student-t",
                "rho": 0.0,
                "nu": nu,
                "N": N,
                "sampling_units": N,
                "payoff_evaluations": N,
                "price": price,
                "standard_error": se,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "runtime": runtime,
                "paths_per_second": N / runtime if runtime > 0 else 0,
                "seed": seed,
                "implementation": "python",
            })
            
            # Worst-of put
            price, se, (ci_low, ci_high), runtime = run_pricing(0.0, nu, N, "put")
            results.append({
                "experiment_id": f"baseline_studentt_put_N{N}_nu{nu}_rho0.0",
                "contract": "worst-of put",
                "copula": "student-t",
                "rho": 0.0,
                "nu": nu,
                "N": N,
                "sampling_units": N,
                "payoff_evaluations": N,
                "price": price,
                "standard_error": se,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "runtime": runtime,
                "paths_per_second": N / runtime if runtime > 0 else 0,
                "seed": seed,
                "implementation": "python",
            })
            print(f"done (call price={price:.4f})..")
    
    # Save results
    results_dir = os.path.abspath("copuladeriv/python/experiments/results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "day10_baseline.csv")
    csv_path = os.path.abspath(csv_path)
    
    fieldnames = [
        "experiment_id", "contract", "copula", "rho", "nu", "N",
        "sampling_units", "payoff_evaluations", "price", "standard_error",
        "ci_low", "ci_high", "runtime", "paths_per_second", "seed", "implementation"
    ]
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    
    print(f"\nResults saved to: {csv_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Baseline Summary")
    print("=" * 80)
    print(f"Total entries: {len(results)}")
    print(f"Path counts: {path_counts}")
    print(f"Rhos: {rho_values}")
    print(f"Student-t nus: {nu_values}")
    print(f"Contracts: best-of call, worst-of put")
    print(f"Copulas: Gaussian, Student-t (nu=10)")
    
    print("\nFirst 8 entries:")
    for r in results[:8]:
        print(f"  {r['experiment_id']}: price={r['price']:.6f}, SE={r['standard_error']:.6f}, N={r['N']:,}, copula={r['copula']}, rho={r['rho']}, nu={r['nu']}")
    
    # Convergence check
    print("\nConvergence check (SE * sqrt(N) approx constant):")
    for N in [10_000, 100_000]:
        call_entries = [r for r in results if r["N"] == N and r["contract"] == "best-of call" and r["copula"] == "gaussian"]
        if call_entries:
            se = call_entries[0]["standard_error"]
            ratio = se * np.sqrt(N)
            print(f"  N={N:,}: SE*sqrt(N)={ratio:.3f}")
    
    # Performance comparison
    print("\nPerformance comparison (paths/sec):")
    gaussian_entries = [r for r in results if r["copula"] == "gaussian" and r["contract"] == "best-of call"]
    student_t_entries = [r for r in results if r["copula"] == "student-t" and r["contract"] == "best-of call"]
    
    if gaussian_entries:
        gauss_rate = np.mean([r["paths_per_second"] for r in gaussian_entries[:3]])
        print(f"  Gaussian: ~{gauss_rate:.0f} paths/sec")
    
    if student_t_entries:
        student_rate = np.mean([r["paths_per_second"] for r in student_t_entries[:3]])
        print(f"  Student-t: ~{student_rate:.0f} paths/sec")
        if gaussian_entries and student_t_entries:
            ratio = gauss_rate / student_rate if student_rate > 0 else float('inf')
            print(f"  Ratio Gaussian/Student-t: ~{ratio:.0f}x")


if __name__ == "__main__":
    main()