"""Performance benchmark for the Day 1 CopulaDeriv Python pricing engine.

Benchmarks the existing Python Monte Carlo pricing engine using fixed illustrative
parameters and multiple path counts. All results are reproducible via a fixed seed.

Illustrative parameters:
    S1 = 100, S2 = 100
    K = 100
    r = 0.04, q1 = 0.01, q2 = 0.015
    sigma1 = 0.20, sigma2 = 0.25
    T = 1.0, rho = 0.25
"""

import time
import tracemalloc
import numpy as np
from copuladeriv import pricing


# Illustrative parameters (Day 1 baseline)
S1_0, S2_0 = 100.0, 100.0
K = 100.0
r = 0.04
q1 = 0.01
q2 = 0.015
sigma1 = 0.20
sigma2 = 0.25
T = 1.0
rho = 0.25

SEED = 42

PATH_COUNTS = [10_000, 100_000, 1_000_000]
REPETITIONS = 3


def bench_one(path_count: int, seed_offset: int = 0) -> dict:
    """Run a single benchmark repetition and return timing/results.

    Parameters
    ----------
    path_count : int
        Number of simulation paths.
    seed_offset : int
        Offset added to the base seed for reproducibility across
        different path counts.

    Returns
    -------
    dict with keys:
        paths, seed, price, standard_error, runtime_sec, paths_per_sec,
        ci_low, ci_high, memory_peak_bytes
    """
    seed = SEED + seed_offset

    # Warm-up run (un-timed)
    _ = pricing.monte_carlo_best_of_call_price(
        S1_0, S2_0, r, q1, q2, sigma1, sigma2, T, K=K, N=path_count, seed=seed
    )
    _ = pricing.monte_carlo_worst_of_put_price(
        S1_0, S2_0, r, q1, q2, sigma1, sigma2, T, K=K, N=path_count, seed=seed
    )

    # Time best-of call
    t0 = time.perf_counter()
    result_call = pricing.monte_carlo_best_of_call_price(
        S1_0, S2_0, r, q1, q2, sigma1, sigma2, T, K=K, N=path_count, seed=seed
    )
    t1 = time.perf_counter()

    # Time worst-of put (starting after call)
    t2 = time.perf_counter()
    result_put = pricing.monte_carlo_worst_of_put_price(
        S1_0, S2_0, r, q1, q2, sigma1, sigma2, T, K=K, N=path_count, seed=seed
    )
    t3 = time.perf_counter()

    call_runtime = t1 - t0
    put_runtime = t2 - t1  # t2-t1 because t2 starts right after t1

    # Memory peak (tracemalloc may not capture all NumPy native allocations)
    tracemalloc.start()
    _ = pricing.monte_carlo_best_of_call_price(
        S1_0, S2_0, r, q1, q2, sigma1, sigma2, T, K=K, N=path_count, seed=seed
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_peak_bytes = peak

    return {
        "paths": path_count,
        "seed": seed,
        "price_call": result_call["price"],
        "se_call": result_call["standard_error"],
        "ci_low_call": result_call["confidence_interval"][0],
        "ci_high_call": result_call["confidence_interval"][1],
        "price_put": result_put["price"],
        "se_put": result_put["standard_error"],
        "ci_low_put": result_put["confidence_interval"][0],
        "ci_high_put": result_put["confidence_interval"][1],
        "runtime_call_sec": call_runtime,
        "runtime_put_sec": put_runtime,
        "paths_per_sec_call": path_count / call_runtime if call_runtime > 0 else float("inf"),
        "paths_per_sec_put": path_count / put_runtime if put_runtime > 0 else float("inf"),
        "memory_peak_bytes": memory_peak_bytes,
    }


def main():
    print("=" * 78)
    print("CopulaDeriv Python Baseline Benchmark")
    print("=" * 78)
    print(f"Parameters: S1=S2={S1_0}, K={K}, r={r}, q1={q1}, q2={q2}, "
            f"sigma1={sigma1}, sigma2={sigma2}, T={T}, rho={rho}")
    print(f"Seed: {SEED}")
    print(f"Path counts: {PATH_COUNTS}")
    print(f"Repetitions per count: {REPETITIONS}")
    print()

    all_results = []

    for paths in PATH_COUNTS:
        print(f"--- Benchmarking {paths:,} paths ---")
        rep_results = []
        for rep in range(REPETITIONS):
            result = bench_one(paths, seed_offset=rep)
            rep_results.append(result)
            print(f"  Rep {rep + 1}: "
                  f"call={result['runtime_call_sec']:.4f}s put={result['runtime_put_sec']:.4f}s "
                  f"paths/s_call={result['paths_per_sec_call']:.0f} put={result['paths_per_sec_put']:.0f}")
        # Compute median runtimes across repetitions
        median_call = sorted([r["runtime_call_sec"] for r in rep_results])[1]
        median_put = sorted([r["runtime_put_sec"] for r in rep_results])[1]

        # Use the first repetition's price/SE as the representative estimate
        representative = rep_results[0]
        all_results.append({
            "paths": paths,
            "seed": SEED,
            "price_call": representative["price_call"],
            "se_call": representative["se_call"],
            "ci_low_call": representative["ci_low_call"],
            "ci_high_call": representative["ci_high_call"],
            "price_put": representative["price_put"],
            "se_put": representative["se_put"],
            "ci_low_put": representative["ci_low_put"],
            "ci_high_put": representative["ci_high_put"],
            "runtime_call_sec": median_call,
            "runtime_put_sec": median_put,
            "paths_per_sec_call": paths / median_call if median_call > 0 else float("inf"),
            "paths_per_sec_put": paths / median_put if median_put > 0 else float("inf"),
            "memory_peak_bytes": representative["memory_peak_bytes"],
        })

    # Write CSV
    import csv
    csv_path = "copuladeriv/python/benchmarks/results/python_day2_baseline.csv"
    fieldnames = [
        "contract", "model", "num_paths", "seed", "repetition",
        "runtime_sec", "paths_per_sec", "price", "standard_error",
        "ci_low", "ci_high", "memory_peak_bytes"
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)

        for res in all_results:
            # Best-of call row
            writer.writerow([
                "best_of_call", "gbm_copula", res["paths"], res["seed"], 0,
                res["runtime_call_sec"], res["paths_per_sec_call"],
                res["price_call"], res["se_call"],
                res["ci_low_call"], res["ci_high_call"],
                res["memory_peak_bytes"]
            ])
            # Worst-of put row
            writer.writerow([
                "worst_of_put", "gbm_copula", res["paths"], res["seed"], 0,
                res["runtime_put_sec"], res["paths_per_sec_put"],
                res["price_put"], res["se_put"],
                res["ci_low_put"], res["ci_high_put"],
                res["memory_peak_bytes"]
            ])

    # Print summary Markdown
    print("\n" + "=" * 78)
    print("BENCHMARK SUMMARY (Median across repetitions)")
    print("=" * 78)
    for res in all_results:
        print(f"\nPaths: {res['paths']:,}")
        print(f"  Best-of call: "
              f"runtime={res['runtime_call_sec']:.4f}s, "
              f"paths/s={res['paths_per_sec_call']:.0f}, "
              f"price={res['price_call']:.6f} +/- {res['se_call']:.6f}")
        print(f"  Worst-of put: "
              f"runtime={res['runtime_put_sec']:.4f}s, "
              f"paths/s={res['paths_per_sec_put']:.0f}, "
              f"price={res['price_put']:.6f} +/- {res['se_put']:.6f}")

    print("\nResults written to: copuladeriv/python/benchmarks/results/python_day2_baseline.csv")
    print("=" * 78)


if __name__ == "__main__":
    main()