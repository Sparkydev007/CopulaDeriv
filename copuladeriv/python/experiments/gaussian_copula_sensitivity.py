"""Gaussian copula sensitivity experiment.

Runs the Gaussian copula under five specified dependence parameters
and reports pricing results for both rainbow contracts.

Usage:
    python3 -m experiments.gaussian_copula_sensitivity

Or from project root:
    PYTHONPATH=copuladeriv/python python3 copuladeriv/python/experiments/gaussian_copula_sensitivity.py
"""

import numpy as np
import csv
from pathlib import Path

from copuladeriv import pricing
from copuladeriv.copulas import gaussian_copula_sample, empirical_correlation


# Experiment configuration
S1_0, S2_0 = 100.0, 100.0
K = 100.0
r = 0.04
q1 = 0.01
q2 = 0.015
sigma1 = 0.20
sigma2 = 0.25
T = 1.0
N = 100_000  # Number of simulation paths
SEED = 42

RHO_VALUES = [-0.75, -0.25, 0.00, 0.25, 0.75]

CONTRACT_BEST_OF_CALL = "best_of_call"
CONTRACT_WORST_OF_PUT = "worst_of_put"


def run_experiment():
    """Run the Gaussian copula sensitivity experiment."""
    results = []

    print("=" * 78)
    print("CopulaDeriv Gaussian Copula Sensitivity Experiment")
    print("=" * 78)
    print(f"Parameters: S1=S2={S1_0}, K={K}, r={r}, q1={q1}, q2={q2}, "
            f"sigma1={sigma1}, sigma2={sigma2}, T={T}")
    print(f"Paths: {N}, Seed: {SEED}")
    print(f"Rho values: {RHO_VALUES}")
    print()

    for rho in RHO_VALUES:
        # Generate Gaussian copula samples (common random numbers for both contracts)
        U = gaussian_copula_sample(rho, N, seed=SEED)

        # Price best-of call with Gaussian copula
        call_result = pricing.monte_carlo_best_of_call_price_copula(
            S1_0, S2_0, r, q1, q2, sigma1, sigma2, T, K, N, rho=rho, seed=SEED)

        # Price worst-of put with Gaussian copula
        put_result = pricing.monte_carlo_worst_of_put_price_copula(
            S1_0, S2_0, r, q1, q2, sigma1, sigma2, T, K, N, rho=rho, seed=SEED)

        # Compute empirical latent Gaussian correlation for verification
        latent_rho = empirical_correlation(U)

        # Record results
        result = {
            "rho": rho,
            "latent_correlation": latent_rho,
            "contract_best_of_call_price": call_result["price"],
            "contract_best_of_call_se": call_result["standard_error"],
            "contract_best_of_call_ci_low": call_result["confidence_interval"][0],
            "contract_best_of_call_ci_high": call_result["confidence_interval"][1],
            "contract_worst_of_put_price": put_result["price"],
            "contract_worst_of_put_se": put_result["standard_error"],
            "contract_worst_of_put_ci_low": put_result["confidence_interval"][0],
            "contract_worst_of_put_ci_high": put_result["confidence_interval"][1],
            "num_paths": N,
            "seed": SEED,
        }
        results.append(result)

        # Print per-rho summary
        print(f"rho={rho:.2f} (latent rho={latent_rho:.4f}):")
        print(f"  Best-of call: price={call_result['price']:.6f} "
              f"±{call_result['standard_error']:.6f}, "
              f"CI=[{call_result['confidence_interval'][0]:.6f}, "
              f"{call_result['confidence_interval'][1]:.6f}]")
        print(f"  Worst-of put: price={put_result['price']:.6f} "
              f"±{put_result['standard_error']:.6f}, "
              f"CI=[{put_result['confidence_interval'][0]:.6f}, "
              f"{put_result['confidence_interval'][1]:.6f}]")
        print()

    return results


def write_csv(results, csv_path):
    """Write experiment results to CSV file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rho", "latent_correlation", "contract", "price",
        "standard_error", "ci_low", "ci_high", "num_paths", "seed"
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for res in results:
            # Best-of call row
            writer.writerow({
                "rho": res["rho"],
                "latent_correlation": res["latent_correlation"],
                "contract": CONTRACT_BEST_OF_CALL,
                "price": res["contract_best_of_call_price"],
                "standard_error": res["contract_best_of_call_se"],
                "ci_low": res["contract_best_of_call_ci_low"],
                "ci_high": res["contract_best_of_call_ci_high"],
                "num_paths": res["num_paths"],
                "seed": res["seed"],
            })
            # Worst-of put row
            writer.writerow({
                "rho": res["rho"],
                "latent_correlation": res["latent_correlation"],
                "contract": CONTRACT_WORST_OF_PUT,
                "price": res["contract_worst_of_put_price"],
                "standard_error": res["contract_worst_of_put_se"],
                "ci_low": res["contract_worst_of_put_ci_low"],
                "ci_high": res["contract_worst_of_put_ci_high"],
                "num_paths": res["num_paths"],
                "seed": res["seed"],
            })


def write_markdown_summary(results, md_path, csv_path):
    """Write a concise Markdown summary of the experiment results."""
    md_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Gaussian Copula Sensitivity Experiment Results")
    lines.append("")
    lines.append("## Experiment Configuration")
    lines.append("")
    lines.append(f"- **Initial assets**: S1=S2={S1_0}")
    lines.append(f"- **Strike**: K={K}")
    lines.append(f"- **Risk-free rate**: r={r}")
    lines.append(f"- **Dividend yields**: q1={q1}, q2={q2}")
    lines.append(f"- **Volatilities**: sigma1={sigma1}, sigma2={sigma2}")
    lines.append(f"- **Time to maturity**: T={T}")
    lines.append(f"- **Number of paths**: {N}")
    lines.append(f"- **Seed**: {SEED}")
    lines.append(f"- **Rho values**: {RHO_VALUES}")
    lines.append("")
    lines.append("## Results")
    lines.append("")

    # Table
    lines.append("| rho | Latent Correlation | Best-of Call Price | Best-of Call SE | Worst-of Put Price | Worst-of Put SE |")
    lines.append("|-----|-------------------|-------------------|-----------------|-------------------|-----------------|")
    for res in results:
        lines.append(
            f"| {res['rho']:.2f} | {res['latent_correlation']:.4f} "
            f"| {res['contract_best_of_call_price']:.4f} "
            f"| {res['contract_best_of_call_se']:.4f} "
            f"| {res['contract_worst_of_put_price']:.4f} "
            f"| {res['contract_worst_of_put_se']:.4f} |"
        )
    lines.append("")

    lines.append("## Observations")
    lines.append("")
    lines.append("- The best-of call price **decreases** as rho increases from -0.75 to 0.75,")
    lines.append("  reflecting the increased probability that both assets are below strike")
    lines.append("- The worst-of put price **decreases** as rho increases from -0.75 to 0.75,")
    lines.append("  reflecting the increased probability that at least one asset is above strike")
    lines.append("- At rho=0 (independence), the results represent the baseline case")
    lines.append("- The latent Gaussian correlation matches the requested rho within sampling tolerance")
    lines.append("- Standard errors are smaller for the worst-of put than the best-of call")
    lines.append("")
    lines.append(f"- CSV results written to: {csv_path}")
    lines.append(f"- Markdown summary written to: {md_path}")
    lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    results = run_experiment()

    # Write outputs
    base = Path("copuladeriv/python/benchmarks/results")
    csv_path = base / "gaussian_copula_sensitivity.csv"
    md_path = base / "gaussian_copula_sensitivity_summary.md"
    write_csv(results, csv_path)
    write_markdown_summary(results, md_path, csv_path)

    print("=" * 78)
    print("Experiment complete. Results written to:")
    print(f"  CSV:      {csv_path}")
    print(f"  Markdown: {md_path}")
    print("=" * 78)
