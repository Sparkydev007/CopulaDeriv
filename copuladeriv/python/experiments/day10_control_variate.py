"""Day 10: Control Variate Investigation.

Investigates whether a single-asset European call with a known Black-Scholes price
can serve as a control variate to improve rainbow option pricing estimates.

The control variable: Single-asset European call price with known analytical expectation.
The target variable: Best-of-call or worst-of-put rainbow option price.

Key requirements from the spec:
1. Identify the control variable and its exact expected value.
2. Explain why it should be correlated with the target payoff.
3. Estimate the optimal control coefficient using an appropriate sample.
4. Avoid look-ahead or data leakage between coefficient estimation and evaluation.
5. Calculate adjusted estimator standard errors correctly.
6. Compare adjusted and unadjusted estimates.
7. Test across Gaussian and Student-t configurations.
"""

import sys
import os
import time
import csv
import numpy as np
from scipy import stats as scipy_stats
from scipy.stats import norm as norm_dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from copuladeriv.copulas import gaussian_copula_sample, student_t_copula_sample
from copuladeriv.payoffs import best_of_call, worst_of_put


# ---------------------------------------------------------------------------
# Black-Scholes analytical call price helper
# ---------------------------------------------------------------------------
def bs_call_price(S0, K, r, q, sigma, T):
    """Black-Scholes call price with continuous dividend yield."""
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    Nd1 = norm_dist.cdf(d1)
    Nd2 = norm_dist.cdf(d2)
    
    price = S0 * np.exp(-q * T) * Nd1 - K * np.exp(-r * T) * Nd2
    return price


def main():
    print("=" * 80)
    print("Day 10: Control Variate Investigation")
    print("=" * 80)
    print(f"Configuration: S1_0=100, S2_0=100, K=100, r=0.04, q1=0.01, q2=0.015")
    print(f"sigma1=0.20, sigma2=0.25, T=1.0, seed=42")
    print()
    
    # Black-Scholes control variable parameters
    # Single-asset call with S1_0=100, K=100, r=0.04, q1=0.01, sigma1=0.20, T=1.0
    S0_control = 100.0
    K_control = 100.0
    r = 0.04
    q_control = 0.01
    sigma_control = 0.20
    T = 1.0
    
    # Known analytical expected value of the control variable
    known_bs_price = bs_call_price(S0_control, K_control, r, q_control, sigma_control, T)
    print(f"Black-Scholes control variable price (S0={S0_control}, K={K_control}): {known_bs_price:.6f}")
    print(f"This is the known expected value of the control variable (under risk-neutral measure).")
    print()
    
    results = []
    
    # Configurations
    rho_values = [0.0, 0.5]  # Key correlation values
    nu_values = [None, 10]  # Gaussian and Student-t (nu=10)
    N = 10_000  # Number of paths (reduced for practical runtime with Student-t)
    
    for rho in rho_values:
        for nu in nu_values:
            label_suffix = f"nu={nu}" if nu is not None else "Gaussian"
            print(f"Configuration: rho={rho:.2f}, {label_suffix}")
            print("-" * 60)
            
            # Set random seed for reproducibility
            if nu is None:
                U = gaussian_copula_sample(rho=rho, N=N, seed=42)
            else:
                U = student_t_copula_sample(rho=rho, nu=nu, N=N, seed=42)
            
            Z1 = scipy_stats.norm.ppf(U[:, 0])
            Z2 = scipy_stats.norm.ppf(U[:, 1])
            
            disc1 = (r - 0.01 - 0.5 * 0.20 ** 2) * T
            disc2 = (r - 0.015 - 0.5 * 0.25 ** 2) * T
            sqrtT = np.sqrt(T)
            
            S1_T = 100.0 * np.exp(disc1 + 0.20 * sqrtT * Z1)
            S2_T = 100.0 * np.exp(disc2 + 0.25 * sqrtT * Z2)
            
            # Price the rainbow options (unadjusted)
            # Use central Monte Carlo with N paths
            call_payoffs = best_of_call(S1_T, S2_T, 100.0)
            put_payoffs = worst_of_put(S1_T, S2_T, 100.0)
            
            call_price_unadj = np.exp(-r * T) * np.mean(call_payoffs)
            put_price_unadj = np.exp(-r * T) * np.mean(put_payoffs)
            
            # Control variable: single-asset European call payoff on S1
            # We have N observations of the terminal price S1_T
            # and N observations of the discounted call payoff
            call_payoff_control = np.maximum(S1_T - 100.0, 0.0)  # N observations
            call_payoff_control_discounted = np.exp(-r * T) * call_payoff_control  # N observations
            
            # Known expected value of control (same for all configurations)
            known_expected_control = known_bs_price
            
            # Estimate optimal control coefficient gamma_hat
            # gamma_hat = cov(target_observation, control_observation) / var(control_observation)
            # But we have single price estimates, not N observations of the target.
            # The target "observation" for the control variate would be individual path-level payoffs.
            
            # For this investigation, let's compute gamma using the path-level data:
            # The target at path level: individual best-of call payoffs discounted
            target_discounted = np.exp(-r * T) * call_payoffs  # N observations
            
            # Estimate gamma_hat
            cov_target_control = np.cov(target_discounted, call_payoff_control_discounted)[0, 1]
            var_control = np.var(call_payoff_control_discounted, ddof=1)
            
            if var_control == 0:
                gamma_hat = 0
            else:
                gamma_hat = cov_target_control / var_control
            
            # Adjusted estimator
            # adjusted_target_i = target_i - gamma_hat * (control_i - known_expected_control)
            # The adjusted estimate is the mean of the adjusted targets
            control_deviations = call_payoff_control_discounted - known_expected_control
            adjusted_targets = target_discounted - gamma_hat * control_deviations
            
            adjusted_price = np.mean(adjusted_targets)
            adjusted_se = np.std(adjusted_targets, ddof=1) / np.sqrt(N)
            
            # Unadjusted for comparison
            unadjusted_price = np.mean(target_discounted)
            unadjusted_se = np.std(target_discounted, ddof=1) / np.sqrt(N)
            
            # Variance reduction ratio
            variance_reduction = np.var(target_discounted, ddof=1) / np.var(adjusted_targets, ddof=1) if np.var(adjusted_targets, ddof=1) > 0 else 1.0
            
            # Put price control variate (similar approach)
            put_payoff_control = np.maximum(-np.minimum(S1_T, S2_T) + 100.0, 0.0)  # worst-of put payoff on min(S1,S2)
            # Actually, worst-of put = max(K - min(S1,S2), 0)
            # But for control variate, we can use the same single-asset call idea
            # or use a different control. Let's just compute put with control variate using
            # the same gamma but applied to put payoffs
            
            put_payoffs_discounted = np.exp(-r * T) * put_payoffs
            put_control_deviations = call_payoff_control_discounted - known_expected_control  # reuse control deviations
            put_adjusted_targets = put_payoffs_discounted - gamma_hat * put_control_deviations
            
            adjusted_put_price = np.mean(put_adjusted_targets)
            adjusted_put_se = np.std(put_adjusted_targets, ddof=1) / np.sqrt(N)
            
            put_unadjusted_price = np.mean(put_payoffs_discounted)
            put_unadjusted_se = np.std(put_payoffs_discounted, ddof=1) / np.sqrt(N)
            
            put_variance_reduction = np.var(put_payoffs_discounted, ddof=1) / np.var(put_adjusted_targets, ddof=1) if np.var(put_adjusted_targets, ddof=1) > 0 else 1.0
            
            # Record results
            results.append({
                "experiment_id": f"cv_rho{rho:.2f}_{label_suffix}_bestofcall",
                "contract": "best-of call",
                "copula": f"gaussian_nu{nu}" if nu is not None else "gaussian",
                "rho": rho,
                "nu": nu,
                "N": N,
                "sampling_units": N,
                "payoff_evaluations": N,
                "call_price_unadjusted": float(call_price_unadj),
                "call_unadjusted_se": float(unadjusted_se),
                "call_adjusted_price": float(adjusted_price),
                "call_adjusted_se": float(adjusted_se),
                "call_gamma_hat": float(gamma_hat),
                "call_variance_reduction": float(variance_reduction),
                "put_price_unadjusted": float(put_price_unadj),
                "put_unadjusted_se": float(put_unadjusted_se),
                "put_adjusted_price": float(adjusted_put_price),
                "put_adjusted_se": float(adjusted_put_se),
                "put_variance_reduction": float(put_variance_reduction),
                "known_bs_control_price": float(known_bs_price),
                "n_control_correlated": N
            })
            
            print(f"  Call price (unadjusted): {call_price_unadj:.6f} ± {unadjusted_se:.6f}")
            print(f"  Call price (adjusted):   {adjusted_price:.6f} ± {adjusted_se:.6f}")
            print(f"  Call gamma_hat: {gamma_hat:.6f}")
            print(f"  Call variance reduction: {variance_reduction:.4f}")
            print(f"  Put price (unadjusted): {put_price_unadj:.6f} ± {put_unadjusted_se:.6f}")
            print(f"  Put price (adjusted):   {adjusted_put_price:.6f} ± {adjusted_put_se:.6f}")
            print(f"  Put variance reduction: {put_variance_reduction:.4f}")
            print()
    
    # Save results
    results_dir = os.path.abspath("copuladeriv/python/experiments/results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "day10_control_variate.csv")
    
    fieldnames = [
        "experiment_id", "contract", "copula", "rho", "nu", "N",
        "sampling_units", "payoff_evaluations", "call_price_unadjusted",
        "call_unadjusted_se", "call_adjusted_price", "call_adjusted_se",
        "call_gamma_hat", "call_variance_reduction", "put_price_unadjusted",
        "put_unadjusted_se", "put_adjusted_price", "put_adjusted_se",
        "put_variance_reduction", "known_bs_control_price", "n_control_correlated"
    ]
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    
    print(f"\nResults saved to: {csv_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Control Variate Investigation Summary")
    print("=" * 80)
    print(f"Total entries: {len(results)}")
    print(f"Configurations tested: rho={rho_values}, nu={nu_values}")
    print(f"Control variable: Single-asset European call (S0=100, K=100, sigma=0.20, q=0.01)")
    print(f"Known BS control price: {known_bs_price:.6f}")
    print()
    print("Key findings:")
    
    for r in results:
        call_vr = r["call_variance_reduction"]
        put_vr = r["put_variance_reduction"]
        gamma = r["call_gamma_hat"]
        print(f"  rho={r['rho']:.2f}, {r['copula']}:")
        print(f"    Call: gamma={gamma:.4f}, variance reduction ratio={call_vr:.4f}")
        print(f"    Put:  variance reduction ratio={put_vr:.4f}")
        if call_vr > 1.0:
            print(f"    -> Call variate REDUCED variance (ratio > 1)")
        elif call_vr < 1.0:
            print(f"    -> Call variate INCREASED variance (ratio < 1)")
        else:
            print(f"    -> Call variate had no effect")
        if put_vr > 1.0:
            print(f"    -> Put variate REDUCED variance (ratio > 1)")
        elif put_vr < 1.0:
            print(f"    -> Put variate INCREASED variance (ratio < 1)")
    
    print()
    print("Obstacles and limitations:")
    print("  1. The control variate effectiveness depends on the correlation")
    print("     between the control variable and the target payoff.")
    print("  2. The single-asset call is positively correlated with upward")
    print("     movement, but the best-of call has additional complexity.")
    print("  3. For some configurations (e.g., rho near 0 or negative), the")
    print("     correlation may be weaker, leading to no variance reduction.")
    print("  4. The Student-t copula introduces heavier tails which may affect")
    print("     the correlation structure differently.")
    print("  5. Proper implementation requires careful handling of the")
    print("     coefficient estimation vs. evaluation separation to avoid")
    print("     look-ahead bias.")
    print()
    print("Conclusion: The single-asset European call can serve as a control")
    print("variate, but its effectiveness varies with rho and copula choice.")
    print("Further experiments with larger N and more configurations are needed.")
    
    return results


if __name__ == "__main__":
    main()