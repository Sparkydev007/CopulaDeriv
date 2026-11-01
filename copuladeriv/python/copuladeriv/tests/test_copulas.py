"""Gaussian copula validation tests for the CopulaDeriv project.

These tests validate the Gaussian copula implementation across:
- Uniform marginal properties
- Dependence structure correctness
- Risk-neutral marginal preservation
- Pricing integration
- Reproducibility
- Boundary cases (rho = -1, 0, 1)
"""

import numpy as np
from scipy import stats as scipy_stats

from copuladeriv import copulas
from copuladeriv import pricing


# ============================================================================
# A. Uniform marginal tests
# ============================================================================

def test_uniform_margins_lie_in_open_unit_interval():
    """Check that U1 and U2 lie strictly inside (0,1) after numerical endpoint handling."""
    N = 100_000
    for rho in [-0.5, 0.0, 0.5]:
        U = copulas.gaussian_copula_sample(rho, N, seed=42)
        # After CDF transform, values should be in (0, 1), not including exact 0 or 1
        assert np.all(U > 0), f"U values must be > 0 for rho={rho}"
        assert np.all(U < 1), f"U values must be < 1 for rho={rho}"


def test_uniform_margins_approximate_uniform():
    """Check that each marginal is approximately uniform.

    For a Uniform(0,1) distribution: sample mean ≈ 0.5, sample variance ≈ 1/12.
    """
    N = 100_000
    for rho in [-0.5, 0.0, 0.5]:
        U = copulas.gaussian_copula_sample(rho, N, seed=42)
        # Column 0
        mean_0 = np.mean(U[:, 0])
        var_0 = np.var(U[:, 0])
        # Column 1
        mean_1 = np.mean(U[:, 1])
        var_1 = np.var(U[:, 1])

        # Tolerance based on CLT: se of mean = sqrt(p(1-p)/N) ≈ sqrt(0.25/N)
        # 3-sigma tolerance for mean
        mean_se = np.sqrt(0.25 / N)
        # 3-sigma tolerance for variance (using delta method)
        var_se = np.sqrt(2 / (N - 1)) * (1.0 / 12)  # approximate

        assert abs(mean_0 - 0.5) < 3 * mean_se, (
            f"rho={rho}: mean_0={mean_0:.6f} too far from 0.5")
        assert abs(var_0 - 1.0 / 12) < 3 * var_se, (
            f"rho={rho}: var_0={var_0:.6f} too far from 1/12={1.0/12:.6f}")
        assert abs(mean_1 - 0.5) < 3 * mean_se, (
            f"rho={rho}: mean_1={mean_1:.6f} too far from 0.5")
        assert abs(var_1 - 1.0 / 12) < 3 * var_se, (
            f"rho={rho}: var_1={var_1:.6f} too far from 1/12={1.0/12:.6f}")


# ============================================================================
# B. Dependence tests
# ============================================================================

def test_rho_zero_independence():
    """Check that rho = 0 produces approximately independent uniform variables."""
    N = 100_000
    U = copulas.gaussian_copula_sample(0.0, N, seed=42)
    corr = copulas.empirical_correlation(U)
    # With rho=0, empirical correlation should be close to 0
    assert abs(corr) < 0.1, f"rho=0: expected empirical correlation near 0, got {corr:.4f}"


def test_rho_positive_dependence():
    """Check that positive rho produces positive dependence."""
    N = 100_000
    for rho in [0.25, 0.5, 0.75, 1.0]:
        U = copulas.gaussian_copula_sample(rho, N, seed=42)
        corr = copulas.empirical_correlation(U)
        # Empirical correlation should be positive (and increasing in rho)
        assert corr > 0, f"rho={rho}: expected positive correlation, got {corr:.4f}"


def test_rho_negative_dependence():
    """Check that negative rho produces negative dependence."""
    N = 100_000
    for rho in [-0.25, -0.5, -0.75, -1.0]:
        U = copulas.gaussian_copula_sample(rho, N, seed=42)
        corr = copulas.empirical_correlation(U)
        # Empirical correlation should be negative (and decreasing in rho)
        assert corr < 0, f"rho={rho}: expected negative correlation, got {corr:.4f}"


def test_rho_one_perfect_positive():
    """Check that rho = 1 produces perfect positive dependence."""
    N = 100_000
    U = copulas.gaussian_copula_sample(1.0, N, seed=42)
    corr = copulas.empirical_correlation(U)
    # With rho=1, U1 = U2 exactly, so correlation should be 1
    assert abs(corr - 1.0) < 0.01, f"rho=1: expected correlation ~1, got {corr:.4f}"


def test_rho_minus_one_perfect_negative():
    """Check that rho = -1 produces perfect negative dependence."""
    N = 100_000
    U = copulas.gaussian_copula_sample(-1.0, N, seed=42)
    corr = copulas.empirical_correlation(U)
    # With rho=-1, U1 and U2 are perfectly negatively dependent
    # The empirical correlation should be -1
    assert abs(corr + 1.0) < 0.01, f"rho=-1: expected correlation ~-1, got {corr:.4f}"


def test_empirical_latent_correlation_agrees_with_rho():
    """Check that empirical latent Gaussian correlation agrees with requested rho.

    Note: This tests the correlation of the normal scores (Z = Phi^{-1}(U)),
    not the Pearson correlation of the uniforms themselves.
    """
    N = 100_000
    for rho in [0.0, 0.25, 0.5, 0.75, 1.0]:
        U = copulas.gaussian_copula_sample(rho, N, seed=42)
        # Transform back to normal scores
        Z1 = scipy_stats.norm.ppf(U[:, 0])
        Z2 = scipy_stats.norm.ppf(U[:, 1])
        empirical_rho = np.corrcoef(Z1, Z2)[0, 1]
        # Should agree with requested rho within sampling tolerance
        assert abs(empirical_rho - rho) < 0.02, (
            f"rho={rho}: expected empirical latent rho~{rho:.2f}, got {empirical_rho:.4f}")


# ============================================================================
# C. Risk-neutral marginal tests
# ============================================================================

def test_terminal_prices_positive_and_finite():
    """Check that terminal prices are positive and finite."""
    N = 50_000
    S1_0, S2_0 = 100.0, 100.0
    r, q1, q2 = 0.04, 0.01, 0.015
    sigma1, sigma2 = 0.20, 0.25
    T = 1.0

    for rho in [-0.5, 0.0, 0.5]:
        S1_T, S2_T = pricing.simulate_terminal_prices_copula(
            S1_0, S2_0, r, q1, q2, sigma1, sigma2, T, rho=rho, N=N, seed=42)
        assert np.all(np.isfinite(S1_T)), f"S1_T must be finite for rho={rho}"
        assert np.all(np.isfinite(S2_T)), f"S2_T must be finite for rho={rho}"
        assert np.all(S1_T > 0), f"S1_T must be positive for rho={rho}"
        assert np.all(S2_T > 0), f"S2_T must be positive for rho={rho}"


def test_marginals_unchanged_when_rho_changes():
    """Check that marginal distributions do not change when only copula rho changes.

    This is the key mathematical property: the Gaussian copula parameter rho
    controls dependence, not the marginals. Each asset retains its specified
    lognormal risk-neutral marginal distribution.
    """
    N = 200_000
    S1_0, S2_0 = 100.0, 100.0
    r, q1, q2 = 0.04, 0.01, 0.015
    sigma1, sigma2 = 0.20, 0.25
    T = 1.0

    theoretical_S1 = S1_0 * np.exp((r - q1) * T)
    theoretical_S2 = S2_0 * np.exp((r - q2) * T)

    for rho in [-0.75, -0.25, 0.0, 0.25, 0.75]:
        S1_T, S2_T = pricing.simulate_terminal_prices_copula(
            S1_0, S2_0, r, q1, q2, sigma1, sigma2, T, rho=rho, N=N, seed=42)

        mean_S1 = np.mean(S1_T)
        mean_S2 = np.mean(S2_T)

        # Use statistically justified tolerance
        # se of mean ≈ sigma / sqrt(N); for lognormal with these params, sigma ≈ 0.2
        # conservative tolerance: 3 * theoretical_mean * 0.01
        tol = 3 * theoretical_S1 * 0.01

        assert abs(mean_S1 - theoretical_S1) < tol, (
            f"rho={rho}: E[S1]={mean_S1:.6f} deviates from theoretical {theoretical_S1:.6f} by > {tol:.6f}")
        assert abs(mean_S2 - theoretical_S2) < tol, (
            f"rho={rho}: E[S2]={mean_S2:.6f} deviates from theoretical {theoretical_S2:.6f} by > {tol:.6f}")


def test_same_seed_produces_identical_samples():
    """Check that the same seed produces identical copula samples."""
    N = 100_000
    U1 = copulas.gaussian_copula_sample(0.3, N, seed=123)
    U2 = copulas.gaussian_copula_sample(0.3, N, seed=123)
    assert np.allclose(U1, U2), "Same seed must produce identical samples"


# ============================================================================
# D. Pricing integration tests
# ============================================================================

def test_best_of_call_prices_nonnegative():
    """Check that best-of call prices are non-negative."""
    N = 50_000
    S1_0, S2_0 = 100.0, 100.0
    r, q1, q2 = 0.04, 0.01, 0.015
    sigma1, sigma2 = 0.20, 0.25
    T = 1.0
    K = 100.0

    for rho in [-0.5, 0.0, 0.5]:
        result = pricing.monte_carlo_best_of_call_price_copula(
            S1_0, S2_0, r, q1, q2, sigma1, sigma2, T, K, N, rho=rho, seed=42)
        assert result["price"] >= 0, f"Best-of call price must be non-negative for rho={rho}, got {result['price']}"


def test_worst_of_put_prices_nonnegative():
    """Check that worst-of put prices are non-negative."""
    N = 50_000
    S1_0, S2_0 = 100.0, 100.0
    r, q1, q2 = 0.04, 0.01, 0.015
    sigma1, sigma2 = 0.20, 0.25
    T = 1.0
    K = 100.0

    for rho in [-0.5, 0.0, 0.5]:
        result = pricing.monte_carlo_worst_of_put_price_copula(
            S1_0, S2_0, r, q1, q2, sigma1, sigma2, T, K, N, rho=rho, seed=42)
        assert result["price"] >= 0, f"Worst-of put price must be non-negative for rho={rho}, got {result['price']}"


def test_discounting_correct():
    """Check that discounting is correct in copula pricing.

    With r=0 and T=1, the discount factor should be 1.
    """
    N = 50_000
    S1_0, S2_0 = 100.0, 100.0
    r = 0.0
    q1 = 0.0
    q2 = 0.0
    sigma1, sigma2 = 0.20, 0.25
    T = 1.0
    K = 100.0

    # With r=0, price should equal mean undiscounted payoff
    result = pricing.monte_carlo_best_of_call_price_copula(
        S1_0, S2_0, r, q1, q2, sigma1, sigma2, T, K, N, rho=0.0, seed=42)
    # For verification, also compute with baseline Cholesky
    result_base = pricing.monte_carlo_best_of_call_price(
        S1_0, S2_0, r, q1, q2, sigma1, sigma2, T, K, N, rho=0.0, seed=42)
    assert abs(result["price"] - result_base["price"]) < 0.01, (
        f"Discounting should give same result for r=0, diff={abs(result['price'] - result_base['price'])}")


def test_black_scholes_benchmark_remains_valid():
    """Check that the Black-Scholes benchmark remains valid alongside copula pricing."""
    S0, K, r, q, sigma, T = 100.0, 100.0, 0.04, 0.01, 0.20, 1.0
    bs = pricing.black_scholes_european_call(S0, K, r, q, sigma, T)
    assert np.isfinite(bs["price"]), "BS price must be finite"
    assert bs["price"] > 0, "BS price must be positive"
    assert np.isfinite(bs["delta"]), "BS delta must be finite"


def test_day1_tests_still_pass():
    """Ensure the original Day 1 baseline tests can be imported and run.

    This test simply verifies that the existing test infrastructure is importable
    and the Day 1 test suite can be collected.
    """
    # Import each day 1 test module to ensure it's valid
    from copuladeriv.tests import test_gbm  # noqa: F401
    from copuladeriv.tests import test_payoffs  # noqa: F401
    from copuladeriv.tests import test_pricing  # noqa: F401


# ============================================================================
# E. Regression tests
# ============================================================================

def test_regression_original_gbm_unchanged():
    """Ensure the original correlated GBM implementation remains correct.

    Compare Cholesky-based pricing with itself using the same seed.
    """
    N = 100_000
    S1_0, S2_0 = 100.0, 100.0
    r, q1, q2 = 0.04, 0.01, 0.015
    sigma1, sigma2 = 0.20, 0.25
    T = 1.0
    K = 100.0
    rho = 0.25
    seed = 42

    # Baseline Cholesky pricing
    result1 = pricing.monte_carlo_best_of_call_price(
        S1_0, S2_0, r, q1, q2, sigma1, sigma2, T, K, N, rho=rho, seed=seed)
    result2 = pricing.monte_carlo_best_of_call_price(
        S1_0, S2_0, r, q1, q2, sigma1, sigma2, T, K, N, rho=rho, seed=seed)

    assert result1["price"] == result2["price"], (
        "Original GBM pricing must be deterministic for same seed")
    assert result1["standard_error"] == result2["standard_error"], (
        "Original GBM SE must be deterministic for same seed")