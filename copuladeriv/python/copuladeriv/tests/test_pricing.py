"""Monte Carlo pricing tests for Day 1 CopulaDeriv project."""

import numpy as np
from copuladeriv import pricing


def test_monte_carlo_price_is_mean_discounted_payoff():
    """Price should equal the mean discounted payoff."""
    np.random.seed(42)
    result = pricing.monte_carlo_best_of_call_price(
        100, 100, 0.04, 0.01, 0.015, 0.2, 0.25, 1.0, K=100, N=1000, seed=42
    )
    # Manually verify
    S1_0, S2_0, r, q1, q2, sigma1, sigma2, T = 100, 100, 0.04, 0.01, 0.015, 0.2, 0.25, 1.0
    rng = np.random.default_rng(42)
    L = np.linalg.cholesky(np.array([[1.0, 0.0], [0.0, 1.0]]))
    Z = (L @ rng.standard_normal((2, 1000))).T
    S1_T = S1_0 * np.exp((r - q1 - 0.5 * sigma1**2) * T + sigma1 * np.sqrt(T) * Z[:, 0])
    S2_T = S2_0 * np.exp((r - q2 - 0.5 * sigma2**2) * T + sigma2 * np.sqrt(T) * Z[:, 1])
    from copuladeriv import payoffs
    payoffs_arr = payoffs.best_of_call(S1_T, S2_T, 100.0)
    discounted = np.exp(-r * T) * payoffs_arr
    manual_price = np.mean(discounted)
    assert np.isclose(result["price"], manual_price), (
        f"Result price {result['price']:.6f} != manual {manual_price:.6f}"
    )


def test_standard_error_format():
    """Standard error should be finite and positive."""
    result = pricing.monte_carlo_best_of_call_price(
        100, 100, 0.04, 0.01, 0.015, 0.2, 0.25, 1.0, K=100, N=1000, seed=42
    )
    assert np.isfinite(result["standard_error"]), "SE must be finite"
    assert result["standard_error"] > 0, "SE must be positive"


def test_confidence_interval_centered_on_estimate():
    """CI should be centered on the estimate."""
    result = pricing.monte_carlo_best_of_call_price(
        100, 100, 0.04, 0.01, 0.015, 0.2, 0.25, 1.0, K=100, N=1000, seed=42
    )
    price, se = result["price"], result["standard_error"]
    ci_low, ci_high = result["confidence_interval"]
    assert np.isclose(ci_low, price - 1.96 * se), f"CI low {ci_low} != {price - 1.96 * se:.6f}"
    assert np.isclose(ci_high, price + 1.96 * se), f"CI high {ci_high} != {price + 1.96 * se:.6f}"
    assert ci_low <= price <= ci_high, "CI must contain the estimate"


def test_identical_seeds_produce_identical_results():
    """Same seed should produce identical results."""
    result1 = pricing.monte_carlo_best_of_call_price(
        100, 100, 0.04, 0.01, 0.015, 0.2, 0.25, 1.0, 100, 1000, seed=42
    )
    result2 = pricing.monte_carlo_best_of_call_price(
        100, 100, 0.04, 0.01, 0.015, 0.2, 0.25, 1.0, 100, 1000, seed=42
    )
    assert result1["price"] == result2["price"], "Identical seeds must produce identical prices"
    assert result1["standard_error"] == result2["standard_error"]


def test_black_scholes_european_call():
    """Test Black-Scholes European call with continuous dividend yield."""
    bs = pricing.black_scholes_european_call(100, 100, 0.04, 0.01, 0.2, 1.0)
    assert np.isfinite(bs["price"]), "BS price must be finite"
    assert bs["price"] > 0, "BS price must be positive"
    assert np.isfinite(bs["delta"]), "BS delta must be finite"


def test_black_scholes_zero_volatility():
    """Test Black-Scholes with zero volatility."""
    bs = pricing.black_scholes_european_call(100, 100, 0.04, 0.01, 0.0, 1.0)
    # With zero vol, option is worth immediate exercise
    assert bs["price"] == 0.0, f"Zero vol price should be 0, got {bs['price']}"


def test_black_scholes_zero_maturity():
    """Test Black-Scholes with zero maturity."""
    bs = pricing.black_scholes_european_call(100, 100, 0.04, 0.01, 0.2, 0.0)
    # At maturity, option worth immediate payoff
    assert bs["price"] == 0.0, f"Zero maturity price should be 0, got {bs['price']}"


def test_black_scholes_out_of_the_money():
    """Test Black-Scholes with S0 below strike."""
    bs = pricing.black_scholes_european_call(80, 100, 0.04, 0.01, 0.2, 1.0)
    assert bs["price"] < 80, f"OTM price should be less than S0=80, got {bs["price"]}"


def test_black_scholes_in_the_money():
    """Test Black-Scholes with S0 above strike."""
    bs = pricing.black_scholes_european_call(120, 100, 0.04, 0.01, 0.2, 1.0)
    assert bs["price"] > 0, f"ITM price should be positive, got {bs['price']}"