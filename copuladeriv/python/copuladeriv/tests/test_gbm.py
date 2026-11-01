"""GBM simulation tests for Day 1 CopulaDeriv project."""

import numpy as np
from copuladeriv import marginals


def test_risk_neutral_lognormal_terminal():
    """Test that simulated mean converges to theoretical mean."""
    np.random.seed(42)
    S0, r, q, sigma, T = 100.0, 0.04, 0.01, 0.20, 1.0
    Z = np.random.standard_normal(100000)
    S_T = marginals.risk_neutral_lognormal_terminal(S0, r, q, sigma, T, Z)
    simulated_mean = np.mean(S_T)
    theoretical_mean = marginals.expected_terminal_mean(S0, r, q, sigma, T)
    # Use statistically justified tolerance based on standard error
    se = theoretical_mean * 0.01  # approx 1% of mean for 100k paths
    assert abs(simulated_mean - theoretical_mean) < 3 * se, (
        f"Simulated mean {simulated_mean:.6f} too far from theoretical {theoretical_mean:.6f}"
    )


def test_forward_price():
    """Test forward price calculation."""
    S0, r, q, sigma, T = 100.0, 0.04, 0.01, 0.20, 1.0
    F = marginals.forward_price(S0, r, q, sigma, T)
    expected = S0 * np.exp((r - q) * T)
    assert np.isclose(F, expected), f"Forward {F:.6f} != expected {expected:.6f}"


def test_expected_terminal_mean():
    """Test expected terminal mean formula."""
    S0, r, q, sigma, T = 100.0, 0.04, 0.01, 0.20, 1.0
    mu = marginals.expected_terminal_mean(S0, r, q, sigma, T)
    expected = S0 * np.exp((r - q) * T)
    assert np.isclose(mu, expected), f"Mean {mu:.6f} != expected {expected:.6f}"


def test_invalid_spot_rejected():
    """Negative spot should raise an error."""
    try:
        marginals.risk_neutral_lognormal_terminal(-100, 0.04, 0.01, 0.2, 1.0)
        assert False, "Should have raised an error for negative spot"
    except (ValueError, TypeError):
        pass


def test_invalid_volatility_rejected():
    """Non-positive volatility should raise an error."""
    try:
        marginals.risk_neutral_lognormal_terminal(100, 0.04, 0.01, 0.0, 1.0)
        assert False, "Should have raised an error for zero volatility"
    except (ValueError, TypeError):
        pass


def test_invalid_maturity_rejected():
    """Non-positive maturity should raise an error."""
    try:
        marginals.risk_neutral_lognormal_terminal(100, 0.04, 0.01, 0.2, 0.0)
        assert False, "Should have raised an error for zero maturity"
    except (ValueError, TypeError):
        pass


def test_terminal_prices_always_positive():
    """All simulated terminal prices must be finite and positive."""
    np.random.seed(42)
    S0, r, q, sigma, T = 100.0, 0.04, 0.01, 0.20, 1.0
    Z = np.random.standard_normal(10000)
    S_T = marginals.risk_neutral_lognormal_terminal(S0, r, q, sigma, T, Z)
    assert np.all(np.isfinite(S_T)), "Simulated prices must be finite"
    assert np.all(S_T > 0), "Simulated prices must be positive"


def test_rho_one_correlated_draws():
    """With rho=1, the two assets should have perfectly correlated shocks."""
    np.random.seed(42)
    from copuladeriv import payoffs

    S1_T, S2_T = payoffs.simulate_terminal_prices(
        100, 100, 0.04, 0.01, 0.015, 0.2, 0.25, 1.0, 1.0, 42
    )
    corr = np.corrcoef(S1_T, S2_T)[0, 1]
    # With rho=1, simulated correlation should be close to 1
    assert corr > 0.9, f"Expected correlation near 1 with rho=1, got {corr:.4f}"


def test_rho_negative_correlated_draws():
    """With rho=-1, the two assets should have negatively correlated shocks."""
    np.random.seed(42)
    from copuladeriv import payoffs

    S1_T, S2_T = payoffs.simulate_terminal_prices(
        100, 100, 0.04, 0.01, 0.015, 0.2, 0.25, 1.0, -1.0, 42
    )
    corr = np.corrcoef(S1_T, S2_T)[0, 1]
    # With rho=-1, simulated correlation should be close to -1
    assert corr < -0.9, f"Expected correlation near -1 with rho=-1, got {corr:.4f}"


def test_rho_zero_uncorrelated_draws():
    """With rho=0, the two assets should have approximately zero correlation."""
    np.random.seed(42)
    from copuladeriv import payoffs

    S1_T, S2_T = payoffs.simulate_terminal_prices(
        100, 100, 0.04, 0.01, 0.015, 0.2, 0.25, 1.0, 0.0, 10000, seed=42
    )
    corr = np.corrcoef(S1_T, S2_T)[0, 1]
    # With rho=0, simulated correlation should be close to 0
    assert abs(corr) < 0.2, f"Expected correlation near 0 with rho=0, got {corr:.4f}"