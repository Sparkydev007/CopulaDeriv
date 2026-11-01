"""Statistical utilities for Monte Carlo pricing."""

import numpy as np
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Confidence interval helpers
# ---------------------------------------------------------------------------

def price_se_ci(price: float, se: float, confidence: float = 0.95) -> tuple[float, float]:
    """Compute confidence interval from price and standard error.

    Uses normal approximation: V_hat ± z_{1-alpha/2} * SE

    Parameters
    ----------
    price : float
        Estimated option price.
    se : float
        Monte Carlo standard error.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    tuple[float, float]
        (lower_bound, upper_bound) of the confidence interval.
    """
    z = scipy_stats.norm.ppf((1.0 + confidence) / 2.0)
    margin = z * se
    return (price - margin, price + margin)


# ---------------------------------------------------------------------------
# Sample statistics
# ---------------------------------------------------------------------------

def sample_mean(values: np.ndarray) -> float:
    """Sample mean."""
    return float(np.mean(values))


def sample_std(values: np.ndarray) -> float:
    """Sample standard deviation (ddof=1)."""
    return float(np.std(values, ddof=1))


# ---------------------------------------------------------------------------
# Convergence utilities
# ---------------------------------------------------------------------------

def price_convergence(prices_by_N: dict[int, float],
                      se_by_N: dict[int, float]) -> dict:
    """Summarise price convergence across different scenario counts.

    Parameters
    ----------
    prices_by_N : dict[int, float]
        Estimated price for each N.
    se_by_N : dict[int, float]
        Standard error for each N.

    Returns
    -------
    dict with convergence summary.
    """
    N_values = sorted(prices_by_N.keys())
    prices = [prices_by_N[n] for n in N_values]
    ses = [se_by_N[n] for n in N_values]

    return {
        "N_values": N_values,
        "prices": prices,
        "standard_errors": ses,
        "price_range": (min(prices), max(prices)),
        "se_range": (min(ses), max(ses)),
    }


# ---------------------------------------------------------------------------
# Black-Scholes agreement check
# ---------------------------------------------------------------------------

def bs_agreement_check(mc_price: float, bs_price: float,
                       mc_se: float,
                       tolerance_multiplier: float = 3.0) -> dict:
    """Check MC price against analytical Black-Scholes benchmark.

    Uses a statistical criterion: |MC - BS| <= z * SE + small_numerical_tol

    Parameters
    ----------
    mc_price : float
        Monte Carlo price estimate.
    bs_price : float
        Analytical Black-Scholes price.
    mc_se : float
        Monte Carlo standard error.
    tolerance_multiplier : float, default 3.0
        Number of SEs to use as tolerance.

    Returns
    -------
    dict with agreement results.
    """
    diff = abs(mc_price - bs_price)
    # Tolerance = z * SE + small numerical tolerance
    z = scipy_stats.norm.ppf((1.0 + 0.95) / 2.0)  # 1.96 for 95%
    numerical_tol = 1e-10
    tolerance = z * mc_se + numerical_tol

    within_tolerance = diff <= tolerance
    excess = diff - tolerance if not within_tolerance else 0.0

    return {
        "mc_price": mc_price,
        "bs_price": bs_price,
        "difference": diff,
        "standard_error": mc_se,
        "tolerance": tolerance,
        "within_tolerance": within_tolerance,
        "excess_over_tolerance": max(0.0, excess),
        "passes": within_tolerance,
    }