"""Option payoff functions: reusable, independent of pricing engine.

Implements the two rainbow option contracts specified in the master prompt:

- Best-of call: H_T = max(max(S1_T, S2_T) - K, 0)
- Worst-of put: H_T = max(K - min(S1_T, S2_T), 0)
- Single-asset European call: H_T = max(S_T - K, 0) (validation reference)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Best-of call payoff
# ---------------------------------------------------------------------------

def best_of_call(S1_T: np.ndarray | float, S2_T: np.ndarray | float,
                 K: float) -> np.ndarray | float:
    """Best-of call payoff at maturity.

    H_T = max(max(S1_T, S2_T) - K, 0)

    Parameters
    ----------
    S1_T, S2_T : np.ndarray or float
        Terminal asset prices (can be scalar or array of shape (N,)).
    K : float
        Strike price.

    Returns
    -------
    payoff : np.ndarray or float
        Payoff value(s) for each simulation path.
    """
    s1 = np.asarray(S1_T, dtype=float)
    s2 = np.asarray(S2_T, dtype=float)
    return np.maximum(np.maximum(s1, s2) - K, 0.0)


# ---------------------------------------------------------------------------
# Worst-of put payoff
# ---------------------------------------------------------------------------

def worst_of_put(S1_T: np.ndarray | float, S2_T: np.ndarray | float,
                 K: float) -> np.ndarray | float:
    """Worst-of put payoff at maturity.

    H_T = max(K - min(S1_T, S2_T), 0)

    Parameters
    ----------
    S1_T, S2_T : np.ndarray or float
        Terminal asset prices (can be scalar or array of shape (N,)).
    K : float
        Strike price.

    Returns
    -------
    payoff : np.ndarray or float
        Payoff value(s) for each simulation path.
    """
    s1 = np.asarray(S1_T, dtype=float)
    s2 = np.asarray(S2_T, dtype=float)
    return np.maximum(K - np.minimum(s1, s2), 0.0)


# ---------------------------------------------------------------------------
# Single-asset European call payoff (validation reference)
# ---------------------------------------------------------------------------

def european_call(S_T: np.ndarray | float, K: float) -> np.ndarray | float:
    """Single-asset European call payoff at maturity.

    H_T = max(S_T - K, 0)

    Parameters
    ----------
    S_T : np.ndarray or float
        Terminal asset price (can be scalar or array of shape (N,)).
    K : float
        Strike price.

    Returns
    -------
    payoff : np.ndarray or float
        Payoff value(s) for each simulation path.
    """
    s = np.asarray(S_T, dtype=float)
    return np.maximum(s - K, 0.0)


# ---------------------------------------------------------------------------
# Terminal price simulation under risk-neutral correlated GBM
# ---------------------------------------------------------------------------

def simulate_terminal_prices(S1_0: float, S2_0: float, r: float, q1: float,
                             q2: float, sigma1: float, sigma2: float,
                             T: float, rho: float, N: int,
                             seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Simulate terminal prices for two assets under risk-neutral correlated GBM.

    Uses the exact terminal GBM formula (no time stepping):
        S_i(T) = S_i(0) * exp((r - q_i - 0.5*sigma_i^2)*T + sigma_i*sqrt(T)*Z_i)
    where Z_1, Z_2 are standard normal with correlation rho.

    The correlated normals are generated via Cholesky decomposition:
        L = [[1, 0], [rho, sqrt(1-rho^2)]]
        [Z1, Z2]^T = L @ [Z_indep_1, Z_indep_2]^T

    Parameters
    ----------
    S1_0, S2_0 : float
        Initial asset prices.
    r : float
        Continuously compounded risk-free rate (annualised).
    q1, q2 : float
        Continuous dividend yields (annualised).
    sigma1, sigma2 : float
        Annualised volatilities.
    T : float
        Time to maturity in years.
    rho : float
        Correlation between the two Brownian motions in [-1, 1].
    N : int
        Number of simulation paths.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    S1_T, S2_T : np.ndarray, shape (N,)
        Terminal asset prices.
    """
    rho = np.clip(rho, -1.0, 1.0)
    rng = np.random.default_rng(seed)

    # Generate independent standard normals
    Z_indep = rng.standard_normal((2, N))  # shape (2, N)

    # Apply Cholesky transformation
    # L = [[1, 0], [rho, sqrt(1-rho^2)]]
    # Z0 = independent normal -> becomes Z_1
    # Z1 = rho * Z0 + sqrt(1-rho^2) * independent normal -> becomes Z_2
    sqrt_one_minus_rho2 = np.sqrt(np.maximum(0.0, 1.0 - rho * rho))

    Z0 = Z_indep[0]  # independent standard normal
    Z1 = rho * Z0 + sqrt_one_minus_rho2 * Z_indep[1]  # correlated

    # Risk-neutral terminal prices
    disc1 = (r - q1 - 0.5 * sigma1 ** 2) * T
    disc2 = (r - q2 - 0.5 * sigma2 ** 2) * T
    sqrtT = np.sqrt(T)

    S1_T = S1_0 * np.exp(disc1 + sigma1 * sqrtT * Z0)
    S2_T = S2_0 * np.exp(disc2 + sigma2 * sqrtT * Z1)

    return S1_T, S2_T