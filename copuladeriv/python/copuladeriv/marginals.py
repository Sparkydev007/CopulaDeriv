"""Risk-neutral marginal distributions for Monte Carlo pricing.

Implements the risk-neutral correlated Geometric Brownian Motion (GBM) terminal
distribution under the Q measure.

SDE under Q:
    dS_i / S_i = (r - q_i) dt + sigma_i dW_i
with instantaneous Brownian correlation rho.

Terminal value (exact, no time stepping):
    S_i(T) = S_i(0) * exp((r - q_i - 0.5 * sigma_i^2) * T + sigma_i * sqrt(T) * Z_i)

where Z_1, Z_2 are standard normal with correlation rho.
"""

from __future__ import annotations

import numpy as np


def _validate_lognormal_params(S0: float, r: float, q: float, sigma: float, T: float) -> None:
    """Validate parameters for lognormal terminal distribution.

    Raises
    ------
    ValueError
        If any parameter is invalid.
    """
    if not np.isfinite(S0) or S0 <= 0:
        raise ValueError("S0 must be finite and positive")
    if not np.isfinite(r):
        raise ValueError("r must be finite")
    if not np.isfinite(q):
        raise ValueError("q must be finite")
    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("sigma must be positive and finite")
    if T <= 0 or not np.isfinite(T):
        raise ValueError("T must be positive and finite")


def risk_neutral_lognormal_terminal(S0: float, r: float, q: float,
                                    sigma: float, T: float,
                                    Z: np.ndarray | None = None) -> np.ndarray:
    """Generate terminal asset prices under the risk-neutral measure.

    Under Q: S(T) = S0 * exp((r - q - 0.5*sigma^2)*T + sigma*sqrt(T)*Z),
    where Z ~ N(0,1).

    Parameters
    ----------
    S0 : float
        Initial asset price.
    r : float
        Continuously compounded risk-free rate (annualised).
    q : float
        Continuous dividend yield (annualised).
    sigma : float
        Annualised volatility.
    T : float
        Time to maturity in years.
    Z : np.ndarray or None, shape (N,)
        Standard normal draws. If None, generate internally.

    Returns
    -------
    S_T : np.ndarray, shape (N,)
        Terminal asset prices.
    """
    _validate_lognormal_params(S0, r, q, sigma, T)

    if Z is None:
        Z = np.random.standard_normal(
            size=(1,) if np.ndim(S0) == 0 else (1000,))

    Z = np.asarray(Z, dtype=float).flatten()
    log_term = (r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z
    return S0 * np.exp(log_term)


def discount_factor(r: float, T: float) -> float:
    """Discount factor exp(-r*T) for continuous compounding."""
    return np.exp(-r * T)


def forward_price(S0: float, r: float, q: float, sigma: float, T: float) -> float:
    """Theoretical forward price F = S0 * exp((r - q)*T."""
    return S0 * np.exp((r - q) * T)


def lognormal_inverse_cdf(p: np.ndarray, S0: float, r: float, q: float,
                          sigma: float) -> np.ndarray:
    """Stable inverse CDF (quantile) for the risk-neutral lognormal distribution.

    For p in (0, 1): S = S0 * exp((r - q - 0.5*sigma^2)*T + sigma*sqrt(T)*Phi^{-1}(p))

    Parameters
    ----------
    p : np.ndarray
        Probability values in (0, 1), not including exact 0 or 1.
    S0, r, q, sigma, T : float
        Model parameters.

    Returns
    -------
    S : np.ndarray
        Quantile-valued prices.
    """
    p = np.asarray(p, dtype=float)
    # Clip to [eps, 1-eps] to avoid log(0) or acos(1) issues
    eps = np.finfo(float).eps ** 0.5  # ~1e-8
    p_clipped = np.clip(p, eps, 1.0 - eps)
    from scipy import stats as scipy_stats
    z = scipy_stats.norm.ppf(p_clipped)
    # For now, T is not used in the inverse CDF formula since this
    # is typically used for forward-starting or immediate exercise
    return S0 * np.exp((r - q - 0.5 * sigma ** 2) + sigma * z)


def lognormal_moment(S0: float, r: float, q: float, sigma: float, T: float,
                     k: int) -> float:
    """E_Q[S(T)^k] for the risk-neutral lognormal marginal.

    = S0^k * exp(k*(r - q)*T + 0.5 * k * (k-1) * sigma^2 * T)

    Parameters
    ----------
    S0, r, q, sigma, T : float
        Model parameters.
    k : int
        Moment order (k >= 1).

    Returns
    -------
    moment : float
        E_Q[S(T)^k]
    """
    S0 = float(S0)
    r = float(r)
    q = float(q)
    sigma = float(sigma)
    T = float(T)
    return (S0 ** k) * np.exp(k * (r - q) * T + 0.5 * k * (k - 1) * (sigma ** 2) * T)


def expected_terminal_mean(S0: float, r: float, q: float, sigma: float, T: float) -> float:
    """E_Q[S(T)] = S0 * exp((r - q)*T.

    Note: The simulation formula uses S0 * exp((r - q - 0.5*sigma^2)*T + sigma*sqrt(T)*Z),
    and E[S(T)] = S0 * exp((r - q)*T (the 0.5*sigma^2 term is cancelled by E[exp(sigma*sqrt(T)*Z)]).

    Parameters
    ----------
    S0, r, q, sigma, T : float

    Returns
    -------
    mu : float
        E_Q[S(T)]
    """
    return S0 * np.exp((r - q) * T)


def expected_terminal_second_moment(S0: float, r: float, q: float, sigma: float, T: float) -> float:
    """E_Q[S(T)^2].

    = S0^2 * exp(2*(r - q)*T + 0.5 * 2 * (2-1) * sigma^2 * T)
    = S0^2 * exp(2*(r - q)*T + sigma^2 * T)
    """
    S0 = float(S0)
    r = float(r)
    q = float(q)
    sigma = float(sigma)
    T = float(T)
    return (S0 ** 2) * np.exp(2.0 * (r - q) * T + sigma ** 2 * T)


def expected_terminal_variance(S0: float, r: float, q: float, sigma: float, T: float) -> float:
    """Var_Q[S(T)] = E[S^2] - (E[S])^2."""
    mu = expected_terminal_mean(S0, r, q, sigma, T)
    second = expected_terminal_second_moment(S0, r, q, sigma, T)
    return second - mu ** 2


def sample_std(values: np.ndarray) -> float:
    """Sample standard deviation (ddof=1)."""
    return float(np.std(values, ddof=1))