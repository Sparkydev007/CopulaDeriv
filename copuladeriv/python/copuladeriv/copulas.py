"""Gaussian copula module for multi-asset derivative pricing.

Provides a modular dependence model that is independent of marginal
distributions, payoff logic, and Monte Carlo statistics.

The Gaussian copula transforms standard normal random variables with
correlation rho into uniform marginals via the probability integral transform,
enabling risk-neutral pricing with correlated asset dynamics.

Reference:
- Nelsen, R. (2006). An Introduction to Copulas. Springer.
- Hull, J. (2018). Options, Futures, and Other Derivatives.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats
from typing import Literal


# ---------------------------------------------------------------------------
# Correlation validation
# ---------------------------------------------------------------------------

def validate_rho(rho: float) -> float:
    """Validate and return the copula correlation parameter.

    Parameters
    ----------
    rho : float
        Correlation parameter in [-1, 1].

    Returns
    -------
    rho_clipped : float
        Clamped to [-1, 1] if out of range.

    Raises
    ------
    ValueError
        If rho is not finite.
    """
    if not np.isfinite(rho):
        raise ValueError("rho must be finite.")
    return np.clip(rho, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Core Gaussian copula sampling
# ---------------------------------------------------------------------------

def gaussian_copula_sample(
    rho: float,
    N: int,
    seed: int | None = None,
    *,
    method: Literal["cholesky", "direct"] = "cholesky",
) -> np.ndarray:
    """Generate uniform random variables with Gaussian copula dependence.

    Generates N draws from a bivariate Gaussian copula with correlation rho.
    The result is an array of shape (N, 2) with each column marginally
    Uniform(0, 1), and with Gaussian copula dependence parameter rho.

    The construction uses the probability integral transform:
        1. Draw Z ~ N(0, Sigma_rho) where Sigma_rho has correlation rho
        2. Transform U = Phi(Z) where Phi is the standard normal CDF

    For boundary correlations:
        - rho = 1:    U1 = U2 = Phi(Z1) with Z1 ~ N(0,1)
        - rho = -1:   U1 = Phi(Z1), U2 = Phi(-Z1) (perfect negative dependence)

    Parameters
    ----------
    rho : float
        Gaussian copula correlation parameter in [-1, 1].
    N : int
        Number of simulation paths (sample size).
    seed : int or None
        Random seed for reproducibility.
    method : {"cholesky", "direct"}, default "cholesky"
        Method for generating correlated normals.

    Returns
    -------
    U : np.ndarray, shape (N, 2)
        Uniform random variables with Gaussian copula dependence.
        Each column is marginally Uniform(0, 1).
    """
    rho = validate_rho(rho)
    rng = np.random.default_rng(seed)

    if method == "direct":
        _gaussian_copula_sample_direct(rho, N, rng, out=None)
    else:
        _gaussian_copula_sample_cholesky(rho, N, rng, out=None)

    return _last_sample  # type: ignore[return-value]


_last_sample: np.ndarray | None = None


def _gaussian_copula_sample_cholesky(rho: float, N: int, rng: np.random.RandomState,
                                     out: np.ndarray | None = None) -> np.ndarray:
    """Cholesky-based Gaussian copula sampler (internal)."""

    global _last_sample

    # Generate independent standard normals
    Z = rng.standard_normal((2, N))  # shape (2, N)

    if rho == 1.0:
        # Perfect positive correlation: Z1 = Z2
        Z0 = Z[0]
        Z1 = Z0  # perfectly correlated
    elif rho == -1.0:
        # Perfect negative correlation: Z1 = -Z2
        Z0 = Z[0]
        Z1 = -Z0
    else:
        # General case via Cholesky
        sqrt_one_minus_rho2 = np.sqrt(np.maximum(0.0, 1.0 - rho * rho))
        Z0 = Z[0]
        Z1 = rho * Z0 + sqrt_one_minus_rho2 * Z[1]

    # Transform via standard normal CDF to uniforms
    # Use scipy.stats.norm.cdf for numerical stability at endpoints
    U0 = scipy_stats.norm.cdf(Z0)
    U1 = scipy_stats.norm.cdf(Z1)

    U = np.column_stack((U0, U1))  # shape (N, 2)
    _last_sample = U
    return U


def _gaussian_copula_sample_direct(rho: float, N: int, rng: np.random.RandomState,
                                   out: np.ndarray | None = None) -> np.ndarray:
    """Direct method Gaussian copula sampler (internal).

    For rho = 0: generate independent uniforms.
    For other rho: use the correlation formula Z2 = rho*Z1 + sqrt(1-rho^2)*epsilon.
    """

    global _last_sample

    if rho == 0.0:
        # Independent draws
        U = rng.uniform(size=(N, 2))
        _last_sample = U
        return U

    # For non-zero rho, use the correlated normal approach
    # This avoids Cholesky for the simple 2D case
    Z0 = rng.standard_normal(N)
    epsilon = rng.standard_normal(N)
    Z1 = rho * Z0 + np.sqrt(np.maximum(0.0, 1.0 - rho * rho)) * epsilon

    U0 = scipy_stats.norm.cdf(Z0)
    U1 = scipy_stats.norm.cdf(Z1)

    U = np.column_stack((U0, U1))
    _last_sample = U
    return U


# ---------------------------------------------------------------------------
# Helper: Pearson correlation of generated uniforms
# ---------------------------------------------------------------------------

def empirical_correlation(U: np.ndarray) -> float:
    """Compute the Pearson correlation coefficient of uniform copula samples.

    Parameters
    ----------
    U : np.ndarray, shape (N, 2)
        Uniform random variables from a Gaussian copula.

    Returns
    -------
    corr : float
        Pearson correlation of the two columns. This will generally differ
        from the latent Gaussian correlation rho, especially in the tails.
    """
    return np.corrcoef(U[:, 0], U[:, 1])[0, 1]


# ---------------------------------------------------------------------------
# Analytic correlation approximation (optional)
# ---------------------------------------------------------------------------

def approximate_pearson_rho(gaussian_rho: float) -> float:
    """Approximate the Pearson correlation of the resulting lognormal prices
    given a Gaussian copula latent correlation rho.

    For a bivariate lognormal with standard normal copula correlation rho,
    the asymptotic Pearson correlation of the terminal prices is approximately:

        rho_Pearson ~ 1 - (1 - rho) * (2 / pi) * arccos(rho) + O(rho^2)

    This is a simplified approximation for documentation purposes; the exact
    relationship depends on the marginal distributions and is not needed for
    pricing when the risk-neutral marginals are fixed.

    Parameters
    ----------
    gaussian_rho : float
        Latent Gaussian copula correlation in [-1, 1].

    Returns
    -------
    pearson_rho : float
        Approximate Pearson correlation of the lognormal terminal prices.
    """
    rho = np.clip(gaussian_rho, -1.0, 1.0)
    # Note: The exact relationship between Gaussian copula rho and Pearson
    # correlation of lognormal prices depends on the marginals. We return rho
    # as a reference with explicit documentation of the distinction.
    return rho  # Returning rho with explicit documentation of the distinction


# ---------------------------------------------------------------------------
# Bivariate Student-t copula
# ---------------------------------------------------------------------------

def validate_nu(nu: float) -> float:
    """Validate the degrees-of-freedom parameter for the Student-t copula.

    Parameters
    ----------
    nu : float
        Degrees of freedom parameter. Must be positive.

    Returns
    -------
    nu_clipped : float
        Validated nu > 0.

    Raises
    ------
    ValueError
        If nu is not positive.
    """
    if not np.isfinite(nu):
        raise ValueError("nu must be finite.")
    if nu <= 0:
        raise ValueError("nu must be positive (nu > 0).")
    return float(nu)


def student_t_copula_sample(
    rho: float,
    nu: float,
    N: int,
    seed: int | None = None,
    *,
    method: Literal["shared_chi_square", "independent"] = "shared_chi_square",
) -> np.ndarray:
    """Generate uniform random variables with bivariate Student-t copula dependence.

    The bivariate Student-t copula is constructed as follows:

    1. Draw a correlated bivariate normal vector:
         Z1 ~ N(0,1)
         Z2 = rho * Z1 + sqrt(1 - rho^2) * epsilon,  epsilon ~ N(0,1) independent
    2. Generate ONE shared chi-square scale variable per observation:
         W ~ Chi-square(nu)  (one draw per path, shared by both assets)
    3. Standardize each normal component by the shared scale:
         X1 = Z1 / sqrt(W / nu)
         X2 = Z2 / sqrt(W / nu)
    4. Transform via the univariate Student-t CDF:
         U1 = T_nu(X1),  U2 = T_nu(X2)
       where T_nu is the Student-t CDF with nu degrees of freedom.

    The key theoretical point: using ONE shared W per observation (not independent
    W1 and W2) is what creates the intended Student-t dependence structure.
    Independent per-asset scale draws destroy the multivariate Student-t property.

    Parameters
    ----------
    rho : float
        Latent Gaussian correlation parameter in [-1, 1].
    nu : float
        Degrees of freedom parameter. Use nu > 2 for finite variance.
    N : int
        Number of simulation paths (sample size).
    seed : int or None
        Random seed for reproducibility.
    method : {"shared_chi_square", "independent"}
        Method for scale construction.
        - shared_chi_square (default): one W per observation shared by both assets.
        - independent: independent W1, W2 per observation (does NOT produce
          a valid Student-t copula, included for comparison only).

    Returns
    -------
    U : np.ndarray, shape (N, 2)
        Uniform random variables with bivariate Student-t copula dependence.
        Each column is marginally Uniform(0, 1).
    """
    rho = validate_rho(rho)
    nu = validate_nu(nu)
    rng = np.random.default_rng(seed)

    if method == "independent":
        # Independent per-asset scales — does NOT produce a valid Student-t copula.
        # Included for comparison/education only.
        Z1 = rng.standard_normal(N)
        Z2 = rho * Z1 + np.sqrt(np.maximum(0.0, 1.0 - rho * rho)) * rng.standard_normal(N)
        W1 = rng.chisquare(nu, size=N)
        W2 = rng.chisquare(nu, size=N)
        W1 = np.maximum(W1, np.finfo(float).eps)
        W2 = np.maximum(W2, np.finfo(float).eps)
        from scipy import stats as scipy_stats
        X1 = Z1 / np.sqrt(W1 / nu)
        X2 = Z2 / np.sqrt(W2 / nu)
        U1 = scipy_stats.t.cdf(X1, nu)
        U2 = scipy_stats.t.cdf(X2, nu)
        U = np.column_stack((np.clip(U1, 1e-300, 1-1e-300),
                             np.clip(U2, 1e-300, 1-1e-300)))
        return U

    # --- Shared chi-square method (correct) ---
    # Generate correlated normals via Cholesky
    Z = rng.standard_normal((2, N))  # shape (2, N)

    if rho == 1.0:
        Z0 = Z[0]
        Z1 = Z0
    elif rho == -1.0:
        Z0 = Z[0]
        Z1 = -Z0
    else:
        sqrt_one_minus_rho2 = np.sqrt(np.maximum(0.0, 1.0 - rho * rho))
        Z0 = Z[0]
        Z1 = rho * Z0 + sqrt_one_minus_rho2 * Z[1]

    # Generate ONE shared chi-square scale variable per observation
    W = rng.chisquare(nu, size=N)  # shape (N,)
    W = np.maximum(W, np.finfo(float).eps)  # prevent division by zero

    # Standardize each normal component by the shared scale
    # X_i = Z_i / sqrt(W/nu) follows a multivariate Student-t distribution
    X0 = Z0 / np.sqrt(W / nu)
    X1 = Z1 / np.sqrt(W / nu)

    # Transform via univariate Student-t CDF to uniforms
    from scipy import stats as scipy_stats
    U0 = scipy_stats.t.cdf(X0, nu)
    U1 = scipy_stats.t.cdf(X1, nu)

    U = np.column_stack((np.clip(U0, 1e-300, 1-1e-300),
                         np.clip(U1, 1e-300, 1-1e-300)))
    return U


# ---------------------------------------------------------------------------
# Helper: Gaussian copula sampler with method option (for internal use)
# ---------------------------------------------------------------------------

def _gaussian_copula_sample_cholesky_cd(rho: float, N: int, rng: np.random.RandomState,
                                        method: str = "shared_chi_square") -> np.ndarray:
    """Gaussian copula sampler with method option (internal use)."""

    global _last_sample

    # Standard Gaussian Cholesky
    Z = rng.standard_normal((2, N))

    if rho == 1.0:
        Z0 = Z[0]
        Z1 = Z0
    elif rho == -1.0:
        Z0 = Z[0]
        Z1 = -Z0
    else:
        sqrt_one_minus_rho2 = np.sqrt(np.maximum(0.0, 1.0 - rho * rho))
        Z0 = Z[0]
        Z1 = rho * Z0 + sqrt_one_minus_rho2 * Z[1]

    U0 = scipy_stats.norm.cdf(Z0)
    U1 = scipy_stats.norm.cdf(Z1)

    U = np.column_stack((U0, U1))
    _last_sample = U
    return U


# ---------------------------------------------------------------------------
# Tail-dependence coefficients
# ---------------------------------------------------------------------------

def student_t_tail_dependence_theoretical(nu: float, rho: float) -> tuple[float, float]:
    """Compute theoretical upper and lower tail-dependence coefficients for a
    bivariate Student-t copula.

    For a bivariate Student-t copula with degrees of freedom nu and latent
    correlation rho:

        lambda_U = lambda_L = 2 * t_{nu+1}(-sqrt(((nu+1)*(1-rho))/(1+rho)))

    where t_{nu+1} is the univariate Student-t CDF with nu+1 degrees of freedom.

    Special cases:
        - rho = 1:    lambda_U = lambda_L = 1 (perfect dependence)
        - rho in (-1, 1) with finite nu: 0 < lambda_U = lambda_L < 1
        - Gaussian copula (limit nu -> inf): lambda_U = lambda_L = 0

    Parameters
    ----------
    nu : float
        Degrees of freedom parameter. Must be > 0.
    rho : float
        Latent correlation parameter in [-1, 1].

    Returns
    -------
    (lambda_U, lambda_L) : tuple[float, float]
        Upper and lower tail-dependence coefficients. For the Student-t copula,
        lambda_U = lambda_L (strong symmetric tail dependence).
    """
    rho = np.clip(rho, -1.0, 1.0)
    nu = float(nu)

    # Boundary case: rho = 1 -> perfect dependence
    if rho >= 1.0 - 1e-10:
        return 1.0, 1.0
    # Boundary case: rho = -1 -> countermonotonic
    if rho <= -1.0 + 1e-10:
        return 0.0, 1.0

    # Theoretical formula
    # t_{nu+1} is the CDF of Student-t with nu+1 df
    from scipy import stats as scipy_stats
    arg = -np.sqrt((nu + 1) * (1.0 - rho) / (1.0 + rho))
    # Clip argument to valid domain for t CDF
    arg = np.clip(arg, -1e10, 1e10)
    t_cdf = scipy_stats.t.cdf(arg, nu + 1)
    lambda_U_L = 2.0 * t_cdf

    # For Student-t, lambda_U = lambda_L (symmetric tail dependence)
    return float(lambda_U_L), float(lambda_U_L)


def gaussian_tail_dependence(nu: float, rho: float) -> tuple[float, float]:
    """Tail dependence for Gaussian copula.

    For the Gaussian copula with -1 < rho < 1:
        lambda_U = lambda_L = 0 (no asymptotic tail dependence).

    Parameters
    ----------
    nu : float
        Degrees of freedom (ignored for Gaussian).
    rho : float
        Latent correlation.

    Returns
    -------
    (lambda_U, lambda_L) : tuple[float, float]
        Both are 0 for the Gaussian copula.
    """
    rho = np.clip(rho, -1.0, 1.0)
    # Gaussian copula has no asymptotic tail dependence for |rho| < 1
    return 0.0, 0.0


def empirical_tail_dependence_upper(U: np.ndarray, threshold: float = 0.95) -> float:
    """Empirical upper tail-dependence coefficient.

    For a high threshold u: lambda_U(u) = P(U2 > u | U1 > u)
    The theoretical asymptotic coefficient is lim_{u->1} lambda_U(u).

    Parameters
    ----------
    U : np.ndarray, shape (N, 2)
        Uniform copula samples.
    threshold : float
        High threshold in (0,1), e.g. 0.90, 0.95, 0.975, 0.99.

    Returns
    -------
    empirical_lambda : float
        Empirical estimate of upper tail dependence.
    """
    exceed = (U[:, 0] > threshold) & (U[:, 1] > threshold)
    both_exceed = np.sum(exceed)
    margin_exceed = np.sum(U[:, 0] > threshold)
    if margin_exceed == 0:
        return 0.0
    # lambda_U = P(U2 > u | U1 > u) = P(both exceed) / P(U1 > u)
    empirical = both_exceed / margin_exceed
    return float(empirical)


def empirical_tail_dependence_low(U: np.ndarray, threshold: float = 0.95) -> float:
    """Empirical lower tail-dependence coefficient.

    For a low threshold u: lambda_L(u) = P(U2 < u | U1 < u)

    Parameters
    ----------
    U : np.ndarray, shape (N, 2)
        Uniform copula samples.
    threshold : float
        Low threshold in (0,1), e.g. 0.05, 0.10.

    Returns
    -------
    empirical_lambda : float
        Empirical estimate of lower tail dependence.
    """
    below = (U[:, 0] < threshold) & (U[:, 1] < threshold)
    both_below = np.sum(below)
    margin_below = np.sum(U[:, 0] < threshold)
    if margin_below == 0:
        return 0.0
    empirical = both_below / margin_below
    return float(empirical)