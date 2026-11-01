"""Monte Carlo pricing engine for multi-asset rainbow options.

Day 1 baseline: risk-neutral correlated Geometric Brownian Motion (GBM)
with European rainbow option payoffs. Independent Black-Scholes benchmark.

Supports:
- Monte Carlo pricing for best-of call and worst-of put
- Standard error and 95% confidence interval
- Independent Black-Scholes European call benchmark
- Gaussian copula dependence model
- Finite-difference Greeks (delta)
"""

from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats

from .payoffs import best_of_call, worst_of_put
from .statistics import price_se_ci, sample_std



# ---------------------------------------------------------------------------
# Risk-neutral terminal price simulation via Cholesky (Day 1 baseline)
# ---------------------------------------------------------------------------

def simulate_terminal_prices(S1_0: float, S2_0: float, r: float, q1: float,
                             q2: float, sigma1: float, sigma2: float,
                             T: float, rho: float, N: int,
                             seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Simulate terminal prices for two assets under risk-neutral correlated GBM.

    Uses the exact terminal GBM formula (no time stepping):
        S_i(T) = S_i(0) * exp((r - q_i - 0.5*sigma_i^2)*T + sigma_i*sqrt(T)*Z_i)
    where Z_1, Z_2 are standard normal with correlation rho.

    The correlated normals are generated via Cholesky decomposition.

    Parameters
    ----------   S1_0, S2_0 : float
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
    # Z0 = independent normal -> Z_1
    # Z1 = rho * Z0 + sqrt(1-rho^2) * independent normal -> Z_2
    sqrt_one_minus_rho2 = np.sqrt(np.maximum(0.0, 1.0 - rho * rho))

    Z0 = Z_indep[0]  # independent -> Z_1
    Z1 = rho * Z0 + sqrt_one_minus_rho2 * Z_indep[1]  # correlated -> Z_2

    # Risk-neutral terminal prices
    disc1 = (r - q1 - 0.5 * sigma1 ** 2) * T
    disc2 = (r - q2 - 0.5 * sigma2 ** 2) * T
    sqrtT = np.sqrt(T)

    S1_T = S1_0 * np.exp(disc1 + sigma1 * sqrtT * Z0)
    S2_T = S2_0 * np.exp(disc2 + sigma2 * sqrtT * Z1)

    return S1_T, S2_T


# ---------------------------------------------------------------------------
# Risk-neutral terminal price simulation via Gaussian copula
# ---------------------------------------------------------------------------

def simulate_terminal_prices_copula(S1_0: float, S2_0: float, r: float, q1: float,
                                     q2: float, sigma1: float, sigma2: float,
                                     T: float, rho: float, N: int,
                                     seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Simulate terminal prices for two assets using Gaussian copula dependence.

    The construction:
        1. Draw U = (U1, U2) from a Gaussian copula with latent correlation rho
        2. Transform to normal scores: Z_i = Phi^{-1}(U_i) = scipy_stats.norm.ppf(U_i)
        3. Apply risk-neutral lognormal terminal formula:
           S_i(T) = S_i(0) * exp((r - q_i - 0.5*sigma_i^2)*T + sigma_i*sqrt(T)*Z_i)

    The risk-neutral marginal distributions are UNCHANGED when rho changes.
    Only the dependence structure varies.

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
        Gaussian copula latent correlation parameter in [-1, 1].
    N : int
        Number of simulation paths.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    S1_T, S2_T : np.ndarray, shape (N,)
        Terminal asset prices with Gaussian copula dependence.
    """
    # Step 1: Generate uniform samples from Gaussian copula
    from .copulas import gaussian_copula_sample
    U = gaussian_copula_sample(rho, N, seed=seed)

    # Step 2: Transform to normal scores via inverse CDF (probability integral transform)
    # Z_i = Phi^{-1}(U_i) ensures Z_i ~ N(0,1) with the desired dependence
    Z1 = scipy_stats.norm.ppf(U[:, 0])  # standard normal for asset 1
    Z2 = scipy_stats.norm.ppf(U[:, 1])  # standard normal for asset 2

    # Step 3: Apply risk-neutral lognormal terminal formula
    disc1 = (r - q1 - 0.5 * sigma1 ** 2) * T
    disc2 = (r - q2 - 0.5 * sigma2 ** 2) * T
    sqrtT = np.sqrt(T)

    S1_T = S1_0 * np.exp(disc1 + sigma1 * sqrtT * Z1)
    S2_T = S2_0 * np.exp(disc2 + sigma2 * sqrtT * Z2)

    return S1_T, S2_T


# ---------------------------------------------------------------------------
# Monte Carlo price estimation for best-of call (baseline Cholesky)
# ---------------------------------------------------------------------------

def monte_carlo_best_of_call_price(S1_0: float, S2_0: float, r: float, q1: float,
                                   q2: float, sigma1: float, sigma2: float,
                                   T: float, K: float, N: int,
                                   rho: float = 0.0, seed: int | None = None) -> dict:
    """Monte Carlo price of a best-of call option.

    Uses risk-neutral correlated GBM and the best-of call payoff.

    H_T = max(max(S1_T, S2_T) - K, 0)

    Parameters
    ----------
    S1_0, S2_0 : float
        Initial asset prices.
    r : float
        Risk-free rate (annualised).
    q1, q2 : float
        Dividend yields (annualised).
    sigma1, sigma2 : float
        Volatilities (annualised).
    T : float
        Maturity in years.
    K : float
        Strike price.
    N : int
        Number of simulation paths.
    rho : float
        Correlation between the two Brownian motions in [-1, 1]. Default 0.0.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        price : float
            Estimated option price.
        standard_error : float
            Monte Carlo standard error.
        confidence_interval : tuple[float, float]
            95% confidence interval (lower, upper).
        num_scenarios : int
            Number of simulated paths.
    """
    # Simulate terminal prices (baseline Cholesky)
    S1_T, S2_T = simulate_terminal_prices(
        S1_0, S2_0, r, q1, q2, sigma1, sigma2, T, rho=rho, N=N, seed=seed
    )

    # Compute payoffs
    payoffs_arr = best_of_call(S1_T, S2_T, K)

    # Discounted mean payoff
    price = np.exp(-r * T) * np.mean(payoffs_arr)

    # Standard error of the price estimator
    # Uses sample std of discounted payoffs divided by sqrt(N)
    se = np.exp(-r * T) * sample_std(payoffs_arr) / np.sqrt(N)

    # 95% confidence interval (normal approximation)
    ci_low, ci_high = price_se_ci(price, se, confidence=0.95)

    return {
        "price": float(price),
        "standard_error": float(se),
        "confidence_interval": (float(ci_low), float(ci_high)),
        "num_scenarios": N,
    }


# ---------------------------------------------------------------------------
# Monte Carlo price estimation for worst-of put (baseline Cholesky)
# ---------------------------------------------------------------------------

def monte_carlo_worst_of_put_price(S1_0: float, S2_0: float, r: float, q1: float,
                                   q2: float, sigma1: float, sigma2: float,
                                   T: float, K: float, N: int,
                                   rho: float = 0.0, seed: int | None = None) -> dict:
    """Monte Carlo price of a worst-of put option.

    Uses risk-neutral correlated GBM and the worst-of put payoff.

    H_T = max(K - min(S1_T, S2_T), 0)

    Parameters
    ----------
    S1_0, S2_0 : float
        Initial asset prices.
    r : float
        Risk-free rate (annualised).
    q1, q2 : float
        Dividend yields (annualised).
    sigma1, sigma2 : float
        Volatilities (annualised).
    T : float
        Maturity in years.
    K : float
        Strike price.
    N : int
        Number of simulation paths.
    rho : float
        Correlation between the two Brownian motions in [-1, 1]. Default 0.0.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        price : float
            Estimated option price.
        standard_error : float
            Monte Carlo standard error.
        confidence_interval : tuple[float, float]
            95% confidence interval (lower, upper).
        num_scenarios : int
            Number of simulated paths.
    """
    # Simulate terminal prices (baseline Cholesky)
    S1_T, S2_T = simulate_terminal_prices(
        S1_0, S2_0, r, q1, q2, sigma1, sigma2, T, rho=rho, N=N, seed=seed
    )

    # Compute payoffs
    payoffs_arr = worst_of_put(S1_T, S2_T, K)

    # Discounted mean payoff
    price = np.exp(-r * T) * np.mean(payoffs_arr)

    # Standard error of the price estimator
    se = np.exp(-r * T) * sample_std(payoffs_arr) / np.sqrt(N)

    # 95% confidence interval (normal approximation)
    ci_low, ci_high = price_se_ci(price, se, confidence=0.95)

    return {
        "price": float(price),
        "standard_error": float(se),
        "confidence_interval": (float(ci_low), float(ci_high)),
        "num_scenarios": N,
    }


# ---------------------------------------------------------------------------
# Monte Carlo price estimation for best-of call (Gaussian copula)
# ---------------------------------------------------------------------------

def monte_carlo_best_of_call_price_copula(S1_0: float, S2_0: float, r: float, q1: float,
                                          q2: float, sigma1: float, sigma2: float,
                                          T: float, K: float, N: int,
                                          rho: float = 0.0, seed: int | None = None) -> dict:
    """Monte Carlo price of a best-of call option using Gaussian copula.

    Uses risk-neutral correlated GBM with Gaussian copula dependence.

    The risk-neutral marginal distributions remain unchanged when rho changes.
    Only the dependence structure varies.

    H_T = max(max(S1_T, S2_T) - K, 0)

    Parameters
    ----------
    S1_0, S2_0 : float
        Initial asset prices.
    r : float
        Risk-free rate (annualised).
    q1, q2 : float
        Dividend yields (annualised).
    sigma1, sigma2 : float
        Volatilities (annualised).
    T : float
        Maturity in years.
    K : float
        Strike price.
    N : int
        Number of simulation paths.
    rho : float
        Gaussian copula latent correlation parameter in [-1, 1]. Default 0.0.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        price : float
            Estimated option price.
        standard_error : float
            Monte Carlo standard error.
        confidence_interval : tuple[float, float]
            95% confidence interval (lower, upper).
        num_scenarios : int
            Number of simulated paths.
    """
    # Simulate terminal prices via Gaussian copula
    S1_T, S2_T = simulate_terminal_prices_copula(
        S1_0, S2_0, r, q1, q2, sigma1, sigma2, T, rho=rho, N=N, seed=seed
    )

    # Compute payoffs
    payoffs_arr = best_of_call(S1_T, S2_T, K)

    # Discounted mean payoff
    price = np.exp(-r * T) * np.mean(payoffs_arr)

    # Standard error of the price estimator
    se = np.exp(-r * T) * sample_std(payoffs_arr) / np.sqrt(N)

    # 95% confidence interval (normal approximation)
    ci_low, ci_high = price_se_ci(price, se, confidence=0.95)

    return {
        "price": float(price),
        "standard_error": float(se),
        "confidence_interval": (float(ci_low), float(ci_high)),
        "num_scenarios": N,
    }


# ---------------------------------------------------------------------------
# Monte Carlo price estimation for worst-of put (Gaussian copula)
# ---------------------------------------------------------------------------

def monte_carlo_worst_of_put_price_copula(S1_0: float, S2_0: float, r: float, q1: float,
                                          q2: float, sigma1: float, sigma2: float,
                                          T: float, K: float, N: int,
                                          rho: float = 0.0, seed: int | None = None) -> dict:
    """Monte Carlo price of a worst-of put option using Gaussian copula.

    Uses risk-neutral correlated GBM with Gaussian copula dependence.

    The risk-neutral marginal distributions remain unchanged when rho changes.
    Only the dependence structure varies.

    H_T = max(K - min(S1_T, S2_T), 0)

    Parameters
    ----------
    S1_0, S2_0 : float
        Initial asset prices.
    r : float
        Risk-free rate (annualised).
    q1, q2 : float
        Dividend yields (annualised).
    sigma1, sigma2 : float
        Volatilities (annualised).
    T : float
        Maturity in years.
    K : float
        Strike price.
    N : int
        Number of simulation paths.
    rho : float
        Gaussian copula latent correlation parameter in [-1, 1]. Default 0.0.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        price : float
            Estimated option price.
        standard_error : float
            Monte Carlo standard error.
        confidence_interval : tuple[float, float]
            95% confidence interval (lower, upper).
        num_scenarios : int
            Number of simulated paths.
    """
    # Simulate terminal prices via Gaussian copula
    S1_T, S2_T = simulate_terminal_prices_copula(
        S1_0, S2_0, r, q1, q2, sigma1, sigma2, T, rho=rho, N=N, seed=seed
    )

    # Compute payoffs
    payoffs_arr = worst_of_put(S1_T, S2_T, K)

    # Discounted mean payoff
    price = np.exp(-r * T) * np.mean(payoffs_arr)

    # Standard error of the price estimator
    se = np.exp(-r * T) * sample_std(payoffs_arr) / np.sqrt(N)

    # 95% confidence interval (normal approximation)
    ci_low, ci_high = price_se_ci(price, se, confidence=0.95)

    return {
        "price": float(price),
        "standard_error": float(se),
        "confidence_interval": (float(ci_low), float(ci_high)),
        "num_scenarios": N,
    }


# ---------------------------------------------------------------------------
# Monte Carlo price estimation for best-of call (Student-t copula)
# ---------------------------------------------------------------------------

def monte_carlo_best_of_call_price_student_t(S1_0: float, S2_0: float, r: float, q1: float,
                                               q2: float, sigma1: float, sigma2: float,
                                               T: float, K: float, N: int,
                                               rho: float = 0.0, nu: float = 5.0, seed: int | None = None) -> dict:
    """Monte Carlo price of a best-of call option using Student-t copula.

    Uses risk-neutral correlated GBM with bivariate Student-t copula dependence.

    The construction:
        1. Draw U = (U1, U2) from a bivariate Student-t copula with latent correlation rho
           and degrees of freedom nu, using ONE shared chi-square scale per observation.
        2. Transform to normal scores: Z_i = Phi^{-1}(U_i)
        3. Apply risk-neutral lognormal terminal formula:
           S_i(T) = S_i(0) * exp((r - q_i - 0.5*sigma_i^2)*T + sigma_i*sqrt(T)*Z_i)

    Key properties:
        - The risk-neutral marginal distributions remain UNCHANGED when rho or nu change.
          Only the dependence structure varies.
        - The shared chi-square scale (per observation, not per asset) is what creates
          the Student-t dependence structure.
        - Using independent per-asset scale draws (incorrect) does NOT produce a valid
          Student-t copula; the implementation uses the correct shared scale.
        - For nu > 2, the Student-t has finite variance; as nu -> inf, the Student-t
          copula converges to the Gaussian copula.
        - For nu <= 2, the variance is infinite, but the copula is still well-defined.

    H_T = max(max(S1_T, S2_T) - K, 0)

    Parameters
    ----------
    S1_0, S2_0 : float
        Initial asset prices.
    r : float
        Risk-free rate (annualised).
    q1, q2 : float
        Dividend yields (annualised).
    sigma1, sigma2 : float
        Volatilities (annualised).
    T : float
        Maturity in years.
    K : float
        Strike price.
    N : int
        Number of simulation paths.
    rho : float
        Latent Gaussian correlation parameter in [-1, 1]. Default 0.0.
    nu : float
        Degrees of freedom parameter. Use nu > 2 for finite variance. Default 5.0.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        price : float
            Estimated option price.
        standard_error : float
            Monte Carlo standard error.
        confidence_interval : tuple[float, float]
            95% confidence interval (lower, upper).
        num_scenarios : int
            Number of simulated paths.
    """
    # Simulate terminal prices via Student-t copula
    from .copulas import student_t_copula_sample
    U = student_t_copula_sample(rho, nu, N, seed=seed)

    # Transform to normal scores via inverse CDF (probability integral transform)
    # Z_i = Phi^{-1}(U_i) ensures Z_i ~ N(0,1) with the desired dependence
    Z1 = scipy_stats.norm.ppf(U[:, 0])  # standard normal for asset 1
    Z2 = scipy_stats.norm.ppf(U[:, 1])  # standard normal for asset 2

    # Apply risk-neutral lognormal terminal formula
    disc1 = (r - q1 - 0.5 * sigma1 ** 2) * T
    disc2 = (r - q2 - 0.5 * sigma2 ** 2) * T
    sqrtT = np.sqrt(T)

    S1_T = S1_0 * np.exp(disc1 + sigma1 * sqrtT * Z1)
    S2_T = S2_0 * np.exp(disc2 + sigma2 * sqrtT * Z2)

    # Compute payoffs
    payoffs_arr = best_of_call(S1_T, S2_T, K)

    # Discounted mean payoff
    price = np.exp(-r * T) * np.mean(payoffs_arr)

    # Standard error of the price estimator
    se = np.exp(-r * T) * sample_std(payoffs_arr) / np.sqrt(N)

    # 95% confidence interval (normal approximation)
    ci_low, ci_high = price_se_ci(price, se, confidence=0.95)

    return {
        "price": float(price),
        "standard_error": float(se),
        "confidence_interval": (float(ci_low), float(ci_high)),
        "num_scenarios": N,
    }


# ---------------------------------------------------------------------------
# Monte Carlo price estimation for worst-of put (Student-t copula)
# ---------------------------------------------------------------------------

def monte_carlo_worst_of_put_price_student_t(S1_0: float, S2_0: float, r: float, q1: float,
                                               q2: float, sigma1: float, sigma2: float,
                                               T: float, K: float, N: int,
                                               rho: float = 0.0, nu: float = 5.0, seed: int | None = None) -> dict:
    """Monte Carlo price of a worst-of put option using Student-t copula.

    Uses risk-neutral correlated GBM with bivariate Student-t copula dependence.

    The construction is identical to the best-of call, except the payoff:

        H_T = max(K - min(S1_T, S2_T), 0)

    Key properties are unchanged from the best-of call version:
        - Risk-neutral marginal distributions remain unchanged when rho or nu change.
        - The shared chi-square scale creates the Student-t dependence.
        - As nu -> inf, the Student-t copula converges to the Gaussian copula.

    Parameters
    ----------
    S1_0, S2_0 : float
        Initial asset prices.
    r : float
        Risk-free rate (annualised).
    q1, q2 : float
        Dividend yields (annualised).
    sigma1, sigma2 : float
        Volatilities (annualised).
    T : float
        Maturity in years.
    K : float
        Strike price.
    N : int
        Number of simulation paths.
    rho : float
        Latent Gaussian correlation parameter in [-1, 1]. Default 0.0.
    nu : float
        Degrees of freedom parameter. Use nu > 2 for finite variance. Default 5.0.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        price : float
            Estimated option price.
        standard_error : float
            Monte Carlo standard error.
        confidence_interval : tuple[float, float]
            95% confidence interval (lower, upper).
        num_scenarios : int
            Number of simulated paths.
    """
    # Simulate terminal prices via Student-t copula
    from .copulas import student_t_copula_sample
    U = student_t_copula_sample(rho, nu, N, seed=seed)

    # Transform to normal scores via inverse CDF (probability integral transform)
    # Z_i = Phi^{-1}(U_i) ensures Z_i ~ N(0,1) with the desired dependence
    Z1 = scipy_stats.norm.ppf(U[:, 0])  # standard normal for asset 1
    Z2 = scipy_stats.norm.ppf(U[:, 1])  # standard normal for asset 2

    # Apply risk-neutral lognormal terminal formula
    disc1 = (r - q1 - 0.5 * sigma1 ** 2) * T
    disc2 = (r - q2 - 0.5 * sigma2 ** 2) * T
    sqrtT = np.sqrt(T)

    S1_T = S1_0 * np.exp(disc1 + sigma1 * sqrtT * Z1)
    S2_T = S2_0 * np.exp(disc2 + sigma2 * sqrtT * Z2)

    # Compute payoffs
    payoffs_arr = worst_of_put(S1_T, S2_T, K)

    # Discounted mean payoff
    price = np.exp(-r * T) * np.mean(payoffs_arr)

    # Standard error of the price estimator
    se = np.exp(-r * T) * sample_std(payoffs_arr) / np.sqrt(N)

    # 95% confidence interval (normal approximation)
    ci_low, ci_high = price_se_ci(price, se, confidence=0.95)

    return {
        "price": float(price),
        "standard_error": float(se),
        "confidence_interval": (float(ci_low), float(ci_high)),
        "num_scenarios": N,
    }


# ---------------------------------------------------------------------------
# Independent Black-Scholes European call benchmark
# ---------------------------------------------------------------------------

def black_scholes_european_call(S0: float, K: float, r: float, q: float,
                                sigma: float, T: float) -> dict:
    """Analytical Black-Scholes price for a European call option on a single asset.

    Uses the standard closed-form formula:

        C = S0 * exp(-q*T) * Phi(d1) - K * exp(-r*T) * Phi(d2)

    where:

        d1 = [ln(S0/K) + (r - q + 0.5*sigma^2)*T] / [sigma*sqrt(T)]
        d2 = d1 - sigma*sqrt(T)

    Parameters
    ----------
    S0 : float
        Initial asset price.
    K : float
        Strike price.
    r : float
        Continuously compounded risk-free rate (annualised).
    q : float
        Continuous dividend yield (annualised).
    sigma : float
        Annualised volatility.
    T : float
        Time to maturity in years.

    Returns
    -------
    dict with keys:
        price : float
            Option price.
        delta : float
            Sensitivity to initial price.
        gamma : float
            Sensitivity to volatility.
        vega : float
            Sensitivity to volatility (per unit).
        theta : float
            Sensitivity to time decay.
    """
    S0 = float(S0)
    K = float(K)
    r = float(r)
    q = float(q)
    sigma = float(sigma)
    T = float(T)

    # Boundary case: zero or negative maturity
    if T <= 0:
        forward = S0 * np.exp(-q * T)
        intrinsic = max(forward - K, 0.0)
        return {
            "price": float(intrinsic),
            "delta": 1.0 if forward > K else 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
        }

    # Boundary case: zero volatility
    if sigma <= 0:
        forward = S0 * np.exp(-q * T)
        intrinsic = max(forward - K, 0.0)
        return {
            "price": float(intrinsic),
            "delta": 1.0 if forward > K else 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
        }

    # Standard Black-Scholes formula
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Normal CDF values
    phi_d1 = scipy_stats.norm.cdf(d1)
    phi_d2 = scipy_stats.norm.cdf(d2)

    # Option price
    price = S0 * np.exp(-q * T) * phi_d1 - K * np.exp(-r * T) * phi_d2

    # Greeks
    delta = np.exp(-q * T) * phi_d1
    gamma = np.exp(-q * T) * scipy_stats.norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    vega = S0 * np.exp(-q * T) * scipy_stats.norm.pdf(d1) * np.sqrt(T)  # per unit sigma change
    theta = -(S0 * np.exp(-q * T) * scipy_stats.norm.pdf(d1) * sigma /
              (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * phi_d2)

    return {
        "price": float(price),
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
    }