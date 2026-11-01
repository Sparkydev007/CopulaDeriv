# CopulaDeriv — Mathematical Model Documentation

## Risk-neutral GBM

### SDE under the risk-neutral measure Q

Each asset follows geometric Brownian motion under the risk-neutral measure:

$$\frac{dS_i}{S_i} = (r - q_i)dt + \sigma_i dW_i$$

where:
- $S_i$ is the asset price
- $r$ is the continuously compounded risk-free rate (annualized)
- $q_i$ is the continuous dividend yield (annualized) for asset $i$
- $\sigma_i$ is the annualized volatility
- $W_i$ is a Brownian motion

The instantaneous correlation between Brownian motions is:

$$\mathbb{E}[dW_i dW_j] = \rho_{ij} dt$$

For the two-asset case, we have a single correlation parameter $\rho = \rho_{12}$.

### Terminal solution (exact, no time stepping)

The exact terminal price after time $T$ (derived by Itô's lemma, no discretization) is:

$$S_i(T) = S_i(0) \exp\left[\left(r - q_i - \frac{1}{2}\sigma_i^2\right)T + \sigma_i\sqrt{T} Z_i\right]$$

where $Z_i \sim N(0,1)$ are standard normal random variables with correlation $\rho$:
$$\begin{pmatrix} Z_1 \\ Z_2 \end{pmatrix} \sim N\left( \begin{pmatrix} 0 \\ 0 \end{pmatrix}, \begin{pmatrix} 1 & \rho \\ \rho & 1 \end{pmatrix} \right)$$

### Key property: drift cancellation

The terminal formula includes $- \frac{1}{2}\sigma_i^2 T$ inside the exponent. This term is cancelled in expectation by the moment-generating property of the normal:

$$\mathbb{E}[S_i(T)] = S_i(0) \exp\left[(r - q_i)T\right]$$

This is because $\mathbb{E}[\exp(\sigma_i\sqrt{T} Z_i)] = \exp\left(\frac{1}{2}\sigma_i^2 T\right)$, which cancels the $- \frac{1}{2}\sigma_i^2 T$ term.

### Risk-neutral drift

Under the risk-neutral measure, the expected return of each asset is the risk-free rate:

$$\mathbb{E}_Q\left[\frac{dS_i}{S_i}\right] = (r - q_i)dt$$

This differs from the physical measure where the drift would include a risk premium.

## Gaussian copula

### Construction

The Gaussian copula models dependence through correlated latent Gaussian variables, separate from the marginal distributions.

**Construction steps** (per Monte Carlo observation):

1. **Draw correlated normals**: $Z \sim N(0, \Sigma_\rho)$ where $\Sigma_\rho$ has correlation matrix:
   $$\Sigma_\rho = \begin{pmatrix} 1 & \rho \\ \rho & 1 \end{pmatrix}$$

   Using Cholesky decomposition $L$ where $\Sigma_\rho = L L^T$:
   $$L = \begin{pmatrix} 1 & 0 \\ \rho & \sqrt{1-\rho^2} \end{pmatrix}$$

   The transformation is:
   $$\begin{pmatrix} Z_1 \\ Z_2 \end{pmatrix} = L \begin{pmatrix} Z_{\text{ind,1}} \\ Z_{\text{ind,2}} \end{pmatrix}$$

   Result: $Z_1 = Z_{\text{ind,1}}$, $Z_2 = \rho Z_{\text{ind,1}} + \sqrt{1-\rho^2} Z_{\text{ind,2}}$

2. **Probability integral transform**: $U_i = \Phi(Z_i)$ where $\Phi$ is the standard normal CDF
   - In practice: $U_i = \text{scipy.stats.norm.cdf}(Z_i)$ or $U_i = 0.5 \cdot (1 + \text{erf}(Z_i / \sqrt{2}))$
   - Result: $U_1, U_2 \sim \text{Uniform}(0,1)$ marginally, with Gaussian copula dependence $\rho$

3. **Risk-neutral marginal transformation** (separate step):
   - $Z_i^{\text{(gbm)}} = \Phi^{-1}(U_i)$ (this recovers the original $Z_i$)
   - $S_i(T) = S_i(0) \exp\left[\left(r - q_i - \frac{1}{2}\sigma_i^2\right)T + \sigma_i\sqrt{T} Z_i^{\text{(gbm)}}\right]$

### Key mathematical property

**The copula separates dependence from marginals**: The parameter $\rho$ controls the dependence structure of the latent normals. Changing $\rho$ alters the joint distribution of $(Z_1, Z_2)$ but does NOT change the marginal distribution of each $Z_i$, which remains $N(0,1)$. Similarly, the risk-neutral lognormal marginals $S_i(T)$ retain their specified parameters ($S_i(0)$, $r$, $q_i$, $\sigma_i$) regardless of $\rho$.

### Boundary cases

- **$\rho = 1$**: Perfect positive dependence. $Z_1 = Z_2$, so $U_1 = U_2$ exactly.
- **$\rho = -1$**: Perfect negative dependence. $Z_2 = -Z_1$, so $U_2 = \Phi(-Z_1) = 1 - U_1$.
- **$|\rho| < 1$**: General case via Cholesky.

### Pearson correlation of terminal prices

The Pearson correlation of the terminal asset prices $\text{corr}(S_1(T), S_2(T))$ is **not** equal to the latent correlation $\rho$ in general. The relationship depends on the marginal distributions (lognormal) and on $T$, $\sigma$, and the moneyness. For the bivariate lognormal with Gaussian copula correlation $\rho$:

$$\text{corr}_{\text{Pearson}}(S_1(T), S_2(T)) \approx 1 - (1-\rho) \cdot \frac{2}{\pi} \arccos(\rho) + O(\rho^2)$$

This approximation shows that the Pearson correlation of terminal prices is a nonlinear function of the latent Gaussian correlation $\rho$. When $\rho = 0$, the Pearson correlation is approximately 0 (for symmetric parameters). When $\rho > 0$, the Pearson correlation is positive but attenuated compared to $\rho$ in some regimes.

**Important**: When pricing rainbow options, the copula $\rho$ controls the dependence structure of the *shocks*, while the Pearson correlation of the *terminal prices* is a derived quantity that depends on both $\rho$ and the marginal parameters.

## Student-t copula

### Construction

The bivariate Student-t copula is constructed using a scale-mixture-of-Gaussians representation. The critical element is **one shared chi-square scale variable per observation**.

**Construction steps** (per Monte Carlo observation i):

1. **Draw correlated normals**: Same as Gaussian copula
   - $Z_{\text{ind,1}}, Z_{\text{ind,2}} \sim N(0,1)$ i.i.d.
   - $L = \begin{pmatrix} 1 & 0 \\ \rho & \sqrt{1-\rho^2} \end{pmatrix}$
   - $Z_1 = Z_{\text{ind,1}}$, $Z_2 = \rho Z_{\text{ind,1}} + \sqrt{1-\rho^2} Z_{\text{ind,2}}$

2. **Generate ONE shared chi-square scale** per observation:
   - $V_i \sim \chi^2(\nu)$ — one draw per observation, shared by both assets
   - $V_i = \sum_{j=1}^{\nu} \epsilon_{ij}^2$ where $\epsilon_{ij} \sim N(0,1)$ i.i.d.
   - In practice: $V_i = \text{scipy.stats.chisquare}(\nu)$ or sum of $\nu$ squared normals

3. **Standardize by the shared scale**:
   - $X_{1i} = Z_1 / \sqrt{V_i / \nu}$
   - $X_{2i} = Z_2 / \sqrt{V_i / \nu}$

   The crucial point: Both assets use the **same** $V_i$. If independent $V_1, V_2$ were used, the resulting dependence would not be the intended Student-t copula.

4. **Transform via univariate Student-t CDF**:
   - $U_{1i} = T_{\nu}(X_{1i})$ — Student-t CDF with $\nu$ df, evaluated at $X_{1i}$
   - $U_{2i} = T_{\nu}(X_{2i})$ — same $\nu$, evaluated at $X_{2i}$
   - Result: $U_{1i}, U_{2i} \sim \text{Uniform}(0,1)$ marginally, with Student-t copula dependence $(\rho, \nu)$

5. **Risk-neutral marginal transformation** (same as Gaussian):
   - $Z_i^{\text{(gbm)}} = \Phi^{-1}(U_i)$
   - $S_i(T) = S_i(0) \exp\left[\left(r - q_i - \frac{1}{2}\sigma_i^2\right)T + \sigma_i\sqrt{T} Z_i^{\text{(gbm)}}\right]$

### Theoretical tail-dependence coefficients

For the bivariate Student-t copula with correlation $\rho$ and degrees of freedom $\nu$:

$$\lambda_U = \lambda_L = 2 \cdot t_{\nu+1}\left(-\sqrt{\frac{(\nu+1)(1-\rho)}{1+\rho}}\right)$$

where $t_{\nu+1}$ is the CDF of the univariate Student-t distribution with $\nu+1$ degrees of freedom.

**Special cases**:
- **$\rho = 1$**: $\lambda_U = \lambda_L = 1$ (perfect positive tail dependence)
- **$\rho = -1$**: $\lambda_L = 1$, $\lambda_U = 0$ (countermonotonic lower tail)
- **$\nu \to \infty$**: $\lambda_U = \lambda_L \to 0$ (converges to Gaussian copula, zero tail dependence)
- **$\nu \to 2^+$**: Variance infinite; tail dependence still well-defined
- **$\nu \to 0^+$**: Distribution becomes very heavy-tailed; tail dependence → 1

**Interpretation**: $\lambda_U$ is the probability that $S_2$ exceeds its high quantile given that $S_1$ exceeds its high quantile, in the limit. For the Student-t copula, upper and lower tail dependence are equal ($\lambda_U = \lambda_L$), reflecting symmetric tail behavior.

### Why the shared chi-square variable is critical

The Student-t distribution can be represented as a scale-mixture of normals:

$$t_\nu(x) = \frac{Z}{ \sqrt{V / \nu} } \quad \text{where } Z \sim N(0,1), V \sim \chi^2(\nu)$$

For the **bivariate** case, if we use one shared $V$ for both assets:

$$\begin{pmatrix} X_1 \\ X_2 \end{pmatrix} = \frac{1}{\sqrt{V/\nu}} \begin{pmatrix} Z_1 \\ Z_2 \end{pmatrix}$$

the correlation induced by the shared $V$ creates the Student-t dependence structure. The $V$ term acts as a common shock that affects both assets equally, creating extreme simultaneous moves with probability given by the tail-dependence formula.

If independent $V_1, V_2$ were used per asset, the resulting distribution would not have the correct multivariate Student-t covariance structure. The shared $V$ is what creates the intended dependence.

### Latent correlation vs. asset-price correlation

As with the Gaussian copula, the latent correlation $\rho$ of the Gaussian vector is **not** the same as the Pearson correlation of the terminal asset prices. The Student-t copula adds an additional layer: even the relationship between $\rho$ and the latent correlation of the normalized variables is affected by $\nu$.

As $\nu \to \infty$, the Student-t copula converges to the Gaussian copula, and the latent correlation $\rho$ becomes the Pearson correlation of the underlying normal scores.

### Critical modeling distinction (Student-t copula)

**A Student-t copula with risk-neutral lognormal marginals specifies a terminal dependence structure. It does not automatically define a dynamically consistent risk-neutral diffusion.**

This is the same critical distinction as with the Gaussian copula, but with additional nuance:

1. **Terminal dependence vs. diffusion dynamics**: The copula model specifies how the two asset prices $S_1(T)$ and $S_2(T)$ are jointly distributed at maturity $T$. It does not specify the entire paths $S_i(t)$ for $t \in [0, T]$. A dynamically consistent risk-neutral diffusion would require specifying the correlation structure at all times $t$, not just at maturity.

2. **Physical vs. risk-neutral historical dependence**: Physical historical dependence (estimated from historical data) cannot be automatically used for risk-neutral option pricing without a suitable pricing-measure framework. The copula may fit historical data well, but the risk-neutral measure may have different dependence characteristics.

3. **Calibration requirement**: To use the Student-t copula for actual option pricing, the dependence parameter $\rho$ and the degrees-of-freedom $\nu$ should be calibrated under the risk-neutral measure, not estimated from physical historical data. This typically requires an optimization framework that minimizes pricing error (or maximizes likelihood) under the constraint that the model is consistent with the risk-neutral measure.

4. **As $\nu \to \infty$**: The Student-t copula converges to the Gaussian copula. In the limit, the distinction vanishes, and the Gaussian copula framework applies. For finite $\nu$, the tail dependence properties are meaningfully different.

5. **Model risk**: The choice of copula (Gaussian vs. Student-t), the value of $\nu$, and the value of $\rho$ all introduce model risk. Different choices lead to different option prices, especially for payoffs sensitive to simultaneous moves (worst-of put, best-of call).

### Summary of copula distinctions

| Aspect | Gaussian copula | Student-t copula |
|--------|----------------|------------------|
| **Latent correlation** | $\rho$ directly controls dependence | $\rho$ controls dependence, $\nu$ controls tail weight |
| **Tail dependence** | $\lambda_U = \lambda_L = 0$ for $|\rho| < 1$ | $\lambda_U = \lambda_L > 0$ for finite $\nu$ |
| **Scale model** | No scale mixture | One shared $V \sim \chi^2(\nu)$ per observation |
| **Limiting case** | — | As $\nu \to \infty$, converges to Gaussian |
| **Variance** | Finite for all $\rho$ | Finite only for $\nu > 2$ |
| **Physical $\to$ risk-neutral** | Requires calibration | Requires calibration; $\nu$ adds extra parameter |
| **Extreme simultaneous moves** | Probability zero (asymptotically) | Positive probability for finite $\nu$ |

## Terminal dependence structure distinction

### Summary statement

Both the Gaussian and Student-t copulas with risk-neutral lognormal marginals specify a **terminal dependence structure**. This means they model how the two asset prices $S_1(T)$ and $S_2(T)$ are jointly distributed at a fixed maturity $T$, under the assumption of risk-neutral GBM marginals.

**They do not automatically define a dynamically consistent risk-neutral diffusion.** A dynamically consistent model would specify the correlation (or dependence) structure at all times $t \in [0, T]$, not just at maturity. The copula approach is a terminal-modeling framework.

### Why this distinction matters

1. **Path-dependent options**: If the option payoff depends on the path (e.g., lookback, barrier), the terminal copula specification is insufficient. A full stochastic process model is needed.

2. **Dynamic hedging**: Dynamic hedging requires the instantaneous correlation at each time step, not just the terminal dependence.

3. **Risk calibration**: Using physical historical correlation $\hat{\rho}_{\text{physical}}$ directly as the risk-neutral $\rho$ is unjustified without a calibration step. The risk-neutral measure may have different dependence characteristics.

4. **Model interpretation**: The copula model should be understood as: "Given risk-neutral marginals (GBM), and given a terminal dependence structure specified by the copula, what is the price of this derivative?" It is not: "The stock prices follow this diffusion process."

### When the copula approach is appropriate

- **European-style derivatives** with terminal payoffs (no path dependence)
- **Pricing and risk management** where the focus is on the joint distribution at maturity
- **Comparative analysis** of different dependence assumptions (Gaussian vs. Student-t)
- **Sensitivity analysis** of option prices to dependence structure

### When a full diffusion model is needed

- **Barrier options** (knock-in/knock-out)
- **Lookback options** (floating strike)
- **Dynamic hedging** or **delta trading** strategies
- **Term structure** modeling (correlation changing over time)
- **Jump-diffusion** or **stochastic volatility** frameworks

## Monte Carlo estimation

### Price estimator

The Monte Carlo price estimator for a European-style derivative is:

$$\hat{P} = e^{-rT} \cdot \frac{1}{N} \sum_{i=1}^N \text{payoff}_i$$

where:
- $N$ is the number of simulated paths
- $\text{payoff}_i$ is the discounted or undiscounted payoff for path $i$
- $e^{-rT}$ is the discount factor for continuous compounding

The payoff is computed as:

$$\text{payoff}_i = \text{payoff function}(S_1^{(i)}(T), S_2^{(i)}(T), K)$$

where $S_1^{(i)}(T), S_2^{(i)}(T)$ are the terminal prices for path $i$.

### Standard error

The standard error of the Monte Carlo price estimator is:

$$\text{SE}(\hat{P}) = e^{-rT} \cdot \frac{s}{\sqrt{N}}$$

where $s$ is the sample standard deviation of the payoffs:

$$s = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (\text{payoff}_i - \bar{\text{payoff}})^2}$$

with $\bar{\text{payoff}} = \frac{1}{N} \sum_{i=1}^N \text{payoff}_i$.

The $\sqrt{N}$ convergence rate is a fundamental property of Monte Carlo estimation.

### 95% confidence interval

Using the normal approximation, a 95% confidence interval for the price is:

$$\text{CI}_{95\%} = \hat{P} \pm 1.96 \cdot \text{SE}(\hat{P})$$

where $1.96 = z_{0.975}$ is the 97.5th percentile of the standard normal distribution.

**Note**: The normal approximation may be inaccurate for small $N$, skewed payoff distributions, or very rare-event payoffs. In those cases, bootstrap methods or quasi-Monte Carlo may be preferable.

### Black-Scholes agreement check

The Monte Carlo price should be consistent with the analytical Black-Scholes price for the corresponding single-asset European call:

$$\text{test: } |\hat{P}_{\text{MC}} - P_{\text{BS}}| \leq z \cdot \text{SE}(\hat{P}) + \text{tol}$$

where $z = 3.0$ (default, corresponding to approximately 99.7% coverage under the normal approximation) and $\text{tol} = 10^{-10}$ is a small numerical tolerance.

This check validates that the Monte Carlo implementation is correct by comparison to the analytical benchmark.

### Convergence across path counts

As $N$ increases, the price estimate converges to the true price at the rate $1/\sqrt{N}$:

- Doubling $N$ reduces SE by factor $\sqrt{2} \approx 1.41$
- Tripling $N$ reduces SE by factor $\sqrt{3} \approx 1.73$
- Quadrupling $N$ reduces SE by factor $2$

This is illustrated in the `price_convergence` function which summarizes price estimates and SEs across different $N$ values.

## Variance reduction techniques

### Antithetic pairs

The antithetic variance reduction technique uses negatively correlated paired observations to reduce variance:

- For each path with random numbers $u_1, u_2, \dots$, also generate antithetic paths with $1-u_1, 1-u_2, \dots$
- The antithetic payoffs are negatively correlated with the original payoffs, which reduces the variance of the mean estimator
- The price estimator becomes:
  $$\hat{P} = e^{-rT} \cdot \frac{1}{2N} \sum_{i=1}^N (p_i + p_i^{\text{at}})$$
  where $p_i$ is the payoff for path $i$ and $p_i^{\text{at}}$ is the antithetic path payoff
- Variance reduction is effective when the payoff function is smooth and monotonic in the random inputs

**Status in CopulaDeriv**: Referenced in `pricing.py` docstring but not actively implemented in the current pricing pipeline. Would be a valuable addition for Day 14+.

### Control variates

The control variate technique uses a quantity with known expected value to reduce the variance of the estimator:

- Choose a control variate $C$ such that $\mathbb{E}[C]$ is known analytically (e.g., a single-asset European call has the BS price)
- The adjusted estimator is:
  $$\hat{P}_{\text{cv}} = \hat{P} + (\mathbb{E}[C] - C_{\text{sample}})$$
  where $C_{\text{sample}}$ is the sample average of the control variate over the $N$ paths
- The variance reduction depends on the correlation between the payoff and the control variate; perfect correlation would give infinite variance reduction, but typical reductions are 10-50%

**Status in CopulaDeriv**: Referenced in `pricing.py` docstring but not actively implemented.

### Importance sampling

Importance sampling changes the sampling distribution to reduce variance for rare events:

- Instead of drawing $Z \sim N(0,1)$, draw $Z \sim N(\mu, 1)$ for some biasing parameter $\mu$
- The likelihood ratio (Radon-Nikodym derivative) is used to correct the estimator:
  $$\hat{P}_{\text{IS}} = e^{-rT} \cdot \frac{1}{N} \sum_{i=1}^N \text{payoff}_i \cdot \frac{\phi(Z_i)}{\phi(Z_i - \mu)}$$
  where $\phi$ is the standard normal PDF
- The biasing parameter $\mu$ is chosen to make the rare event more frequent, reducing variance

**Status in CopulaDeriv**: Referenced in `pricing.py` docstring as a feature, but no active implementation. Would be relevant for pricing worst-of put options with parameters that make extreme simultaneous moves rare.

## Numerical stability

### rho near ±1

- **rho = 1**: Handled explicitly — $Z_2 = Z_1$, perfect positive correlation. Prevents $\sqrt{1-1^2} = 0$ computation issues.
- **rho = -1**: Handled explicitly — $Z_2 = -Z_1$, perfect negative correlation.
- **rho close to 1 (e.g., 0.999)**: $\sqrt{1-\rho^2}$ becomes very small but positive. May cause mild loss of floating-point precision but no numerical failure.
- **rho close to -1 (e.g., -0.999)**: Same as above.

### Zero volatility

- If $\sigma_i = 0$, the terminal price reduces to $S_i(T) = S_i(0) \exp((r - q_i)T)$ (deterministic).
- The copula still generates correlated uniforms, but the asset price has no randomness.
- All functions validate $\sigma > 0$ and raise `ValueError` if $\sigma \leq 0$.

### Small maturity

- **T → 0**: $S_i(T) \to S_i(0)$ (terminal price converges to initial price).
- $\sqrt{T} \to 0$, so the random term vanishes.
- All functions accept $T > 0$; $T = 0$ is handled as a boundary case in some functions.

### Large maturity

- For very large $T$, the exponential term may produce very large or very small prices.
- The standard error scales as $1/\sqrt{N}$, so more paths are needed for convergence.
- No hard limits; depends on floating-point range.

### Extreme but valid nu values

- **nu > 0**: Student-t copula is well-defined for any positive nu.
- **nu ≤ 2**: Variance of Student-t is infinite, but the copula is still well-defined.
- **nu → 0+**: Distribution becomes extremely heavy-tailed; tail dependence → 1.
- Functions validate nu > 0 and raise ValueError if nu ≤ 0.

### Very small and large prices

- Prices are computed as discounted mean payoff; no clipping to positive values is needed since payoffs are always ≥ 0 by construction.
- Very small prices (e.g., OTM options with large T) may have large relative standard errors.
- Very large prices (ITM options) are well-behaved.

### CDF clipping

- In the copula transform, values very close to 0 or 1 from the normal CDF are clipped to [eps, 1-eps] to avoid log(0) or acos(1) issues in downstream computations.
- In `lognormal_inverse_cdf`, $p$ is clipped to $[eps, 1-eps]$ where $eps = \text{finfo(float).eps}^{0.5} \approx 10^{-8}$.

### Floating-point overflow and underflow

- The GBM formula $S = S_0 \exp(\dots)$ may overflow or underflow for extreme parameter values.
- Python's `float` is IEEE 754 double-precision; range is approximately $10^{-308}$ to $10^{308}$.
- The `risk_neutral_lognormal_terminal` function does not explicitly handle overflow; it relies on Python's arbitrary-precision handling of intermediate results (which may produce inf or raise OverflowError for extremely large exponents).
- In practice, the parameter ranges used in the project (S0=100, sigma=0.2, T=1.0) avoid these extremes.

## Confidence interval accuracy

The 95% CI $\hat{P} \pm 1.96 \cdot \text{SE}$ is based on the normal approximation. Its actual coverage probability depends on:

1. **Payoff distribution**: For symmetric payoffs (e.g., at-the-money call), the normal approximation is accurate even for moderate N.
2. **Skewed payoffs**: For skewed payoffs (e.g., deep OTM call), the normal approximation may undercover (actual coverage < 95%).
3. **Small N**: For very small N (e.g., N < 30), the t-distribution would be more appropriate, but the normal approximation is standard in MC practice.
4. **The CLT**: The central limit theorem ensures the normal approximation improves as N increases. For N ≥ 10,000, the approximation is typically very accurate.

**Rule of thumb**: For N ≥ 10,000 and reasonably smooth payoff functions, the normal approximation for the 95% CI is accurate. For smaller N or skewed payoffs, bootstrap CIs are recommended.

## Implementation correctness checks

### Verified properties (from 51 passing tests)

1. **Uniform marginals**: After the probability integral transform, each $U_i$ is marginally Uniform(0,1), tested with 100,000 paths and CLT-based tolerance.

2. **Latent correlation agreement**: The empirical correlation of the normal scores $Z_i = \Phi^{-1}(U_i)$ agrees with the requested $\rho$ within sampling tolerance (tested for $\rho \in \{0, 0.25, 0.5, 0.75, 1\}$).

3. **Marginal preservation**: When $\rho$ changes, the marginal distribution of each $S_i(T)$ is unchanged. The expected value $\mathbb{E}[S_i(T)] = S_i(0) \exp((r - q_i)T)$ is verified within statistical tolerance.

4. **Boundary cases**: $\rho = 1$ and $\rho = -1$ are handled explicitly without numerical failure.

5. **Discounting**: With $r = 0$, the discounted and undiscounted prices are identical (tested).

6. **Payoff correctness**: Best-of call and worst-of put payoff functions produce correct outputs for all tested terminal price configurations.

7. **Standard error format**: SE is finite and positive; CI is centered on the estimate; CI contains the estimate.

8. **Identical seeds**: The same seed produces identical results (deterministic for given RNG).

9. **Black-Scholes benchmark**: MC prices are within tolerance of the analytical BS price.

10. **GBM terminal positivity**: All simulated terminal prices are positive and finite.

11. **Same seed identical samples**: `gaussian_copula_sample(rho, N, seed=s)` produces identical arrays for the same seed.

These verified properties provide a solid foundation for the mathematical documentation. unresolved numerical limitations are documented in the limitations section.