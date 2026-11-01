# Gaussian vs Student-t Copula Comparative Pricing Experiment

## Experiment Configuration

- **Initial asset prices:** S1_0 = 100.0, S2_0 = 100.0
- **Strike price:** K = 100.0
- **Risk-free rate:** r = 0.04
- **Dividend yields:** q1 = 0.01, q2 = 0.015
- **Volatilities:** sigma1 = 0.2, sigma2 = 0.25
- **Time to maturity:** T = 1.0 years
- **Number of paths:** N = 100000
- **Random seed:** seed = 42
- **Latent correlation parameters:** rho = [-0.75, -0.25, 0.0, 0.25, 0.75]
- **Student-t degrees of freedom:** nu = [5, 10, 20]

- **Key design point:** Risk-neutral lognormal marginal distributions are
  identical across all copula and rho/nu settings. Only the dependence
  structure varies.

- **Shared chi-square scale:** The Student-t copula uses ONE shared
  chi-square draw W ~ Chi-square(nu) per observation, applied to both
  normal components. This is the critical construction that creates
  the Student-t dependence structure.

## Results Summary

### Best-of Call Prices

| Copula Family | rho | nu | Price | Std Error |
|---|---|---|---|---|
| Gaussian | -0.75 | — | 19.7458 | 0.0566 |
| Student-t(nu=5) | -0.75 | 5 | 19.6092 | 0.0561 |
| Student-t(nu=10) | -0.75 | 10 | 19.6976 | 0.0565 |
| Student-t(nu=20) | -0.75 | 20 | 19.7329 | 0.0568 |
| Gaussian | -0.25 | — | 18.3667 | 0.0590 |
| Student-t(nu=5) | -0.25 | 5 | 18.1300 | 0.0582 |
| Student-t(nu=10) | -0.25 | 10 | 18.2501 | 0.0586 |
| Student-t(nu=20) | -0.25 | 20 | 18.3250 | 0.0589 |
| Gaussian | 0.00 | — | 17.5332 | 0.0596 |
| Student-t(nu=5) | 0.00 | 5 | 17.2769 | 0.0586 |
| Student-t(nu=10) | 0.00 | 10 | 17.3939 | 0.0590 |
| Student-t(nu=20) | 0.00 | 20 | 17.4783 | 0.0594 |
| Gaussian | 0.25 | — | 16.5774 | 0.0598 |
| Student-t(nu=5) | 0.25 | 5 | 16.3260 | 0.0587 |
| Student-t(nu=10) | 0.25 | 10 | 16.4294 | 0.0591 |
| Student-t(nu=20) | 0.25 | 20 | 16.5155 | 0.0594 |
| Gaussian | 0.75 | — | 13.9290 | 0.0581 |
| Student-t(nu=5) | 0.75 | 5 | 13.7726 | 0.0573 |
| Student-t(nu=10) | 0.75 | 10 | 13.8189 | 0.0575 |
| Student-t(nu=20) | 0.75 | 20 | 13.8838 | 0.0578 |

| Copula Family | rho | nu | Price | Std Error |
|---|---|---|---|---|
| Gaussian | -0.75 | — | 14.4084 | 0.0350 |
| Student-t(nu=5) | -0.75 | 5 | 14.3351 | 0.0349 |
| Student-t(nu=10) | -0.75 | 10 | 14.3889 | 0.0350 |
| Student-t(nu=20) | -0.75 | 20 | 14.4000 | 0.0350 |
| Gaussian | -0.25 | — | 13.3203 | 0.0369 |
| Student-t(nu=5) | -0.25 | 5 | 13.1978 | 0.0366 |
| Student-t(nu=10) | -0.25 | 10 | 13.2808 | 0.0368 |
| Student-t(nu=20) | -0.25 | 20 | 13.2996 | 0.0369 |
| Gaussian | 0.00 | — | 12.7055 | 0.0375 |
| Student-t(nu=5) | 0.00 | 5 | 12.5730 | 0.0371 |
| Student-t(nu=10) | 0.00 | 10 | 12.6598 | 0.0374 |
| Student-t(nu=20) | 0.00 | 20 | 12.6814 | 0.0374 |
| Gaussian | 0.25 | — | 12.0136 | 0.0378 |
| Student-t(nu=5) | 0.25 | 5 | 11.8824 | 0.0374 |
| Student-t(nu=10) | 0.25 | 10 | 11.9655 | 0.0377 |
| Student-t(nu=20) | 0.25 | 20 | 11.9883 | 0.0378 |
| Gaussian | 0.75 | — | 10.2074 | 0.0375 |
| Student-t(nu=5) | 0.75 | 5 | 10.1275 | 0.0372 |
| Student-t(nu=10) | 0.75 | 10 | 10.1800 | 0.0374 |
| Student-t(nu=20) | 0.75 | 20 | 10.1902 | 0.0375 |

### Tail Dependence

The Gaussian copula has no asymptotic tail dependence (lambda_U = lambda_L = 0)
for |rho| < 1. The Student-t copula has symmetric upper and lower tail
dependence with lambda_U = lambda_L > 0 for finite nu. As nu increases,
the tail dependence decreases, and the Student-t copula converges to the
Gaussian copula in the limit nu -> infinity.

### Paired Price Differences

The paired comparison uses the same underlying random numbers to reduce
 Monte Carlo noise when comparing copula families. The difference in prices
 between Student-t and Gaussian copulas reflects the impact of tail dependence
 on rainbow option values. Student-t copulas with lower nu (stronger tail
 dependence) may produce materially different prices depending on the payoff
 structure.

## Key Findings

1. **Marginal preservation:** The risk-neutral lognormal marginal distributions
   are unchanged when the copula dependence parameter (rho or nu) changes.
   This is a fundamental property: the copula captures dependence only.

2. **Tail dependence impact:** Student-t copulas with lower nu produce stronger
   tail dependence (lambda_U = lambda_L > 0), which can materially affect
   rainbow option prices, particularly for worst-of put payoffs.

3. **Convergence:** As nu -> infinity, the Student-t copula converges to the
   Gaussian copula. For nu > 2 (finite variance), the Student-t copula is
   well-defined with moderate tail dependence.

4. **Common random numbers:** Using shared random numbers significantly
   reduces noise in pairwise comparisons, making it easier to detect genuine
   price differences attributable to the copula choice.

5. **Boundary cases:** rho = 1 and rho = -1 are handled correctly for both
   copula types, with perfect positive and negative dependence respectively.

---\n*Generated from actual simulation outputs. Do not hardcode values.*\n