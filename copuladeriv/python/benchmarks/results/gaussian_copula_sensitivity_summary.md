# Gaussian Copula Sensitivity Experiment Results

## Experiment Configuration

- **Initial assets**: S1=S2=100.0
- **Strike**: K=100.0
- **Risk-free rate**: r=0.04
- **Dividend yields**: q1=0.01, q2=0.015
- **Volatilities**: sigma1=0.2, sigma2=0.25
- **Time to maturity**: T=1.0
- **Number of paths**: 100000
- **Seed**: 42
- **Rho values**: [-0.75, -0.25, 0.0, 0.25, 0.75]

## Results

| rho | Latent Correlation | Best-of Call Price | Best-of Call SE | Worst-of Put Price | Worst-of Put SE |
|-----|-------------------|-------------------|-----------------|-------------------|-----------------|
| -0.75 | -0.7341 | 19.7458 | 0.0566 | 14.4084 | 0.0350 |
| -0.25 | -0.2413 | 18.3667 | 0.0590 | 13.3203 | 0.0369 |
| 0.00 | -0.0016 | 17.5332 | 0.0596 | 12.7055 | 0.0375 |
| 0.25 | 0.2383 | 16.5774 | 0.0598 | 12.0136 | 0.0378 |
| 0.75 | 0.7340 | 13.9290 | 0.0581 | 10.2074 | 0.0375 |

## Observations

- The best-of call price **decreases** as rho increases from -0.75 to 0.75,
  reflecting the increased probability that both assets are below strike
- The worst-of put price **decreases** as rho increases from -0.75 to 0.75,
  reflecting the increased probability that at least one asset is above strike
- At rho=0 (independence), the results represent the baseline case
- The latent Gaussian correlation matches the requested rho within sampling tolerance
- Standard errors are smaller for the worst-of put than the best-of call

- CSV results written to: copuladeriv\python\benchmarks\results\gaussian_copula_sensitivity.csv
- Markdown summary written to: copuladeriv\python\benchmarks\results\gaussian_copula_sensitivity_summary.md
