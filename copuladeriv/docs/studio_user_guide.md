# CopulaDeriv Studio — User Guide

*Dependence-aware Monte Carlo pricing of multi-asset rainbow options.*

## Starting the application

From the `copuladeriv/` directory (local):

```bash
pip install -r requirements.txt       # first time
streamlit run app.py
```

The sidebar contains the shared parameter panel: financial parameters
(S₁(0), S₂(0), K, r, q₁, q₂, σ₁, σ₂, T) and simulation parameters (option type,
copula, ρ, ν, path count, seed). Pages consume exactly what is shown there.

## Pages

### 1. Research Overview
Motivation, the mathematical model (GBM marginals, both payoffs, the Monte Carlo
estimator), and the framing statement: copulas here specify **terminal
dependence** over risk-neutral lognormal marginals — not a dynamically
consistent stochastic process.

### 2. Pricing Laboratory
Runs one experiment through the existing engine
(`pricing.monte_carlo_*_price…`). Displays price, standard error, 95% CI, paths,
runtime, paths/second, seed and the exact engine function called. Confidence
intervals are sampling intervals, not guaranteed bounds. Export JSON/CSV, save
to history.

### 3. Copula Comparison
Gaussian vs Student-t with **identical marginals** and **common random numbers**
(same seed → shared latent Gaussian vector). Reports full side-by-side metrics,
the paired difference ΔV, paired SE (from pathwise discounted payoff
differences), paired CI and percentage difference (suppressed when the Gaussian
price is ≈0). Differences within two paired SEs are explicitly labelled *not
statistically distinguishable*.

### 4. Sensitivity Analysis
Four sweeps, each varying one parameter only: ρ grid, ν grid, either volatility,
strike. Paired ρ mode plots both prices with 1.96·SE bars and ΔV with paired SE
bars. All grids are editable; every sweep exports CSV.

### 5. Tail Dependence
Theoretical λ's (Gaussian 0; Student-t formula with the ν+1 t-CDF) versus
finite-threshold empirical estimates from regenerated copula uniforms at
thresholds u ∈ {0.90, 0.95, 0.975, 0.99} and their lower counterparts. Estimates
based on <30 joint exceedances are flagged as unreliable; counts are shown for
every estimate.

### 6. Greeks and Risk
Honest status page: **rainbow-contract Greeks are not implemented** and are not
displayed. Implemented instead: the analytic single-asset Black–Scholes Greeks
for asset 1 as a validation reference, plus payoff-distribution analytics
(mean, sd, quantiles, P(payoff=0), VaR and CVaR with the documented unhedged
-loss convention `loss = e^{rT}·V₀ − e^{−rT}·H_T`). This is the risk of the
terminal payoff distribution, not the market risk of a hedged book.

### 7. Monte Carlo Convergence
Price/CI, SE (with a 1/√N reference line), CI width and runtime versus path
count. Streaming chunks of ≤100k paths keep memory bounded; sub-stream seeds are
documented on the page.

### 8. Performance Benchmarks
Runs the Python engine live on the current machine (median over repetitions,
min/median/max variability, throughput, price ± SE of the last repetition) and
documents the C++20 engine as a **local-only** workflow with verbatim build
commands. Throughput is never presented as HFT latency.

### 9. Experiment History
Loads `research/dashboard_experiments/experiments.jsonl`, lists experiments,
compares selections by differing fields, exports CSV/JSONL, allows clearing.
Cloud storage is non-durable — export to keep results.

### 10. Methodology and Assumptions
The complete statistical contract (shared χ² rule, ρ ≠ Pearson correlation,
paired-SE rule, CI semantics, significance language), modelling assumptions,
numerical limitations, and the runtime environment report.

## Reproducibility rules of thumb

- Same seed + same inputs ⇒ same numbers (engine-verified for every sampler).
- To reproduce an exported JSON: set the exact parameters and seed from the
  record in the sidebar and press Run on the corresponding page.
- To compare copulas fairly, keep *all* financial parameters and the seed
  identical and change only the copula selector (ρ, ν are the copula's own
  parameters).
