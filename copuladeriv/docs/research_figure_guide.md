# Research Figure & Table Export Guide

How to produce reproducible figures for the research manuscript from CopulaDeriv
Studio (or its scripts). Every figure must state model parameters, copula,
path count, seed and the estimator's uncertainty.

## Exportable data sources

| Source | How to obtain | CSV/JSON |
|---|---|---|
| ρ sensitivity + paired ΔV | Copula Comparison / Sensitivity Analysis | ⬇ buttons on the pages |
| ν sensitivity | Sensitivity Analysis → ν tab | CSV |
| Volatility / strike sweeps | Sensitivity Analysis | CSV |
| Tail dependence (theoretical + empirical) | Tail Dependence page | CSV (empirical) |
| Monte Carlo convergence | Convergence page | CSV |
| Single experiments | Pricing Laboratory | JSON / CSV |
| Python throughput | Performance Benchmarks | CSV |
| Full experiment history | Experiment History page | CSV / JSONL |

## Recommended manuscript figures

1. **Price vs ρ, both copulas** — from the paired ρ sweep. Plot
   `gaussian_price` and `studentt_price` with ±1.96·SE bars; overlay
   `paired_price_difference` with ±1.96·paired-SE bars in a second panel.
   *Caption must state:* identical risk-neutral marginals, path count, seed,
   paired-CRN design.
2. **Student-t price vs ν** — ν sweep with a horizontal Gaussian reference line;
   document convergence toward the Gaussian copula and its rate.
3. **Theoretical tail dependence λ(ρ)** — curves for ν ∈ {3, 5, 10, 30} with the
   Gaussian zero line; delivered by the Tail Dependence page figure.
4. **Empirical tail dependence vs threshold** — bar chart with
   `empirical_lambda` and a secondary annotation of `joint_exceedances`;
   visually flag cells with <30 exceedances as unreliable.
5. **Monte Carlo convergence** — price vs N with CI bars, SE vs N with the
   1/√N reference on log-log axes, runtime vs N.
6. **Throughput comparison** — Python paths/second from the Benchmarks page;
   add C++ numbers only from the local workflow *with* compiler/flags/CPU/OS
   documented and a like-for-like workload. Never label this latency.

## Style requirements

- Titles state the contract ("Best-of call", "Worst-of put") and the copula.
- Axis labels carry units ("Years", "Per unit σ", "Paths", "USD").
- Legends name copula and ν explicitly ("Student-t, ν = 5").
- Every figure caption lists: parameter set, N, seed(s), and the statement that
  error bars are 1.96·SE Monte Carlo sampling intervals.
- Tables include N and SE columns next to every price.
- Do not derive "publication-quality" claims from a single-point estimate —
  claims of copula differences require paired differences with paired SEs
  (and ideally repeated independent seeds).

## Reproducibility checklist for each exported figure

- [ ] Seeds recorded (one per model; identical seed within paired designs).
- [ ] N and runtime recorded.
- [ ] Full parameter set from the JSON export attached.
- [ ] Error bars defined as 1.96·SE (or 1.96·paired-SE) in the caption.
- [ ] Theoretical positions distinguished from empirical estimates
      (especially in tail-dependence exhibits).
