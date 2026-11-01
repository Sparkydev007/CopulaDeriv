# CopulaDeriv Studio — Repository Architecture Audit

*Date: Day 15 audit of the actual repository, performed by direct inspection of source
files, not by trusting README descriptions.*

## 1. Scope

This document records what CopulaDeriv Studio can reuse from the existing repository,
what is missing, how the dashboard integrates with the validated pricing engine, and
which components must **not** be rewritten.

Repository layout relevant to Studio:

```text
copuladeriv/
    app.py                    # Streamlit application (Day 15)
    studio_core.py            # Experiment controller layer (Day 15)
    pyproject.toml            # package metadata (name copuladeriv, v0.1.0)
    python/
        copuladeriv/          # validated reference package
            copulas.py        # Gaussian + Student-t copula samplers, tail dependence
            marginals.py      # risk-neutral lognormal terminal distribution
            payoffs.py        # best_of_call, worst_of_put, european_call
            pricing.py        # MC pricing engine (Cholesky, Gaussian copula, Student-t)
            statistics.py     # price_se_ci, sample_std, price_convergence, bs_agreement_check
            tests/            # 51 passing pytest tests (test_gbm, test_payoffs,
                              #   test_pricing, test_copulas)
        experiments/          # day4/day10 scripts incl. variance reduction (see §4)
        benchmarks/           # benchmark scripts + results
    cpp/                      # C++20 engine; compiled locally (copuladeriv.exe)
    docs/, reports/, benchmarks/
```

## 2. Existing reusable components (verified against source)

| Component | Module / function | Status |
|---|---|---|
| Gaussian copula uniforms | `copulas.gaussian_copula_sample(rho, N, seed)` (Cholesky / direct) | Implemented + tested |
| Student-t copula uniforms, **one shared chi-square W per observation** | `copulas.student_t_copula_sample(rho, nu, N, seed)` | Implemented + tested |
| rho / nu validation | `copulas.validate_rho`, `copulas.validate_nu` | Implemented + tested |
| Theoretical tail dependence | `copulas.student_t_tail_dependence_theoretical(nu, rho)`; `copulas.gaussian_tail_dependence(nu, rho)` | Implemented; formula = `2*t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho)))` |
| Empirical tail dependence | `copulas.empirical_tail_dependence_upper(U, u)`, `.empirical_tail_dependence_low(U, u)` | Implemented (finite-threshold estimates) |
| Risk-neutral lognormal terminal marginal | `marginals.risk_neutral_lognormal_terminal(S0, r, q, sigma, T, Z)`; `discount_factor` | Implemented + tested |
| Rainbow payoffs | `payoffs.best_of_call(S1_T, S2_T, K)`, `payoffs.worst_of_put(S1_T, S2_T, K)` | Implemented + tested |
| Cholesky-baseline GBM simulation | `pricing.simulate_terminal_prices(...)` | Implemented + tested |
| Gaussian-copula terminal simulation | `pricing.simulate_terminal_prices_copula(...)` | Implemented + tested |
| MC pricing — Cholesky / Gaussian copula / Student-t, both contracts | `pricing.monte_carlo_{best_of_call,worst_of_put}_price({...}_copula)(..., rho, seed)`, `..._student_t(..., rho, nu, seed)` | Implemented + tested; return `price`, `standard_error`, `confidence_interval`, `num_scenarios` |
| CI helper | `statistics.price_se_ci(price, se, confidence)` | Implemented + tested |
| Analytic Black-Scholes European call **with delta, gamma, vega, theta** | `pricing.black_scholes_european_call(S0, K, r, q, sigma, T)` | Implemented + tested (single asset only) |
| BS agreement check | `statistics.bs_agreement_check(mc_price, bs_price, mc_se)` | Implemented |
| C++20 engine | `cpp/` sources; local build artifact `copuladeriv/cpp/build/copuladeriv.exe` (g++ toolchain present; CMake not on PATH) | Built locally; **cloud-deployable? No** |

Reproducible randomness: all samplers accept an integer seed and use
`np.random.default_rng(seed)`; the same seed reproduces the same samples (tested).

## 3. Integration strategy (what Studio calls)

Studio is a **thin controller + visualization layer** on top of the package above.

1. **Pricing Laboratory** calls the existing pricing functions directly
   (`monte_carlo_*_price`, `..._price_copula`, `..._price_student_t`). No formula is
   duplicated; results are rendered as returned.
2. **Copula Comparison** needs *paired* common-random-number statistics, which the
   existing pricing functions do not expose (they return only summary statistics).
   Studio therefore calls the engine's building blocks — `copulas.*_copula_sample`,
   `marginals.risk_neutral_lognormal_terminal`, `marginals.discount_factor`,
   `payoffs.best_of_call / worst_of_put` — and composes them in
   `copuladeriv/studio_core.py`. This is **composition, not reimplementation**: every
   pricing formula lives in the package. The paired SE is the sample std (ddof=1) of
   the pathwise discounted payoff *differences* divided by sqrt(N) — not a combination
   of two independent SEs.
   - CRN property used: with the same integer seed, `gaussian_copula_sample` and
     `student_t_copula_sample` both draw `standard_normal((2, N))` first, so the latent
     Gaussian vector is common across the two models; only the Student-t adds a
     subsequent chi-square draw. This yields genuinely paired samples.
3. **Tail Dependence** calls `copulas.student_t_tail_dependence_theoretical`,
   `copulas.gaussian_tail_dependence`, and the empirical estimators on samples
   regenerated with the recorded seed.
4. **Monte Carlo Convergence** reuses marginal transform + payoff + `price_se_ci`.
5. **Experiment History** stores JSON records (seed, parameters, results) under
   `copuladeriv/research/dashboard_experiments/`; export as JSON/CSV.

## 4. Missing / not-validated features (kept honest)

| Feature | Reality | Studio decision |
|---|---|---|
| Antithetic variates in the pricing engine | Exists only as a Day-10 experiment script (`python/experiments/day10_antithetic.py`), loop-per-pair, with results CSV. Not a package function. | **Not offered** as a Pricing-Lab estimator. Correctly implemented vectorized antithetic estimators would be new engine code + tests (a Day 16 candidate). |
| Control variates | Day-10 script only; coefficient estimation/evaluation separation flagged by the experiment itself as needing care. | Not offered; referenced from Methodology page. |
| Importance sampling | No implementation found in the repository. | Not shown anywhere in the app. |
| Rainbow-option Greeks (finite-difference delta/vega of best-of call / worst-of put) | **Not implemented** (`pricing.py` header mentions finite-difference delta, but no such function exists). | Greeks page shows only the *implemented* analytic single-asset Black-Scholes Greeks as a reference, and explicitly marks rainbow Greeks "not implemented", with what building them would require. |
| Value-at-Risk / CVaR analytics | Payoff distributions can be computed via engine building blocks; VaR/CVaR itself is not in the package. | Offered as explicitly-labelled payoff-distribution analytics with a documented loss convention, computed from engine outputs. |
| C++ in the cloud app | Local build only (g++ on this machine). Streamlit Community Cloud cannot run this artifact. | C++ / Benchmarks page explains the local workflow; the app remains fully functional without the C++ engine. |

## 5. Mathematical assumptions the dashboard must preserve

- Terminal-dependence specification: copulas are applied to **risk-neutral lognormal
  marginals at T**. This is *not* a dynamically consistent risk-neutral diffusion;
  the Student-t copula experiment is not automatically equivalent to changing
  physical-market dynamics. (Disclosed on Research Overview and Methodology pages.)
- Marginals fixed while dependence varies: changing rho/nu must not change
  `E_Q[S_i(T)] = S_i(0) * exp((r - q_i) * T)` (engine-verified by `test_marginals_unchanged_when_rho_changes`).
- Latent copula rho ≠ Pearson correlation of terminal asset prices (the engine returns
  rho as a parameter, with the distinction documented).
- The Student-t sampler uses **one shared chi-square draw per multivariate observation**
  (per-asset independent scales are invalid; available only as an educational flag).
- Historical physical-measure dependence is **never** silently reused as risk-neutral
  dependence (no historical data is used anywhere in Studio).

## 6. Potential compatibility issues

- **Package not pip-installed locally**: tests are run from `copuladeriv/python` with
  cwd on `sys.path`. Studio bootstraps `sys.path` to `<app_dir>/python` so it works
  both locally and from a fresh clone.
- Python 3.13.5 locally; numpy/scipy/matplotlib used by package; streamlit/plotly/pandas
  added for Studio (`requirements.txt` in the app directory).
- Large path counts must be capped for the free Streamlit environment (memory/time);
  the app reports path counts, runtime and SE with every result and explains the
  deployment limits.
- Experiment history written to the repository folder is **not durable** on Streamlit
  Community Cloud (instance restarts wipe it); export is provided.

## 7. Components that must not be rewritten

- `copuladeriv/python/copuladeriv/*` — all financial formulas, samplers and statistics.
- The existing tests and their tolerances.
- Existing experiment CSV result files (Day-1..Day-10 research record).

Studio may compose them, wrap them with timing/validation, and visualize their outputs.
