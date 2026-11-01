# CopulaDeriv Studio — Application Architecture

## 1. High-level architecture

```mermaid
flowchart TD
    A["Streamlit app<br/>(copuladeriv/app.py)<br/>navigation · inputs · charts · exports"] --> B["Experiment controller<br/>(copuladeriv/studio_core.py)<br/>validation · composition · history"]
    B --> C["Validated CopulaDeriv package<br/>(copuladeriv/python/copuladeriv)"]
    C --> C1["copulas<br/>Gaussian sampler<br/>Student-t sampler (shared χ²)<br/>tail-dependence estimators"]
    C --> C2["marginals<br/>risk-neutral lognormal terminal<br/>discount factor"]
    C --> C3["payoffs<br/>best_of_call · worst_of_put"]
    C --> C4["pricing<br/>MC engines (Cholesky / copula / Student-t)<br/>Black–Scholes analytic benchmark"]
    C --> C5["statistics<br/>price_se_ci · bs_agreement_check"]
    B --> D["Experiment records<br/>research/dashboard_experiments/*.jsonl<br/>JSON · CSV export"]
```

## 2. Data flow of a Pricing Laboratory run

```mermaid
flowchart LR
    U["UI inputs<br/>(S0, K, r, q, σ, T, copula, ρ, ν, N, seed)"] --> V["validate_config<br/>studio_core"]
    V --> E["pricing.monte_carlo_*<br/>(exact engine functions)"]
    E --> R["record: price, SE, 95% CI,<br/>N, runtime, paths/s, seed,<br/>engine call name"]
    R --> UI1["metrics + disclaimers"]
    R --> X["JSON / CSV download"]
    R --> H["💾 save to history"]
```

Nothing financial is computed in the UI layer: the app displays exactly what the
engine returns and decorates it with uncertainty language.

## 3. Data flow of the paired copula comparison (CRN)

```mermaid
flowchart TD
    same_seed["seed s"] --> GS["copulas.gaussian_copula_sample(ρ, N, s)"]
    same_seed --> TS["copulas.student_t_copula_sample(ρ, ν, N, s)<br/>(adds one shared χ² draw per observation)"]
    GS --> PG["marginals.risk_neutral_lognormal_terminal (asset 1, asset 2)"]
    TS --> PT["same marginal transform (identical parameters)"]
    PG --> HG["payoffs best_of_call / worst_of_put → e^{-rT}·H"]
    PT --> HT["payoffs best_of_call / worst_of_put → e^{-rT}·H"]
    HG & HT --> D["pathwise differences Δ = V_t − V_g"]
    D --> S["paired SE = sd(Δ, ddof=1)/√N<br/>CI via price_se_ci"]
```

Key property: both samplers consume `standard_normal((2, N))` first from
`default_rng(seed)`, so the latent Gaussian vector is shared; the χ² draw only
extends the Student-t stream. This produces genuinely paired samples and a paired
SE computed from payoff **differences** — never from two independent SEs.

## 4. Streaming convergence

Path counts above 100 000 are split into chunks of ≤ 100 000 paths with
deterministic sub-stream seeds (`base_seed + chunk·1000003`). Counter aggregates
(n, ΣV, ΣV²) combine the chunks, so memory stays bounded and the estimator equals
the composite stream exactly — verified by
`test_convergence_single_chunk_matches_standard_pipeline`.

## 5. Experiment history

JSON-lines file `research/dashboard_experiments/experiments.jsonl` (append-only,
never silently overwritten). Each record carries `experiment_id`, UTC timestamp,
studio version, copula, contract, every financial parameter, seed, result
metrics, runtime and `experiment_kind` (`single_pricing`,
`paired_copula_comparison`, `convergence`). Storage is non-durable on Streamlit
Community Cloud; export buttons exist on every experiment page.

## 6. Component boundaries (what must not change)

- All formulas live in `copuladeriv/python/copuladeriv/` (see
  `docs/studio_architecture_audit.md` §2 and §7).
- `studio_core.py` may only compose, validate, time and record engine calls.
- The C++20 engine stays an independent local tool (`copuladeriv/cpp/`); the app
  documents it but does not embed or fake it.
