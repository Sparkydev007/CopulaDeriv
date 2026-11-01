# CopulaDeriv Studio — Implementation Plan

Ordered plan derived from the Day-15 specification and the repository audit
(`docs/studio_architecture_audit.md`). Each phase lists its dependencies and its
verification action. Status is recorded honestly at completion.

## Phase 0 — Repository audit (complete)
Items: package APIs, C++20 build status, tests, docs, deployment constraints.
Verification: source inspection + `pytest` (51 package tests passing at entry).
**Produced:** `docs/studio_architecture_audit.md`.

## Phase 1 — Controller layer and validation (complete)
Depends on: Phase 0.
`copuladeriv/studio_core.py` composes the validated engine into typed experiment
configurations (`validate_config`), paired common-random-number comparisons,
sensitivity sweeps, chunked streaming convergence, tail-dependence studies,
payoff-distribution analytics, and JSONL experiment persistence.
Verification: 26 controller tests (engine parity, paired statistics,
tail-dependence reference formula, CI arithmetic, validation matrix,
single-chunk equivalence, history round trip).

## Phase 2 — Application shell and integration (complete)
Depends on: Phase 1.
`copuladeriv/app.py` (Streamlit), `.streamlit/config.toml`, module-level cache
wrappers keyed on the full configuration (no stale results after input changes),
sidebar parameter panel shared by experiment sections.
Verification: HTTP 200 on `streamlit run app.py --server.headless`;
`test_app_module_imports`.

## Phase 3 — Research sections (complete)
Depends on: Phase 2.
1. Research Overview (mathematics + terminal-dependence framing disclaimer)
2. Pricing Laboratory — engine call-through + provenance display
3. Copula Comparison — paired CRN statistics with paired-SE language rules
4. Sensitivity Analysis — one-parameter sweeps, editable grids, CSV export
5. Tail Dependence — theoretical λ vs finite-threshold empirical estimates with
   exceedance counts and reliability warnings
6. Greeks & Risk — implementation status disclosed; implemented BS reference
   Greeks; VaR/CVaR with documented unhedged-loss convention
7. Monte Carlo Convergence — SE vs 1/√N reference, runtime, CI width
8. Performance Benchmarks — live-measured Python throughput; C++ as local
   workflow only; explicit throughput ≠ HFT-latency statement
9. Experiment History — load, compare, export, clear
10. Methodology and Assumptions
11. System Diagnostics — environment, dependency status, engine availability,
    logging ticker (no secrets, no machine paths)

## Phase 4 — Reliability, logging, observability (complete)
Depends on: Phase 3.
Standard-library logging in `studio_core` (experiment start/complete/failure —
parameters and errors, never credentials); typed exception categories
(`ExperimentError`); runtime-reliability notes surfaced in diagnostics; exports
restricted to Streamlit download widgets (no arbitrary file paths).

## Phase 5 — CI and repository hygiene (complete)
Depends on: Phase 2.
`.github/workflows/ci.yml`: Python 3.11/3.12 matrix → pip install → lint
(ruff) → pytest (from `copuladeriv/python`) → app import smoke test → C++ build
on ubuntu-latest when a C++ toolchain image is available (kept simple; C++ tests
do not exist yet, so the build step compiles only).
Verification: workflow syntax reviewed by hand; actual CI run happens on push.

## Phase 6 — Research integration and documentation (complete)
Depends on: Phase 3.
`research/` skeleton (`literature/`, `dashboard_experiments/`, `figures/`,
`tables/`, `exported_results/`, `manuscript/` – with README notes),
`docs/research_figure_guide.md`, `docs/studio_user_guide.md`,
`docs/studio_validation.md`, `docs/studio_architecture.md`,
`docs/streamlit_deployment.md`, README update.

## Deferred items (not in this release, tracked intentionally)
- Vectorized antithetic-variate and control-variate estimators *as engine
  functions* (Day-10 offline scripts remain the research record).
- Rainbow finite-difference Greeks (requires seeded CRN bump-and-reprice engine
  functions + bump-size sensitivity tests).
- Importance sampling (no validated likelihood-ratio implementation exists).
- C++ unit tests (GTest wiring exists in CMakeLists but no test sources).
- GitHub-hosted CI verification of the workflow (after the first push).
- Streamlit Community Cloud deployment verification (requires a human account
  action; instructions complete).
