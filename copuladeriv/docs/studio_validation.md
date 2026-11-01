# CopulaDeriv Studio — Validation Summary

## What was validated (Day 15)

### Automated tests
- **77 tests pass** locally (`python -m pytest copuladeriv/tests/ -q` from
  `copuladeriv/python`): the 51 pre-existing package tests, unmodified, plus
  **26 new Studio tests** (`copuladeriv/tests/test_studio.py`).
  Deprecation warnings are escalated to errors and pass.

### Engine parity (`test_studio.py::test_dashboard_matches_existing_engine_exactly`)
For all four combinations (Gaussian / Student-t × best-of call / worst-of put),
the Studio record's price, standard error and CI are **exactly equal** to the
existing engine functions' output for identical configuration: the dashboard
calls the same functions, it does not reimplement them.

### Paired common-random-number comparison
- `test_paired_difference_and_ci_are_statistically_consistent`: ΔV identity,
  paired SE ≠ naive independent-SE combination (they differ by construction),
  paired SE bounded by the single-model SEs, CI centred on ΔV.
- `test_same_seed_pairing_shares_latent_gaussian_draw`: both samplers consume an
  identical `standard_normal((2, N))` stream from the same seed — the property
  the pairing relies on — while rank identity across copulas is explicitly not
  assumed (the shared χ² draw reshapes Student-t scores per observation).

### Theoretical tail dependence
`test_theoretical_tail_dependence_matches_reference_formula`: the returned
λ values equal an independently written reference computation
`2·t_{ν+1}(−√((ν+1)(1−ρ)/(1+ρ)))` to 1e-10 relative tolerance for 16 (ν, ρ)
combinations; Gaussian λ = 0 for interior ρ; Student-t symmetric
(λ_U = λ_L).

### Confidence intervals
`test_confidence_interval_is_z_times_se_around_price`: CI = price ± 1.96·SE
exactly, matching `statistics.price_se_ci` behaviour.

### Input validation
Invalid financial parameters, out-of-range ρ, non-positive ν, zero path count,
path counts above the deployment cap and missing ν for the Student-t copula all
raise `ExperimentError`; the UI renders these as user-visible errors and never
substitutes a default price.

### Chunked convergence streaming
- `test_convergence_single_chunk_matches_standard_pipeline`: a one-chunk
  convergence run equals the standard pipeline exactly (same sub-stream seed).
- `test_convergence_multi_chunk_statistics_are_consistent`: two-chunk run yields
  a positive, consistent SE with the documented aggregate formula.

### Payoff-distribution analytics
`test_payoff_analytics_loss_convention`: quantiles are monotone, VaR ≤ CVaR
under the documented loss convention, P(zero payoff) ∈ [0, 1], record carries N
and seed.

### Experiment history
`test_experiment_history_roundtrip`: JSONL save/load round trip with typed
fields preserved.

### Application startup
`streamlit run app.py --server.headless true --server.port 8599` was started,
served **HTTP 200**, logged no errors or tracebacks, and was shut down.
`test_app_module_imports` additionally imports the app module in bare mode and
verifies the 10-section navigation.

### Not validated on the cloud platform
Deployment to Streamlit Community Cloud is *prepared* (requirements, config,
docs) but **not executed from this session** — per project rules, deployment is
only claimed when actually deployed and verified. Local C++ engine behaviour was
verified earlier (binary present); no C++ benchmark numbers are embedded in the
app, which measures Python throughput live instead.
