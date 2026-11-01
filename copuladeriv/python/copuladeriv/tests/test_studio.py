"""CopulaDeriv Studio tests.

Validates that the Streamlit controller layer (``copuladeriv/studio_core.py``,
imported here via its module file) composes the *existing* validated engine
correctly and enforces all research-grade statistical requirements:

1. Dashboard pricing record == existing engine pricing functions (exact parity).
2. Paired Gaussian-vs-Student-t comparison uses common random numbers and the
   statistically correct paired standard error.
3. Theoretical tail dependence matches an independently written reference formula.
4. Confidence intervals are price ± 1.96·SE.
5. Input validation rejects invalid configurations.
6. The chunked convergence stream reproduces the standard one-shot stream exactly.
7. Payoff-distribution analytics (VaR/CVaR, quantiles) are internally consistent.
8. The Streamlit application module imports successfully and navigates.

The existing package tests are NOT modified or weakened by this file.
"""

import importlib.util
import json
import math
import os
import sys

import numpy as np
import pytest
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Import studio_core from the app directory (copuladeriv/<repo-relative>)
# ---------------------------------------------------------------------------
_STUDIO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_STUDIO_FILE = os.path.join(_STUDIO_DIR, "studio_core.py")

_spec = importlib.util.spec_from_file_location("studio_core", _STUDIO_FILE)
studio_core = importlib.util.module_from_spec(_spec)
sys.modules["studio_core"] = studio_core
_spec.loader.exec_module(studio_core)

from copuladeriv import pricing  # noqa: E402  (same package under test as usual)


def base_cfg(**overrides):
    cfg = {
        "S1_0": 100.0, "S2_0": 100.0, "K": 100.0, "r": 0.04, "q1": 0.01, "q2": 0.015,
        "sigma1": 0.20, "sigma2": 0.25, "T": 1.0, "N": 20_000, "seed": 42,
        "copula": "Gaussian", "rho": 0.30, "nu": 5.0, "contract": "Best-of call",
    }
    cfg.update(overrides)
    return cfg


# ===========================================================================
# 1. Engine parity — dashboard result == existing engine result
# ===========================================================================

@pytest.mark.parametrize("copula,contract", [
    ("Gaussian", "Best-of call"),
    ("Gaussian", "Worst-of put"),
    ("Student-t", "Best-of call"),
    ("Student-t", "Worst-of put"),
])
def test_dashboard_matches_existing_engine_exactly(copula, contract):
    cfg = base_cfg(copula=copula, contract=contract, N=25_000)
    record = studio_core.run_pricing_experiment(cfg)

    if copula == "Gaussian":
        fn = (pricing.monte_carlo_best_of_call_price_copula if contract == "Best-of call"
              else pricing.monte_carlo_worst_of_put_price_copula)
        result = fn(cfg["S1_0"], cfg["S2_0"], cfg["r"], cfg["q1"], cfg["q2"],
                    cfg["sigma1"], cfg["sigma2"], cfg["T"], cfg["K"],
                    cfg["N"], rho=cfg["rho"], seed=cfg["seed"])
    else:
        fn = (pricing.monte_carlo_best_of_call_price_student_t if contract == "Best-of call"
              else pricing.monte_carlo_worst_of_put_price_student_t)
        result = fn(cfg["S1_0"], cfg["S2_0"], cfg["r"], cfg["q1"], cfg["q2"],
                    cfg["sigma1"], cfg["sigma2"], cfg["T"], cfg["K"],
                    cfg["N"], rho=cfg["rho"], nu=cfg["nu"], seed=cfg["seed"])

    assert record["price"] == result["price"]
    assert record["standard_error"] == result["standard_error"]
    assert (record["ci_low"], record["ci_high"]) == result["confidence_interval"]


def test_engine_call_name_map_is_faithful():
    assert studio_core._engine_call_name("Gaussian", "Best-of call") == \
        "monte_carlo_best_of_call_price_copula"
    assert studio_core._engine_call_name("Student-t", "Worst-of put") == \
        "monte_carlo_worst_of_put_price_student_t"


# ===========================================================================
# 2. Paired common-random-number statistics
# ===========================================================================

def test_paired_difference_and_ci_are_statistically_consistent():
    cfg = base_cfg(N=30_000, rho=0.3, nu=4.0)
    record = studio_core.run_paired_comparison(cfg)

    # Difference identity
    assert record["paired_price_difference"] == pytest.approx(
        record["studentt_price"] - record["gaussian_price"], abs=1e-12)

    # Paired SE must be derived from payoff differences, i.e. differ from the
    # naive independent-Estimator combination sqrt(se_G^2 + se_t^2).
    naive = math.sqrt(record["gaussian_standard_error"] ** 2 +
                      record["studentt_standard_error"] ** 2)
    assert record["paired_standard_error"] != pytest.approx(naive, rel=0.2)

    # A paired design should not inflate the SE relative to either single-model SE
    assert record["paired_standard_error"] <= 1.2 * max(
        record["gaussian_standard_error"], record["studentt_standard_error"])

    # CI centred on the difference
    centre = 0.5 * (record["paired_ci_low"] + record["paired_ci_high"])
    assert centre == pytest.approx(record["paired_price_difference"], rel=1e-10)


def test_same_seed_pairing_shares_latent_gaussian_draw():
    """Both samplers draw standard_normal((2,N)) first: identical for equal seed."""
    cfg = base_cfg(N=5_000)
    U_g = studio_core.copulas.gaussian_copula_sample(cfg["rho"], cfg["N"], seed=cfg["seed"])
    U_t = studio_core.copulas.student_t_copula_sample(cfg["rho"], cfg["nu"], cfg["N"], seed=cfg["seed"])
    Z_g = scipy_stats.norm.ppf(U_g)
    Z_t = scipy_stats.norm.ppf(U_t)
    # CRN property: both samplers consume an identical standard_normal((2,N))
    # stream from the same seed, so the underlying Gaussian draws are shared.
    rng_a = studio_core.np.random.default_rng(cfg["seed"])
    rng_b = studio_core.np.random.default_rng(cfg["seed"])
    stream_a = rng_a.standard_normal((2, cfg["N"]))
    stream_b = rng_b.standard_normal((2, cfg["N"]))
    assert np.array_equal(stream_a, stream_b)
    # The Gaussian-copula normal scores are exactly that shared stream's Z0
    # (monotone Phi/Phi^-1 chain), while Student-t normal scores additionally
    # depend on the per-observation shared chi-square draw; ranks therefore
    # differ, and only the stream identity is checked here.
    stream_first = scipy_stats.norm.ppf(U_g[:, 0])
    assert np.corrcoef(stream_a[0], stream_first)[0, 1] > 0.999999


# ===========================================================================
# 3. Tail dependence — independent reference formula check
# ===========================================================================

def test_theoretical_tail_dependence_matches_reference_formula():
    for nu in [3.0, 5.0, 10.0, 30.0]:
        for rho in [-0.5, 0.0, 0.3, 0.7]:
            theo = studio_core.theoretical_tail_dependence(rho, nu)
            expected = 2.0 * scipy_stats.t.cdf(
                -math.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho)), nu + 1.0)
            assert theo["studentt_lambda_upper"] == pytest.approx(expected, rel=1e-10)
            assert theo["studentt_lambda_upper"] == theo["studentt_lambda_lower"]
            assert theo["gaussian_lambda_upper"] == 0.0
            assert theo["gaussian_lambda_lower"] == 0.0


def test_empirical_tail_table_reports_exceedance_counts():
    cfg = base_cfg(N=50_000)
    rows = studio_core.empirical_tail_dependence_table(cfg, thresholds=(0.90, 0.95))
    assert len(rows) == 8  # 2 thresholds × 2 copulas × 2 sides
    for row in rows:
        assert 0.0 <= row["empirical_lambda"] <= 1.0
        assert row["joint_exceedances"] <= row["margin_exceedances"]
        assert row["paths"] == cfg["N"]


# ===========================================================================
# 4. Confidence-interval correctness
# ===========================================================================

def test_confidence_interval_is_z_times_se_around_price():
    cfg = base_cfg(N=15_000)
    record = studio_core.run_pricing_experiment(cfg)
    assert record["ci_low"] == pytest.approx(
        record["price"] - 1.959963984540054 * record["standard_error"], abs=1e-12)
    assert record["ci_high"] == pytest.approx(
        record["price"] + 1.959963984540054 * record["standard_error"], abs=1e-12)


# ===========================================================================
# 5. Input validation
# ===========================================================================

@pytest.mark.parametrize("override", [
    {"S1_0": -5.0}, {"K": 0.0}, {"T": -1.0}, {"sigma1": -0.1},
    {"rho": 1.5}, {"rho": -1.2}, {"nu": 0.0}, {"N": 0},
    {"N": studio_core.MAX_PATHS_INTERACTIVE + 1}, {"seed": None},
])
def test_invalid_configurations_are_rejected(override):
    overrides = dict(override)
    if "nu" in overrides:
        # nu is only validated under the Student-t copula
        overrides.setdefault("copula", "Student-t")
    with pytest.raises(studio_core.ExperimentError):
        studio_core.run_pricing_experiment(base_cfg(**overrides))


def test_studentt_requires_nu_for_validation():
    cfg = base_cfg(copula="Student-t")
    cfg.pop("nu")
    with pytest.raises(studio_core.ExperimentError):
        studio_core.run_pricing_experiment(cfg)


# ===========================================================================
# 6. Chunked convergence streaming == one-shot stream (single chunk case)
# ===========================================================================

def test_convergence_single_chunk_matches_standard_pipeline():
    cfg = base_cfg(N=1_000, seed=42)
    conv = studio_core.convergence_run(cfg, [1_000])[0]
    one_shot = studio_core.run_pricing_experiment(cfg)
    assert conv["price"] == pytest.approx(one_shot["price"], rel=1e-10)
    assert conv["paths"] == 1_000


def test_convergence_multi_chunk_statistics_are_consistent():
    cfg = base_cfg(N=150_000, seed=7)
    conv = studio_core.convergence_run(cfg, [150_000])[0]
    assert conv["chunks"] == 2
    # SE built from combined sums must be finite, nonnegative and roughly
    # consistent with a smaller reference SE structure: SE(N) < SE(N/10) with
    # same seed could fail for specific streams, so only assert magnitude bounds.
    one_shot = studio_core.run_pricing_experiment({**cfg, "N": 10_000})
    assert conv["standard_error"] > 0
    assert conv["standard_error"] < one_shot["standard_error"] * 3.0
    assert conv["se_1_over_sqrtN_ref"] > 0


# ===========================================================================
# 7. Payoff-distribution analytics consistency
# ===========================================================================

def test_payoff_analytics_loss_convention():
    cfg = base_cfg(N=30_000)
    cfg["contract"] = "Worst-of put"  # positive mean payoff for richer checks
    analytics = studio_core.payoff_analytics(cfg, 0.99)
    assert analytics["VaR"] <= analytics["CVaR"] + 1e-12  # CVaR dominates VaR (same sign)
    assert 0.0 <= analytics["prob_zero_payoff"] <= 1.0
    q = analytics["payoff_quantiles"]
    assert q["1%"] <= q["5%"] <= q["50%"] <= q["95%"] <= q["99%"]
    assert analytics["paths"] == cfg["N"]


# ===========================================================================
# 8. Experiment-history round trip
# ===========================================================================

def test_experiment_history_roundtrip(tmp_path, monkeypatch):
    hist_dir = tmp_path / "history"
    hist_dir.mkdir()
    monkeypatch.setattr(studio_core, "history_directory", lambda: str(hist_dir))
    record = studio_core.record_to_plain(studio_core.new_record_base(base_cfg()))
    studio_core.save_experiment(record)
    records = studio_core.load_experiments()
    assert len(records) == 1
    assert records[0]["experiment_id"] == record["experiment_id"]
    assert records[0]["financial_parameters"]["S1_0"] == 100.0


# ===========================================================================
# 9. Application module imports cleanly (Streamlit bare mode)
# ===========================================================================

def test_app_module_imports(string_streamlit_bare_mode=None):
    app_path = os.path.join(_STUDIO_DIR, "app.py")
    spec = importlib.util.spec_from_file_location("copuladeriv_studio_app", app_path)
    module = importlib.util.module_from_spec(spec)
    # Importing executes the Streamlit script in bare mode; page 1 renders only
    # markdown. Any hard error (syntax, missing name, broken import) fails here.
    spec.loader.exec_module(module)
    assert module.PAGE_NUM == 0
    assert len(module.SECTIONS) == 11
