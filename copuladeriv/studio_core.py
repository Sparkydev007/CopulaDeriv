"""Experiment controller layer for CopulaDeriv Studio.

This module is the thin bridge between the Streamlit application (``app.py``)
and the existing validated ``copuladeriv`` package. It intentionally contains
**no financial formulas of its own**:

- copula sampling        -> ``copuladeriv.copulas``
- terminal marginals     -> ``copuladeriv.marginals``
- payoffs                -> ``copuladeriv.payoffs``
- standard pricing calls -> ``copuladeriv.pricing``
- statistics             -> ``copuladeriv.statistics`` / ``marginals.discount_factor``

It only *composes* those validated components into interactive experiments
(paired comparisons, sensitivity sweeps, convergence runs, tail-dependence
estimates, payoff-distribution analytics) and into reproducible experiment
records. Anything it computes is labelled with its seed, path count and runtime
so every displayed number traces back to an actual computation.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import sys
import time
import uuid

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger("copuladeriv.studio")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Path bootstrap: make `copuladeriv` importable from the app directory
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.join(_HERE, "python")
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from copuladeriv import copulas  # noqa: E402
from copuladeriv import payoffs as payoff_modules  # noqa: E402
from copuladeriv import pricing  # noqa: E402
from copuladeriv import statistics as pricing_statistics  # noqa: E402
from copuladeriv.marginals import discount_factor, risk_neutral_lognormal_terminal  # noqa: E402

APP_VERSION = "1.0.0"

OPTION_CONTRACTS = {
    "Best-of call": payoff_modules.best_of_call,
    "Worst-of put": payoff_modules.worst_of_put,
}

COPULAS = ("Gaussian", "Student-t")

# Conservative ceiling for interactive runs (Streamlit Community Cloud free tier)
MAX_PATHS_INTERACTIVE = 500_000
CHUNK_SIZE = 100_000


class ExperimentError(RuntimeError):
    """Raised when an experiment cannot be executed (bad inputs or compute failure)."""


def _validation_error(msg: str):
    """Log a validation failure and raise the typed exception."""
    logger.warning("Invalid configuration: %s", msg)
    raise ExperimentError(msg)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_config(cfg: dict) -> dict:
    """Validate an experiment configuration; raise ExperimentError on bad input."""
    for key in ("S1_0", "S2_0", "K", "T"):
        value = float(cfg[key])
        if not np.isfinite(value) or value <= 0:
            _validation_error(f"{key} must be a positive finite number (got {cfg[key]}).")

    for key in ("sigma1", "sigma2"):
        value = float(cfg[key])
        if not np.isfinite(value) or value < 0:
            _validation_error(f"{key} must be non-negative and finite (got {cfg[key]}).")

    for key in ("r", "q1", "q2"):
        value = float(cfg[key])
        if not np.isfinite(value):
            _validation_error(f"{key} must be a finite number (got {cfg[key]}).")

    rho = float(cfg["rho"])
    if not (-1.0 <= rho <= 1.0):
        _validation_error(f"rho must lie in [-1, 1] (got {rho}).")

    if cfg.get("copula") == "Student-t":
        if cfg.get("nu") is None:
            _validation_error("The Student-t copula requires a degrees-of-freedom value (nu).")
        nu = float(cfg["nu"])
        if not (np.isfinite(nu) and nu > 0):
            _validation_error(f"nu must be positive and finite (got {cfg.get('nu')}).")
        if nu <= 0.05:
            _validation_error("nu below 0.05 is numerically unreliable; increase nu.")

    N = int(cfg["N"])
    if N <= 0:
        _validation_error("Path count N must be positive.")
    if N > MAX_PATHS_INTERACTIVE:
        _validation_error(
            f"Path count N = {N:,} exceeds the deployment limit of {MAX_PATHS_INTERACTIVE:,}."
        )

    seed = cfg.get("seed")
    if seed is None or not np.isfinite(float(seed)):
        _validation_error("A finite integer seed is required for reproducibility.")

    return cfg


# ---------------------------------------------------------------------------
# Deterministic pricing call into the existing engine
# ---------------------------------------------------------------------------

def run_pricing_experiment(cfg: dict) -> dict:
    """Execute one pricing experiment using the existing pricing engine ONLY.

    Returns a result record with price / standard error / 95% CI / runtime.
    """
    logger.info("Experiment start: %s %s / %s N=%s seed=%s",
                cfg.get("copula"), cfg.get("contract"), cfg.get("rho"),
                cfg.get("N"), cfg.get("seed"))
    validate_config(cfg)
    contract = cfg["contract"]
    copula = cfg["copula"]
    N, seed = int(cfg["N"]), int(cfg["seed"])
    rho = float(cfg["rho"])

    t0 = time.perf_counter()
    if copula == "Gaussian":
        if contract == "Best-of call":
            result = pricing.monte_carlo_best_of_call_price_copula(
                cfg["S1_0"], cfg["S2_0"], cfg["r"], cfg["q1"], cfg["q2"],
                cfg["sigma1"], cfg["sigma2"], cfg["T"], cfg["K"], N, rho=rho, seed=seed)
        else:
            result = pricing.monte_carlo_worst_of_put_price_copula(
                cfg["S1_0"], cfg["S2_0"], cfg["r"], cfg["q1"], cfg["q2"],
                cfg["sigma1"], cfg["sigma2"], cfg["T"], cfg["K"], N, rho=rho, seed=seed)
    else:  # Student-t
        if contract == "Best-of call":
            result = pricing.monte_carlo_best_of_call_price_student_t(
                cfg["S1_0"], cfg["S2_0"], cfg["r"], cfg["q1"], cfg["q2"],
                cfg["sigma1"], cfg["sigma2"], cfg["T"], cfg["K"], N, rho=rho,
                nu=cfg["nu"], seed=seed)
        else:
            result = pricing.monte_carlo_worst_of_put_price_student_t(
                cfg["S1_0"], cfg["S2_0"], cfg["r"], cfg["q1"], cfg["q2"],
                cfg["sigma1"], cfg["sigma2"], cfg["T"], cfg["K"], N, rho=rho,
                nu=cfg["nu"], seed=seed)
    runtime = time.perf_counter() - t0

    price, se = result["price"], result["standard_error"]
    if not (np.isfinite(price) and np.isfinite(se)):
        logger.error("Experiment failed: engine returned non-finite estimate "
                     "(price=%r se=%r)", price, se)
        raise ExperimentError("Engine returned a non-finite estimate; check parameters.")
    logger.info("Experiment complete: price=%.6f se=%.6f runtime=%.3fs",
                price, se, runtime)

    record = new_record_base(cfg)
    record.update({
        "engine_call": _engine_call_name(copula, contract),
        "price": price,
        "standard_error": se,
        "ci_low": result["confidence_interval"][0],
        "ci_high": result["confidence_interval"][1],
        "paths": N,
        "runtime_s": runtime,
        "paths_per_second": N / runtime if runtime > 0 else None,
    })
    return record


def _engine_call_name(copula: str, contract: str) -> str:
    base = ("monte_carlo_best_of_call_price" if contract == "Best-of call"
            else "monte_carlo_worst_of_put_price")
    if copula == "Gaussian":
        base += "_copula"
    else:
        base += "_student_t"
    return base


# ---------------------------------------------------------------------------
# Composition of validated building blocks (used when payoff arrays are needed)
# ---------------------------------------------------------------------------

def _terminal_prices_from_uniforms(U: np.ndarray, cfg: dict):
    """Transform copula uniforms into terminal prices using the package marginals.

    Z_i = Phi^{-1}(U_i) then the risk-neutral lognormal terminal formula from
    ``copuladeriv.marginals.risk_neutral_lognormal_terminal``.
    """
    Z1 = scipy_stats.norm.ppf(U[:, 0])
    Z2 = scipy_stats.norm.ppf(U[:, 1])
    S1_T = risk_neutral_lognormal_terminal(cfg["S1_0"], cfg["r"], cfg["q1"], cfg["sigma1"], cfg["T"], Z1)
    S2_T = risk_neutral_lognormal_terminal(cfg["S2_0"], cfg["r"], cfg["q2"], cfg["sigma2"], cfg["T"], Z2)
    return S1_T, S2_T


def _discounted_payoffs_from_uniforms(U: np.ndarray, cfg: dict) -> np.ndarray:
    payoff_fn = OPTION_CONTRACTS[cfg["contract"]]
    S1_T, S2_T = _terminal_prices_from_uniforms(U, cfg)
    H = payoff_fn(S1_T, S2_T, cfg["K"])
    return discount_factor(cfg["r"], cfg["T"]) * H


# ---------------------------------------------------------------------------
# Paired common-random-number comparison (Gaussian vs Student-t)
# ---------------------------------------------------------------------------

def run_paired_comparison(cfg: dict) -> dict:
    """Compare Gaussian and Student-t copulas under identical marginals.

    Common random numbers: both samplers draw ``standard_normal((2, N))`` first,
    so with the same seed the latent Gaussian vector is shared between the two
    models; the Student-t sampler adds a chi-square draw afterwards. Payoffs are
    computed path-by-path and the paired difference is analysed directly.

    The paired SE is the sample sd (ddof=1) of the N discounted payoff
    *differences* divided by sqrt(N) — never a combination of two independent
    standard errors.
    """
    validate_config(cfg)
    N, seed = int(cfg["N"]), int(cfg["seed"])
    t0 = time.perf_counter()

    U_g = copulas.gaussian_copula_sample(cfg["rho"], N, seed=seed)
    U_t = copulas.student_t_copula_sample(cfg["rho"], cfg["nu"], N, seed=seed)

    V_g = _discounted_payoffs_from_uniforms(U_g, cfg)
    V_t = _discounted_payoffs_from_uniforms(U_t, cfg)
    runtime = time.perf_counter() - t0

    diff = V_t - V_g
    price_g = float(np.mean(V_g))
    price_t = float(np.mean(V_t))
    se_g = float(np.std(V_g, ddof=1) / np.sqrt(N))
    se_t = float(np.std(V_t, ddof=1) / np.sqrt(N))
    paired_se = float(np.std(diff, ddof=1) / np.sqrt(N))
    paired_price = price_t - price_g
    ci = pricing_statistics.price_se_ci(paired_price, paired_se, 0.95)

    record = new_record_base(cfg)
    record.update({
        "experiment_kind": "paired_copula_comparison",
        "paths": N,
        "paired_price_difference": paired_price,
        "paired_standard_error": paired_se,
        "paired_ci_low": ci[0],
        "paired_ci_high": ci[1],
        "gaussian_price": price_g,
        "gaussian_standard_error": se_g,
        "gaussian_ci_low": price_g - 1.959963984540054 * se_g,
        "gaussian_ci_high": price_g + 1.959963984540054 * se_g,
        "studentt_price": price_t,
        "studentt_standard_error": se_t,
        "studentt_ci_low": price_t - 1.959963984540054 * se_t,
        "studentt_ci_high": price_t + 1.959963984540054 * se_t,
        "gaussian_payoff_SD": float(np.std(V_g, ddof=1)),
        "studentt_payoff_SD": float(np.std(V_t, ddof=1)),
        "runtime_s": runtime,
        "paths_per_second": N / runtime if runtime > 0 else None,
        "p_within_2se": int(abs(paired_price) <= 2.0 * paired_se),
    })
    return record


# ---------------------------------------------------------------------------
# Sensitivity sweeps (fixed parameters except the swept one)
# ---------------------------------------------------------------------------

def run_sweep(base_cfg: dict, parameter: str, values: list, paired: bool = False) -> list:
    """Sweep one parameter over a grid, keeping every other parameter fixed.
    Nuisance/detail: same seed pinned for each grid point to allow comparison."""
    records = []
    for value in values:
        cfg = dict(base_cfg)
        if parameter == "nu":
            cfg["copula"] = "Student-t"
        cfg[parameter] = value
        try:
            if paired and parameter == "rho":
                records.append(run_paired_comparison(cfg))
            else:
                records.append(run_pricing_experiment(cfg))
        except ExperimentError:
            raise
    return records


# ---------------------------------------------------------------------------
# Payoff-distribution / risk analytics (documented, unhedged contract convention)
# ---------------------------------------------------------------------------

def payoff_analytics(cfg: dict, alpha: float = 0.99) -> dict:
    """Statistics of the discounted payoff distribution of a single contract.

    Loss convention: the contract is held unhedged to maturity and valued at
    t=0 with price V0. Define the discounted loss as
        loss = e^{rT} * V0 - e^{-rT} * H_T
    i.e. the realised discounted payoff is compared with the t=0 fair value.

    This describes the risk of the **terminal payoff distribution of an unhedged
    option**, not the market risk of an actively hedged trading book.
    """
    validate_config(cfg)
    N, seed = int(cfg["N"]), int(cfg["seed"])
    if cfg["copula"] == "Gaussian":
        U = copulas.gaussian_copula_sample(cfg["rho"], N, seed=seed)
    else:
        U = copulas.student_t_copula_sample(cfg["rho"], cfg["nu"], N, seed=seed)
    V = _discounted_payoffs_from_uniforms(U, cfg)
    V0 = float(np.mean(V))
    loss = float(discount_factor(cfg["r"], cfg["T"])) * V0 - V

    var_level = float(np.quantile(loss, alpha))
    tail = loss[loss >= var_level]
    cvar = float(np.mean(tail)) if tail.size else var_level

    return {
        "discounted_payoff_mean": V0,
        "discounted_payoff_SD": float(np.std(V, ddof=1)),
        "payoff_quantiles": {
            "1%": float(np.quantile(V, 0.01)),
            "5%": float(np.quantile(V, 0.05)),
            "50%": float(np.quantile(V, 0.50)),
            "95%": float(np.quantile(V, 0.95)),
            "99%": float(np.quantile(V, 0.99)),
        },
        "prob_zero_payoff": float(np.mean(V == 0.0)),
        "VaR_alpha": alpha,
        "VaR": var_level,
        "CVaR": cvar,
        "paths": N,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Monte Carlo convergence (memory-efficient chunked streaming)
# ---------------------------------------------------------------------------

def convergence_run(cfg: dict, path_counts: list) -> list:
    """Run pricing at increasing path counts with a memory-efficient estimator.

    For each requested N, the simulation is performed in chunks of at most
    CHUNK_SIZE paths. Each chunk is a deterministic sub-stream with seed
    ``base_seed + chunk_index * 1000003``; counter aggregates (n, sum of
    discounted payoffs, sum of squared discounted payoffs) are combined across
    chunks, so arrays never exceed CHUNK_SIZE in memory. Estimates at different
    N are independent Monte Carlo runs from different sub-stream compositions;
    they do not need to converge monotonically.
    """
    validate_config(cfg)
    records = []
    for N in path_counts:
        t0 = time.perf_counter()
        n_total, s1, s2 = 0, 0.0, 0.0
        n_chunks = int(np.ceil(N / CHUNK_SIZE))
        remaining = N
        for chunk_index in range(n_chunks):
            m = min(CHUNK_SIZE, remaining)
            blend_seed = int(cfg["seed"]) + chunk_index * 1_000_003
            if cfg["copula"] == "Gaussian":
                U = copulas.gaussian_copula_sample(cfg["rho"], m, seed=blend_seed)
            else:
                U = copulas.student_t_copula_sample(cfg["rho"], cfg["nu"], m, seed=blend_seed)
            V = _discounted_payoffs_from_uniforms(U, cfg)
            n_total += m
            s1 += float(np.sum(V))
            s2 += float(np.sum(V ** 2))
            remaining -= m
        runtime = time.perf_counter() - t0

        mean = s1 / n_total
        # Sample variance from sums (with n-1 correction)
        var = max((s2 - n_total * mean ** 2) / (n_total - 1), 0.0)
        se = float(np.sqrt(var) / np.sqrt(n_total))
        ci = pricing_statistics.price_se_ci(mean, se, 0.95)

        record = new_record_base(cfg)
        record.update({
            "experiment_kind": "convergence",
            "paths": n_total,
            "price": float(mean),
            "standard_error": se,
            "ci_low": ci[0],
            "ci_high": ci[1],
            "se_1_over_sqrtN_ref": se * np.sqrt(N),  # SE * sqrt(N) proxy for the 1/sqrt(N) trend
            "runtime_s": runtime,
            "paths_per_second": n_total / runtime if runtime > 0 else None,
            "chunks": n_chunks,
        })
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Tail dependence
# ---------------------------------------------------------------------------

def theoretical_tail_dependence(rho: float, nu: float) -> dict:
    lam_u_t, lam_l_t = copulas.student_t_tail_dependence_theoretical(nu, rho)
    lam_u_g, lam_l_g = copulas.gaussian_tail_dependence(nu, rho)
    return {
        "gaussian_lambda_upper": lam_u_g,
        "gaussian_lambda_lower": lam_l_g,
        "studentt_lambda_upper": lam_u_t,
        "studentt_lambda_lower": lam_l_t,
    }


def empirical_tail_dependence_table(cfg: dict, thresholds=(0.90, 0.95, 0.975, 0.99)) -> list:
    """Finite-threshold empirical tail-dependence estimates plus exceedance counts.

    Gaussian copula with the same seed/marginal parameters is compared against
    the Student-t copula using regenerated copula samples (not asset prices).
    """
    validate_config(cfg)
    N, seed = int(cfg["N"]), int(cfg["seed"])
    lower_thresholds = [1.0 - u for u in thresholds]

    U_g = copulas.gaussian_copula_sample(cfg["rho"], N, seed=seed)
    U_t = copulas.student_t_copula_sample(cfg["rho"], cfg["nu"], N, seed=seed)

    rows = []
    for u in thresholds:
        for label, U in (("Gaussian", U_g), ("Student-t", U_t)):
            lam_u = copulas.empirical_tail_dependence_upper(U, u)
            exceed_u = int(np.sum((U[:, 0] > u) & (U[:, 1] > u)))
            rows.append({
                "copula": label,
                "threshold": u,
                "side": "upper",
                "empirical_lambda": lam_u,
                "joint_exceedances": exceed_u,
                "margin_exceedances": int(np.sum(U[:, 0] > u)),
                "paths": N,
                "seed": seed,
            })
    for u in lower_thresholds:
        for label, U in (("Gaussian", U_g), ("Student-t", U_t)):
            lam_l = copulas.empirical_tail_dependence_low(U, u)
            joint = int(np.sum((U[:, 0] < u) & (U[:, 1] < u)))
            rows.append({
                "copula": label,
                "threshold": 1.0 - u,
                "side": "lower",
                "empirical_lambda": lam_l,
                "joint_exceedances": joint,
                "margin_exceedances": int(np.sum(U[:, 0] < u)),
                "paths": N,
                "seed": seed,
            })
    return rows


# ---------------------------------------------------------------------------
# Experiment history (JSON records on local disk; non-durable on Streamlit Cloud)
# ---------------------------------------------------------------------------

def history_directory() -> str:
    path = os.path.join(_HERE, "research", "dashboard_experiments")
    os.makedirs(path, exist_ok=True)
    return path


HISTORY_FILE = "experiments.jsonl"


def new_record_base(cfg: dict) -> dict:
    return {
        "experiment_id": f"exp_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "studio_version": APP_VERSION,
        "implementation": "python",
        "copula": cfg["copula"],
        "contract": cfg["contract"],
        "rho": float(cfg["rho"]),
        "nu": cfg.get("nu"),
        "seed": int(cfg["seed"]),
        "financial_parameters": {
            "S1_0": cfg["S1_0"], "S2_0": cfg["S2_0"], "K": cfg["K"], "r": cfg["r"],
            "q1": cfg["q1"], "q2": cfg["q2"], "sigma1": cfg["sigma1"],
            "sigma2": cfg["sigma2"], "T": cfg["T"],
        },
        "experiment_kind": "single_pricing",
    }


def save_experiment(record: dict) -> str:
    path = os.path.join(history_directory(), HISTORY_FILE)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=_json_default) + "\n")
    return path


def load_experiments() -> list:
    path = os.path.join(history_directory(), HISTORY_FILE)
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r.get("timestamp_utc", ""), reverse=True)
    return records


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def record_to_plain(record: dict) -> dict:
    return json.loads(json.dumps(record, default=_json_default))


# ---------------------------------------------------------------------------
# Utility lookups used by the UI
# ---------------------------------------------------------------------------

def snd_estimate(rho: float) -> float:
    """Implementation detail quick lookup helper (kept for UI tooltips)."""
    return float(np.clip(rho, -1.0, 1.0))
