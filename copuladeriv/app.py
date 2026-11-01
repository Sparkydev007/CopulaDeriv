"""CopulaDeriv Studio — Dependence-Aware Monte Carlo Pricing of Multi-Asset Rainbow Options.

A research-grade Streamlit interface to the validated CopulaDeriv pricing engine.
All financial formulas live in ``copuladeriv/python/copuladeriv``; this application
is a thin experiment-control, analysis and visualization layer (see
``copuladeriv/docs/studio_architecture_audit.md``).

Run locally from this directory:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import platform
import sys

import numpy as np
import pandas as pd
# plotly.express unused (only graph_objects used) — kept for parity, linted out
import plotly.graph_objects as go
import streamlit as st

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

import studio_core  # noqa: E402

st.set_page_config(
    page_title="CopulaDeriv Studio",
    page_icon="∏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULTS = {
    "S1_0": 100.0, "S2_0": 100.0, "K": 100.0, "r": 0.04,
    "q1": 0.01, "q2": 0.015, "sigma1": 0.20, "sigma2": 0.25, "T": 1.0,
    "N": 100_000, "seed": 42, "copula": "Gaussian", "rho": 0.30, "nu": 5.0,
    "contract": "Best-of call",
}

SECTIONS = [
    "1. Research Overview",
    "2. Pricing Laboratory",
    "3. Copula Comparison",
    "4. Sensitivity Analysis",
    "5. Tail Dependence",
    "6. Greeks and Risk",
    "7. Monte Carlo Convergence",
    "8. Performance Benchmarks",
    "9. Experiment History",
    "10. Methodology and Assumptions",
    "11. System Diagnostics",
]


# ---------------------------------------------------------------------------
# Cached wrappers (engine calls are deterministic in (cfg, seed, copula, ...))
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Running Monte Carlo experiment …")
def cached_pricing(cfg_json: str) -> dict:
    return studio_core.run_pricing_experiment(json.loads(cfg_json))


@st.cache_data(show_spinner="Running paired copula comparison …")
def cached_paired(cfg_json: str) -> dict:
    return studio_core.run_paired_comparison(json.loads(cfg_json))


@st.cache_data(show_spinner="Running parameter sweep …")
def cached_sweep(cfg_json: str, parameter: str, values_json: str, paired: bool) -> list:
    return studio_core.run_sweep(json.loads(cfg_json), parameter,
                                 json.loads(values_json), paired)


@st.cache_data(show_spinner="Running convergence study …")
def cached_convergence(cfg_json: str, path_counts_json: str) -> list:
    return studio_core.convergence_run(json.loads(cfg_json), json.loads(path_counts_json))


def params_as_cfg(**overrides) -> dict:
    cfg = {**st.session_state["p"], **overrides}
    return {k: cfg[k] for k in
            ("S1_0", "S2_0", "K", "r", "q1", "q2", "sigma1", "sigma2", "T",
             "N", "seed", "copula", "rho", "nu", "contract")}


# ---------------------------------------------------------------------------
# Sidebar: parameter panel shared by the experiment pages
# ---------------------------------------------------------------------------

st.sidebar.title("CopulaDeriv Studio")
st.sidebar.caption("Dependence-aware Monte Carlo pricing of multi-asset rainbow options")
page = st.sidebar.radio("Navigation", SECTIONS)

if "p" not in st.session_state:
    st.session_state["p"] = dict(DEFAULTS)

p = st.session_state["p"]

show_params = page in (SECTIONS[1], SECTIONS[2], SECTIONS[3], SECTIONS[4], SECTIONS[6], SECTIONS[5])
if show_params:
    with st.sidebar.expander("Financial parameters", expanded=True):
        cols = st.columns(2)
        p["S1_0"] = cols[0].number_input("S₁(0)", min_value=0.01, value=float(p["S1_0"]), format="%.2f")
        p["S2_0"] = cols[1].number_input("S₂(0)", min_value=0.01, value=float(p["S2_0"]), format="%.2f")
        p["K"] = st.number_input("Strike K", min_value=0.01, value=float(p["K"]), format="%.2f")
        p["r"] = st.number_input("Risk-free rate r", min_value=-0.10, max_value=1.0, value=float(p["r"]), format="%.4f")
        p["q1"] = st.number_input("Dividend yield q₁", min_value=-0.10, max_value=1.0, value=float(p["q1"]), format="%.4f")
        p["q2"] = st.number_input("Dividend yield q₂", min_value=-0.10, max_value=1.0, value=float(p["q2"]), format="%.4f")
        p["sigma1"] = st.number_input("Volatility σ₁", min_value=0.0, max_value=5.0, value=float(p["sigma1"]), format="%.4f")
        p["sigma2"] = st.number_input("Volatility σ₂", min_value=0.0, max_value=5.0, value=float(p["sigma2"]), format="%.4f")
        p["T"] = st.number_input("Maturity T (years)", min_value=0.01, max_value=50.0, value=float(p["T"]), format="%.2f")

    with st.sidebar.expander("Simulation parameters", expanded=True):
        p["contract"] = st.radio("Option type", ("Best-of call", "Worst-of put"),
                                 horizontal=True)
        p["copula"] = st.radio("Copula", ("Gaussian", "Student-t"), horizontal=True)
        p["rho"] = st.slider("Latent correlation ρ", -1.00, 1.00, float(p["rho"]), 0.05,
                             help="Gaussian-copula / Student-t copula latent parameter. "
                                  "This is NOT the Pearson correlation of the terminal asset prices.")
        if p["copula"] == "Student-t":
            p["nu"] = st.slider("Degrees of freedom ν", 2.0, 50.0, float(p["nu"]), 0.5,
                                help="Shared chi-square scale uses ONE χ²(ν) draw per observation. "
                                     "Larger ν → Student-t copula approaches the Gaussian copula.")
        p["N"] = st.select_slider("Monte Carlo paths N",
                                  options=[1_000, 5_000, 10_000, 50_000, 100_000, 200_000, 500_000],
                                  value=int(p["N"]))
        p["seed"] = st.number_input("Random seed", min_value=0, value=int(p["seed"]), step=1)

    st.sidebar.caption(
        "Marginals stay risk-neutral lognormal with the parameters above while the "
        "copula changes dependence only. Terminal-dependence specification: the copula "
        "couples the *terminal* marginals at T, not a dynamically consistent diffusion."
    )


# ---------------------------------------------------------------------------
# Shared small UI helpers
# ---------------------------------------------------------------------------

def cfg_json(**overrides) -> str:
    return json.dumps(params_as_cfg(**overrides), sort_keys=True, default=float)


def export_row(record: dict, binder: "st.delta_generator.DeltaGenerator"):
    plain = studio_core.record_to_plain(record)
    binder.download_button("⬇ Export result (JSON)",
                           json.dumps(plain, indent=2, default=str).encode(),
                           file_name=f"{plain.get('experiment_id', 'experiment')}.json",
                           mime="application/json")
    binder.download_button("⬇ Export result (CSV)",
                           pd.json_normalize(plain).to_csv(index=False).encode(),
                           file_name=f"{plain.get('experiment_id', 'experiment')}.csv",
                           mime="text/csv")


def save_to_history(record: dict):
    if st.button("💾 Save experiment to local history"):
        path = studio_core.save_experiment(studio_core.record_to_plain(record))
        st.success(f"Saved to `{os.path.relpath(path)}` — export before container restarts (non-durable on Cloud).")


def show_pricing_record(record: dict):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price estimate", f"{record['price']:.6f}")
    c2.metric("Standard error", f"{record['standard_error']:.6f}")
    c3.metric("95% CI", f"[{record['ci_low']:.4f}, {record['ci_high']:.4f}]")
    c4.metric("Paths", f"{record['paths']:,}")
    d1, d2, d3 = st.columns(3)
    d1.metric("Runtime", f"{record['runtime_s']:.3f} s")
    d2.metric("Paths / second", f"{(record.get('paths_per_second') or 0):,.0f}")
    d3.metric("Seed", str(record["seed"]))
    engine = st.container()
    engine.caption(f"Engine call: `{record['engine_call']}` · copula **{record['copula']}** · "
                   f"{record['contract']} · ρ={record['rho']:.2f}"
                   + (f" · ν={record['nu']}" if record["copula"] == "Student-t" else ""))
    st.caption("The 95% confidence interval is a Monte Carlo sampling interval "
               "(normal approximation), not a guaranteed bound on the true model price.")


PAGE_NUM = SECTIONS.index(page)

# ===========================================================================
# 1. RESEARCH OVERVIEW
# ===========================================================================
if PAGE_NUM == 0:
    st.title("CopulaDeriv Studio")
    st.markdown("### *Dependence-Aware Monte Carlo Pricing of Multi-Asset Rainbow Options*")
    st.markdown(
        "**Research title:** Dependence-Aware Rainbow Option Pricing: A Hybrid "
        "Quantitative and Computational Study of Gaussian and Student-t Copulas."
    )

    st.markdown("#### Why this research problem")
    st.markdown(r"""
Multi-asset (rainbow) derivatives derive their value from the joint behaviour of several
underlyings, so the contract price depends on **both** the individual marginal
distributions of each asset **and** the dependence structure that links them. The central
methodological point of this project is that these two components can be separated:

- **Marginal distributions** describe what each asset does individually (here: risk-neutral
  lognormal marginals from geometric Brownian motion).
- **A copula** describes how realisations of the assets move together, and can be varied
  *without* changing the marginals.

This separation lets us ask clean research questions:
1. How does dependence (the copula parameter ρ) change the price of a best-of call and a
   worst-of put — in opposite directions?
2. How do a **Gaussian copula** and a **Student-t copula** with the same latent ρ differ?
   The Student-t copula introduces *tail dependence*: joint extreme moves that the Gaussian
   copula asymptotically rules out.
3. How do tail-dependence differences show up in payoff distributions — and how large must
   the Monte Carlo uncertainty be before we can claim a price difference at all?
4. What does each estimator cost in wall-clock terms (Python reference vs the local C++20
   engine)?
""")

    st.markdown("#### Concise mathematical model")
    st.markdown(r"""
Under the risk-neutral measure $Q$, each underlying follows a geometric Brownian motion
with drift $(r-q_i)$ and the exact terminal solution

$$S_i(T)=S_i(0)\exp\left[(r-q_i-\tfrac{1}{2}\sigma_i^2)T+\sigma_i\sqrt{T}\,Z_i\right],$$

with standard normal $Z_i$. Dependence enters through the copula that maps the pair
$(Z_1,Z_2)$ (or rather the uniform pair $U_i = F_i(\cdot)$) to the joint distribution of
terminal prices. The two contracts studied are

$$H_{\text{best-call}}=\max(\max(S_1(T),S_2(T))-K,\,0),$$

$$H_{\text{worst-put}}=\max(K-\min(S_1(T),S_2(T)),\,0).$$

Each is priced by Monte Carlo as the discounted average terminal payoff

$$\hat V_0=e^{-rT}\,\frac{1}{N}\sum_{j=1}^{N}H_j,$$

with a standard error $SE(\hat V)$ and a 95% confidence interval
$\hat V_0 \pm 1.96\,SE(\hat V)$ attached to every displayed estimate.
""")

    st.markdown("#### What this application is (and is not)")
    st.info(
        "**Methodological framing.** Copulas here are applied to *risk-neutral lognormal "
        "marginals at maturity* — a **terminal dependence specification**. Running the "
        "Student-t copula through this pipeline is *not* automatically equivalent to a "
        "dynamically consistent risk-neutral diffusion with Student-t features; it is a "
        "comparable dependence scenario at the terminal date, holding marginals fixed. "
        "No historical (physical-measure) calibration is performed, and no historical "
        "correlation estimate is silently reused as a risk-neutral dependence parameter."
    )
    st.markdown(
        "- **Implemented & validated in the engine:** Gaussian copula, Student-t copula "
        "(one shared χ²(ν) draw per observation), Cholesky-baseline GBM, best-of call, "
        "worst-of put, MC standard errors and confidence intervals, analytic "
        "Black–Scholes single-asset benchmark with Greeks, empirical and theoretical "
        "tail-dependence estimators.\n"
        "- **Not implemented here:** variance-reduction estimators (antithetic / control "
        "variate exist only as offline Day-10 experiment scripts), finite-difference Greeks "
        "for the rainbow contracts, importance sampling. The corresponding pages disclose "
        "this honestly.\n"
        "- **Performance section:** Python throughput measurements from *this deployed "
        "instance*; C++20 numbers require the local benchmark workflow (the cloud backend "
        "runs Python only)."
    )

    st.markdown("#### Navigation")
    st.markdown(
        "| Page | Purpose |\n"
        "|---|---|\n"
        "| Pricing Laboratory | Configure and run a single pricing experiment |\n"
        "| Copula Comparison | Gaussian vs Student-t under identical marginals (paired CRN) |\n"
        "| Sensitivity Analysis | Price response to ρ, ν, volatility and strike |\n"
        "| Tail Dependence | Theoretical vs finite-threshold empirical tail dependence |\n"
        "| Greeks and Risk | What is implemented; payoff-distribution risk analytics |\n"
        "| Monte Carlo Convergence | Price / SE / runtime versus path count |\n"
        "| Performance Benchmarks | Python throughput here; C++ local workflow |\n"
        "| Experiment History | Saved runs, export, re-run |\n"
        "| Methodology | Full assumptions, limits and validation summary |"
    )

# ===========================================================================
# 2. PRICING LABORATORY
# ===========================================================================
elif PAGE_NUM == 1:
    st.title("Pricing Laboratory")
    st.caption("Runs the existing validated engine directly. Every estimate ships with its "
               "standard error, 95% CI, path count, seed and runtime.")

    cfg_string = cfg_json()
    colA, colB = st.columns([1, 2])
    run_clicked = colA.button("▶ Run pricing experiment", type="primary")
    copy_conf = colB.caption(
        f"Contract **{p['contract']}** · {p['copula']} · ρ={p['rho']:.2f}"
        + (f" · ν={p['nu']:.1f}" if p["copula"] == "Student-t" else "")
        + f" · N={p['N']:,} · seed={p['seed']}")

    if run_clicked:
        try:
            record = cached_pricing(cfg_string)
            st.session_state["last_record"] = record
            show_pricing_record(record)
            save_to_history(record)
            export_row(record, st)
        except studio_core.ExperimentError as err:
            st.error(f"Invalid experiment configuration: {err}")
        except MemoryError:
            st.error("Out of memory: reduce the path count.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Computation failed: {exc}")
    elif "last_record" in st.session_state:
        show_pricing_record(st.session_state["last_record"])

    st.markdown("#### Payoff definitions")
    st.markdown(r"""
$$H_{\text{best-call}}=\max(\max(S_1(T),S_2(T))-K,\,0),\qquad
H_{\text{worst-put}}=\max(K-\min(S_1(T),S_2(T)),\,0)$$

and the price estimate $\hat V_0=e^{-rT}\,\frac1N\sum_j H_j$, standard error
$SE=\frac{e^{-rT}\,\mathrm{sd}(H)}{\sqrt N}$, 95% CI $\hat V_0\pm1.96\,SE$.
""")

# ===========================================================================
# 3. COPULA COMPARISON
# ===========================================================================
elif PAGE_NUM == 2:
    st.title("Copula Comparison — Gaussian vs Student-t")
    st.markdown(
        "Both models use **identical risk-neutral lognormal marginals**; only the copula "
        "changes. The same seed produces **common random numbers**: both samplers share "
        "the latent Gaussian vector, so the comparison is paired."
    )

    if st.button("▶ Run paired comparison", type="primary"):
        try:
            record = cached_paired(cfg_json())
            st.session_state["last_paired"] = record
        except studio_core.ExperimentError as err:
            st.error(f"Invalid experiment configuration: {err}")
    if "last_paired" not in st.session_state:
        st.info("Press ▶ Run paired comparison to execute the experiment.")
        st.stop()
    record = st.session_state["last_paired"]

    st.markdown("#### Side-by-side results")
    fmt_ci = lambda lo, hi: f"[{lo:.4f}, {hi:.4f}]"  # noqa: E731
    table = pd.DataFrame({
        "Metric": ["Estimated price", "Standard error", "95% CI",
                   "Mean discounted payoff", "Payoff standard deviation",
                   "Runtime for paired run (s)", "Paths per second (combined)"],
        "Gaussian": [f"{record['gaussian_price']:.6f}",
                     f"{record['gaussian_standard_error']:.6f}",
                     fmt_ci(record['gaussian_ci_low'], record['gaussian_ci_high']),
                     f"{record['gaussian_price']:.6f}",
                     f"{record['gaussian_payoff_SD']:.6f}",
                     f"{record['runtime_s']:.3f}",
                     f"{(record.get('paths_per_second') or 0):,.0f}"],
        "Student-t": [f"{record['studentt_price']:.6f}",
                      f"{record['studentt_standard_error']:.6f}",
                      fmt_ci(record['studentt_ci_low'], record['studentt_ci_high']),
                      f"{record['studentt_price']:.6f}",
                      f"{record['studentt_payoff_SD']:.6f}",
                      f"{record['runtime_s']:.3f}",
                      f"{(record.get('paths_per_second') or 0):,.0f}"],
    })
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.markdown("#### Paired difference (Student-t − Gaussian)")
    diff, dse = record["paired_price_difference"], record["paired_standard_error"]
    c1, c2, c3 = st.columns(3)
    c1.metric("ΔV (Student-t − Gaussian)", f"{diff:.6f}")
    c2.metric("Paired standard error", f"{dse:.6f}")
    c3.metric("95% CI (paired)",
              f"[{record['paired_ci_low']:.4f}, {record['paired_ci_high']:.4f}]")

    if abs(record["gaussian_price"]) < 5.0 * record["gaussian_standard_error"]:
        st.warning("The Gaussian reference price is zero or numerically close to zero "
                   "relative to its standard error, so percentage differences are "
                   "uninformative and are not reported.")
    else:
        pct = diff / record["gaussian_price"] * 100.0
        st.markdown(f"Percentage difference: **{pct:+.3f}%** of the Gaussian price.")

    if record["p_within_2se"]:
        st.info("The paired difference is **within two paired standard errors of zero**: "
                "this experiment does **not** establish a statistically detectable price "
                "difference. Increase N or change ν/ρ before interpreting the sign.")
    else:
        st.info("The paired difference exceeds two paired standard errors; using seed-based "
                "common random numbers this is indicative, but a formal significance claim "
                "would require a full statistical procedure (e.g. repeated independent runs).")

    st.caption(f"Paths N={record['paths']:,} · seed={record['seed']} · paired SE computed "
               "from the sample sd of pathwise discounted payoff differences (ddof=1) / √N — "
               "*not* from combining two independent standard errors.")
    save_to_history(record)
    export_row(record, st)

# ===========================================================================
# 4. SENSITIVITY ANALYSIS
# ===========================================================================
elif PAGE_NUM == 3:
    st.title("Sensitivity Analysis")
    st.markdown(
        "Each sweep changes **exactly one** parameter and keeps every other parameter, the "
        "path count and the seed fixed. Points are separate Monte Carlo estimates — treat "
        "differences smaller than the plotted CIs as noise."
    )

    s1, s2, s3, s4 = st.tabs(["ρ (correlation)", "ν (degrees of freedom)", "Volatility", "Strike"])

    with s1:
        rho_grid = st.multiselect("ρ grid", [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75],
                                  default=[v for v in (-0.75, -0.25, 0.0, 0.25, 0.75) if -1 <= v <= 1])
        paired = st.checkbox("Paired Gaussian-vs-Student-t sweep (CRN)", value=True,
                             help="Sorts each ρ through the paired difference of §3.")
        if st.button("Run ρ sweep", type="primary"):
            try:
                corr_records = cached_sweep(cfg_json(),
                                            "rho", json.dumps([float(v) for v in rho_grid]), paired)
                df = pd.DataFrame(corr_records)
                fig = go.Figure()
                if paired:
                    fig.add_trace(go.Scatter(x=df["rho"], y=df["gaussian_price"],
                                             error_y=dict(array=1.959963984540054 * df["gaussian_standard_error"]),
                                             name="Gaussian: best-of call", mode="lines+markers"))
                    fig.add_trace(go.Scatter(x=df["rho"], y=df["studentt_price"],
                                             error_y=dict(array=1.959963984540054 * df["studentt_standard_error"]),
                                             name=f"Student-t (ν={p['nu']:.0f}): best-of call", mode="lines+markers"))
                    st.plotly_chart(fig, use_container_width=True)
                    figd = go.Figure()
                    figd.add_trace(go.Scatter(x=df["rho"], y=df["paired_price_difference"],
                                              error_y=dict(array=1.959963984540054 * df["paired_standard_error"]),
                                              name="ΔV (Student-t − Gaussian)", mode="lines+markers"))
                    figd.update_layout(title="Paired price difference vs ρ (with 1.96·paired SE)",
                                       xaxis_title="ρ", yaxis_title="ΔV")
                    st.plotly_chart(figd, use_container_width=True)
                else:
                    for contract_name in ("Best-of call", "Worst-of put"):
                        sub = []
                        for value in rho_grid:
                            record = cached_pricing(cfg_json(contract=contract_name, rho=float(value)))
                            sub.append(record)
                        subdf = pd.DataFrame(sub)
                        fig.add_trace(go.Scatter(x=subdf["rho"], y=subdf["price"],
                                                 error_y=dict(array=1.959963984540054 * subdf["standard_error"]),
                                                 name=contract_name, mode="lines+markers"))
                    st.plotly_chart(fig, use_container_width=True)
                st.caption("Interpretation: best-of calls are *hurt* by strong positive "
                           "dependence; worst-of puts are *boosted* by it. Error bars are "
                           "1.96·SE per point; overlapping bars ⇒ differences not resolved.")
                st.download_button("⬇ Export sweep (CSV)",
                                   df.to_csv(index=False).encode(),
                                   file_name="sensitivity_rho.csv", mime="text/csv")
            except studio_core.ExperimentError as err:
                st.error(f"Invalid sweep configuration: {err}")

    with s2:
        nu_grid = st.multiselect("ν grid", [3.0, 4.0, 6.0, 10.0, 20.0, 40.0], default=[4.0, 6.0, 10.0, 20.0])
        if nu_grid and st.button("Run ν sweep (Student-t)", type="primary"):
            records = cached_sweep(cfg_json(copula="Student-t"), "nu",
                                   json.dumps([float(v) for v in nu_grid]), False)
            df = pd.DataFrame(records)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["nu"], y=df["price"],
                                     error_y=dict(array=1.959963984540054 * df["standard_error"]),
                                     name=f"{p['contract']}", mode="lines+markers"))
            gauss = cached_pricing(cfg_json(copula="Gaussian"))
            fig.add_hline(y=gauss["price"], line_dash="dot",
                          annotation_text=f"Gaussian copula price ρ={p['rho']:.2f}")
            fig.update_layout(xaxis_title="Degrees of freedom ν", yaxis_title="Option price")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Larger ν pulls the Student-t copula toward the Gaussian copula "
                       "(up to how the latent parameter is specified), so the curve should "
                       "drift toward the dotted Gaussian reference as ν grows.")
            st.download_button("⬇ Export sweep (CSV)", df.to_csv(index=False).encode(),
                               file_name="sensitivity_nu.csv", mime="text/csv")

    with s3:
        which = st.radio("Vary which volatility?", ("σ₁ (asset 1)", "σ₂ (asset 2)"), horizontal=True)
        key = "sigma1" if which.startswith("σ₁") else "sigma2"
        vol_grid = st.multiselect("Volatility grid", [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
                                  default=[0.10, 0.15, 0.20, 0.25, 0.30])
        if vol_grid and st.button("Run volatility sweep", type="primary"):
            records = cached_sweep(cfg_json(), key, json.dumps(vol_grid), False)
            df = pd.DataFrame(records)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df[key], y=df["price"],
                                     error_y=dict(array=1.959963984540054 * df["standard_error"]),
                                     name=p["contract"], mode="lines+markers"))
            fig.update_layout(xaxis_title=which, yaxis_title="Option price")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("All other parameters (including the *other* asset's volatility) are held fixed.")
            st.download_button("⬇ Export sweep (CSV)", df.to_csv(index=False).encode(),
                               file_name=f"sensitivity_{key}.csv", mime="text/csv")

    with s4:
        strike_grid = st.multiselect("Strike grid", [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0],
                                     default=[80.0, 100.0, 120.0])
        if strike_grid and st.button("Run strike sweep", type="primary"):
            records = cached_sweep(cfg_json(), "K", json.dumps(strike_grid), False)
            df = pd.DataFrame(records)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["K"], y=df["price"],
                                     error_y=dict(array=1.959963984540054 * df["standard_error"]),
                                     name=p["contract"], mode="lines+markers"))
            fig.update_layout(xaxis_title="Strike K", yaxis_title="Option price")
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("⬇ Export sweep (CSV)", df.to_csv(index=False).encode(),
                               file_name="sensitivity_strike.csv", mime="text/csv")

# ===========================================================================
# 5. TAIL DEPENDENCE
# ===========================================================================
elif PAGE_NUM == 4:
    st.title("Tail Dependence")
    st.markdown(r"""
**Asymptotic (theoretical) tail dependence.** For the Gaussian copula with $-1<\rho<1$,

$$\lambda_U=\lambda_L=0,$$

so joint extreme events are asymptotically rare. For the bivariate Student-t copula,

$$\lambda_U=\lambda_L=\;2\,t_{\nu+1}\!\left(-\sqrt{\tfrac{(\nu+1)(1-\rho)}{1+\rho}}\right),$$

which is strictly positive for finite $\nu$ (equal for upper and lower tails).
**Finite-threshold empirical estimates** $\lambda_U(u)=P(U_2>u\mid U_1>u)$ below use a fixed
threshold $u$, estimate something else (a finite-level exceedance probability), and are
noisy — their reliability depends on the number of joint exceedances, displayed next to
each estimate.
""")
    st.markdown("Run the tail-dependence study with your current parameter panel settings.")
    if st.button("Run tail-dependence analysis", type="primary"):
        try:
            theo = studio_core.theoretical_tail_dependence(p["rho"], p["nu"])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Gaussian λ_U (theoretical)", theo["gaussian_lambda_upper"])
            c2.metric("Gaussian λ_L (theoretical)", theo["gaussian_lambda_lower"])
            c3.metric("Student-t λ_U (theoretical)",
                      f"{theo['studentt_lambda_upper']:.4f}", help=f"ν={p['nu']:.1f}, ρ={p['rho']:.2f}")
            c4.metric("Student-t λ_L (theoretical)", f"{theo['studentt_lambda_lower']:.4f}")

            rows = studio_core.empirical_tail_dependence_table(p and params_as_cfg())
            df = pd.DataFrame(rows)
            df["threshold"] = df["threshold"].round(4)
            df["empirical_lambda"] = df["empirical_lambda"].round(4)
            st.markdown("#### Empirical finite-threshold estimates (copula uniforms)")
            st.dataframe(df, use_container_width=True, hide_index=True)

            sparse = df[df["joint_exceedances"] < 30]
            if len(sparse):
                st.warning(f"{len(sparse)} estimate(s) are based on fewer than 30 joint "
                           "exceedances and are **not reliable**: "
                           + ", ".join(f"{r.copula} {r.side} u={r.threshold} (n={r.joint_exceedances})"
                                       for r in sparse.itertuples()) + ".")
            st.caption("A noisy empirical value does not prove or refute the theoretical "
                       "coefficient. Note also: the Gaussian copula has *zero asymptotic* "
                       "tail dependence, yet its finite-threshold exceedance probability can "
                       "still be substantial — these are different quantities.")
            st.download_button("⬇ Export empirical table (CSV)", df.to_csv(index=False).encode(),
                               file_name="tail_dependence_empirical.csv", mime="text/csv")

            # Theoretical sweeps
            rho_grid = np.linspace(-0.9, 0.9, 19)
            nu_grid = [3.0, 5.0, 10.0, 30.0]
            fig = go.Figure()
            for nu_g in nu_grid:
                ys = [studio_core.theoretical_tail_dependence(r, nu_g)["studentt_lambda_upper"] for r in rho_grid]
                fig.add_trace(go.Scatter(x=rho_grid, y=ys, name=f"Student-t ν={nu_g:.0f}"))
            fig.add_hline(y=0.0, annotation_text="Gaussian λ = 0", line_dash="dot")
            fig.update_layout(title="Theoretical Student-t tail dependence λ(ρ) and the Gaussian zero line",
                              xaxis_title="ρ", yaxis_title="λ_U = λ_L")
            st.plotly_chart(fig, use_container_width=True)
        except studio_core.ExperimentError as err:
            st.error(f"Invalid configuration: {err}")

# ===========================================================================
# 6. GREEKS AND RISK
# ===========================================================================
elif PAGE_NUM == 5:
    st.title("Greeks and Risk")
    st.subheader("What is implemented in CopulaDeriv — and what is not")
    st.error(
        "**Rainbow-contract Greeks are NOT implemented in the repository.** The engine "
        "provides no finite-difference delta/vega for best-of calls or worst-of puts, so "
        "none are displayed. Adding them would require: (a) seeded common-random-number "
        "bump-and-reprice functions in the engine, (b) bump-size sensitivity testing, and "
        "(c) validation against the existing test suite."
    )
    st.markdown("#### Implemented reference: analytic single-asset Black–Scholes Greeks")
    st.markdown(
        "The engine has closed-form Greeks for the **single-asset European call benchmark** "
        "(`pricing.black_scholes_european_call`). Shown here as a validation reference for "
        "asset 1 with this page's panel parameters — *not* a rainbow-contract Greek."
    )
    bs = studio_core.pricing.black_scholes_european_call(
        p["S1_0"], p["K"], p["r"], p["q1"], p["sigma1"], p["T"])
    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("BS price (asset 1)", f"{bs['price']:.4f}")
    g2.metric("Delta", f"{bs['delta']:.4f}")
    g3.metric("Gamma", f"{bs['gamma']:.6f}")
    g4.metric("Vega (per unit σ)", f"{bs['vega']:.4f}")
    g5.metric("Theta (per year)", f"{bs['theta']:.4f}")
    st.caption("Analytic, deterministic, no Monte Carlo error. Reproduce with: "
               "`black_scholes_european_call(S1_0, K, r, q1, sigma1, T)`.")

    st.markdown("#### Payoff-distribution risk analytics (unhedged contract convention)")
    st.markdown(
        "Computed from the engine's copula sampler and payoff functions for the panel "
        "configuration. **Loss convention:** the contract is held *unhedged* to maturity and "
        "compared against its t=0 fair value: `loss = e^{rT}·V₀ − e^{−rT}·H_T` (positive loss "
        "= realised discounted payoff below fair value). Its α-VaR is the empirical "
        "α-quantile of this loss; its α-CVaR is the mean loss among paths at or beyond that "
        "quantile. **This is the risk of the terminal payoff distribution of an unhedged "
        "option — not the market risk of an actively hedged portfolio.**"
    )
    alpha = st.select_slider("VaR / CVaR level α", options=[0.90, 0.95, 0.975, 0.99], value=0.99)
    if st.button("Run payoff-distribution analytics", type="primary"):
        try:
            analytics = studio_core.payoff_analytics(params_as_cfg(), float(alpha))
            q = analytics["payoff_quantiles"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Discounted payoff mean", f"{analytics['discounted_payoff_mean']:.6f}")
            c2.metric("Discounted payoff sd", f"{analytics['discounted_payoff_SD']:.6f}")
            c3.metric(f"VaR (α={alpha:.2f})", f"{analytics['VaR']:.4f}")
            c4.metric(f"CVaR (α={alpha:.2f})", f"{analytics['CVaR']:.4f}")
            st.markdown(f"Discounted-payoff quantiles — "
                        f"1%: {q['1%']:.4f} · 5%: {q['5%']:.4f} · median: {q['50%']:.4f} · "
                        f"95%: {q['95%']:.4f} · 99%: {q['99%']:.4f}")
            st.metric("P(zero payoff)", f"{analytics['prob_zero_payoff']:.4%}",
                      help="Share of paths where the contract expires worthless.")
            st.caption(f"Paths N={analytics['paths']:,} · seed={analytics['seed']} · "
                       "copula " + p["copula"] + ".")
        except studio_core.ExperimentError as err:
            st.error(f"Invalid configuration: {err}")

# ===========================================================================
# 7. MONTE CARLO CONVERGENCE
# ===========================================================================
elif PAGE_NUM == 6:
    st.title("Monte Carlo Convergence")
    st.markdown(r"""
Expected statistical scaling: $SE(\hat V)\propto 1/\sqrt N$. Doubling accuracy costs
**4×** work. Estimates at different N are independent Monte Carlo runs — prices need not
converge monotonically; the standard error *should* shrink like $N^{-1/2}$.
The simulation uses a chunked stream (chunks of 100k paths, deterministic sub-stream
seeds) so no single array exceeds 100k rows regardless of N.
""")
    counts = st.multiselect("Path counts", [1_000, 5_000, 10_000, 50_000, 100_000, 500_000],
                            default=[1_000, 10_000, 50_000, 100_000])
    if counts and st.button("Run convergence study", type="primary"):
        try:
            records = cached_convergence(cfg_json(), json.dumps([int(n) for n in sorted(counts)]))
            df = pd.DataFrame(records)
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["paths"], y=df["price"],
                                         error_y=dict(array=1.959963984540054 * df["standard_error"]),
                                         name=p["contract"], mode="lines+markers"))
                fig.update_layout(title="Price estimate vs path count (bars: 1.96·SE)",
                                  xaxis_type="log", xaxis_title="N (log)", yaxis_title="Price")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig_se = go.Figure()
                fig_se.add_trace(go.Scatter(x=df["paths"], y=df["standard_error"],
                                            name="SE (measured)", mode="lines+markers"))
                ref = df.iloc[0]
                fig_se.add_trace(go.Scatter(x=df["paths"],
                                            y=ref["standard_error"] * np.sqrt(ref["paths"] / df["paths"]),
                                            name="1/√N reference", line=dict(dash="dot")))
                fig_se.update_layout(title="Standard error vs path count",
                                     xaxis_type="log", yaxis_type="log",
                                     xaxis_title="N (log)", yaxis_title="SE (log)")
                st.plotly_chart(fig_se, use_container_width=True)
            c3, c4 = st.columns(2)
            with c3:
                fig_c = go.Figure()
                fig_c.add_trace(go.Scatter(x=df["paths"], y=df["ci_high"] - df["ci_low"],
                                           name="95% CI width", mode="lines+markers"))
                fig_c.update_layout(title="95% CI width vs path count", xaxis_type="log",
                                    xaxis_title="N (log)", yaxis_title="CI width")
                st.plotly_chart(fig_c, use_container_width=True)
            with c4:
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatter(x=df["paths"], y=df["runtime_s"],
                                           name="Runtime (s)", mode="lines+markers"))
                fig_r.update_layout(title="Runtime vs path count", xaxis_type="log",
                                    xaxis_title="N (log)", yaxis_title="Seconds")
                st.plotly_chart(fig_r, use_container_width=True)
            st.dataframe(df[["paths", "price", "standard_error", "ci_low", "ci_high", "runtime_s"]],
                         use_container_width=True, hide_index=True)
            st.download_button("⬇ Export convergence data (CSV)", df.to_csv(index=False).encode(),
                               file_name="monte_carlo_convergence.csv", mime="text/csv")
            st.caption("Streaming detail: each N is split into deterministic sub-streams "
                       "(seed = base seed + chunk·1000003); price and variance are combined "
                       "across chunks, so the estimate matches the composite stream exactly.")
        except studio_core.ExperimentError as err:
            st.error(f"Invalid configuration: {err}")

# ===========================================================================
# 8. PERFORMANCE BENCHMARKS
# ===========================================================================
elif PAGE_NUM == 7:
    st.title("Performance Benchmarks")
    st.markdown(
        "All numbers below are **measured live on this machine/instance** by running the "
        "Python engine; nothing is imported from a stored benchmark table. The C++20 "
        "engine is available **locally only** (see the local workflow below)."
    )

    st.subheader("Python engine (this instance)")
    bench_n = st.select_slider("Benchmark path count", options=[10_000, 50_000, 100_000, 200_000],
                               value=100_000)
    reps = st.select_slider("Repetitions", options=[3, 5, 7], value=5)
    if st.button("▶ Run Python benchmark", type="primary"):
        cfg = params_as_cfg(N=int(bench_n))
        times, last = [], None
        prog = st.progress(0.0)
        for i in range(reps):
            record = studio_core.run_pricing_experiment(cfg)  # bypass cache: measure real runs
            times.append(record["runtime_s"])
            last = record
            prog.progress((i + 1) / reps)
        median = float(np.median(times))
        st.markdown(
            f"- Median runtime of **{reps} runs** (N={bench_n:,}): **{median:.3f} s** "
            f"(min {min(times):.3f} s, max {max(times):.3f} s)\n"
            f"- Throughput (median): **{bench_n / median:,.0f} paths/second**\n"
            f"- Price estimate (last run): {last['price']:.6f} ± {last['standard_error']:.6f}\n"
            f"- Implementation: Python {platform.python_version()}, "
            f"NumPy {np.__version__}, SciPy {__import__('scipy').__version__}"
        )
        st.caption(
            "Methodology: end-to-end timing (sampler + marginal transform + payoff + "
            "statistics). No warm-up loops are claimed beyond the cache-free repetitions; "
            "variability across repetitions is reported as min/median/max. "
            "**Monte Carlo throughput (paths/second) is NOT HFT event latency** — it says "
            "nothing about per-event trading-system response times."
        )
        st.download_button("⬇ Export benchmark (CSV)",
                           pd.DataFrame([{**last, "median_runtime_s": median,
                                          "min_runtime_s": min(times), "max_runtime_s": max(times),
                                          "repetitions": reps}]).to_csv(index=False).encode(),
                           file_name="python_benchmark.csv", mime="text/csv")

    st.subheader("C++20 engine — local workflow (not deployable on Streamlit Cloud)")
    st.info(
        "The C++20 engine cannot run as part of this Streamlit Community Cloud deployment, "
        "and this page will not pretend otherwise. Use the local C++ benchmark described "
        "below; when you publish local benchmark numbers, document compiler, flags, CPU "
        "and OS with them and compare like-for-like (end-to-end vs end-to-end, or core "
        "compute vs core compute)."
    )
    with st.expander("Local C++ benchmark workflow (verbatim commands)"):
        st.markdown(r"""
```bash
# from the repository root (CopulaDeriv)
cd copuladeriv/cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
./build/copuladeriv          # runs the engine's built-in pricing example
```
The engine binary performs the same pipeline as the Python reference (correlated
normals → copula/marginal transform → rainbow payoff → MC statistics), so a fair
comparison is to give both implementations the same configuration, path count and
repetition scheme and compare *end-to-end* medians, reported together with:
compiler and flags, CPU model, OS, Python/package versions, path counts, and the
price estimate ± standard error from both implementations.
""")
        st.markdown(
            "Never compare one implementation's core computation time against the other's "
            "complete runtime, and never divide one elapsed time by another to advertise a "
            "global speedup factor."
        )

# ===========================================================================
# 9. EXPERIMENT HISTORY
# ===========================================================================
elif PAGE_NUM == 8:
    st.title("Experiment History & Reproducibility")
    st.markdown(
        "Experiments saved from any page are appended to "
        "`research/dashboard_experiments/experiments.jsonl`. **Storage on Streamlit "
        "Community Cloud is not durable** — a restart can wipe it — so use the export "
        "buttons to keep local copies. Records distinguish plain user experiments by "
        "their `experiment_kind` field; offline research-experiment results live in the "
        "separate `python/experiments/results/` CSV collection and synthetic examples are "
        "not generated here."
    )
    records = studio_core.load_experiments()
    if not records:
        st.info("No experiments saved yet. Run an experiment and press 💾 Save.")
    else:
        df = pd.json_normalize(records)
        keep = [c for c in df.columns if c in
                ("experiment_id", "timestamp_utc", "copula", "contract", "rho", "nu",
                 "seed", "price", "standard_error", "paired_price_difference",
                 "paths", "runtime_s", "experiment_kind", "studio_version")]
        st.dataframe(df[keep], use_container_width=True, hide_index=True)
        ids = df["experiment_id"].tolist()
        chosen = st.multiselect("Select experiments to compare", ids)
        if chosen:
            sel = df[df["experiment_id"].isin(chosen)]
            shared_cols = [c for c in keep if sel[c].nunique(dropna=False) <= 1]
            diff_cols = [c for c in keep if c not in shared_cols]
            st.markdown("Differences between the selected experiments: "
                        + (", ".join(f"`{c}`" for c in diff_cols) or "none within the summary columns") + ".")
            st.dataframe(sel[keep], use_container_width=True, hide_index=True)
        st.download_button("⬇ Export full history (CSV)", df.to_csv(index=False).encode(),
                           file_name="experiment_history.csv", mime="text/csv")
        st.download_button("⬇ Export full history (JSONL)",
                           "\n".join(json.dumps(r, default=str) for r in records).encode(),
                           file_name="experiment_history.jsonl", mime="application/json")
        if st.button("🗑 Clear local history"):
            os.remove(os.path.join(studio_core.history_directory(),
                                   studio_core.HISTORY_FILE))
            st.success("History cleared.")

# ===========================================================================
# 11. SYSTEM DIAGNOSTICS
# ===========================================================================
else:
    st.title("System Diagnostics")
    st.markdown(
        "Environment and availability status of this CopulaDeriv Studio instance. "
        "No secrets, tokens, or server paths are exposed on this page."
    )

    deps = {}
    for module_name in ("numpy", "scipy", "pandas", "plotly", "streamlit"):
        try:
            mod = __import__(module_name)
            deps[module_name] = getattr(mod, "__version__", "installed")
        except ImportError:
            deps[module_name] = "NOT INSTALLED"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Runtime")
        st.code(
            f"Studio version      {studio_core.APP_VERSION}\n"
            f"Python              {platform.python_version()}\n"
            f"Operating system    {platform.system()} {platform.release()}\n"
            f"Machine type        {platform.machine()}", language="text")
        st.markdown("#### Dependency status")
        for name, version in deps.items():
            ok = version != "NOT INSTALLED"
            st.markdown(f"{'✅' if ok else '❌'} `{name}` — {version}")
    with c2:
        st.markdown("#### Engine availability")
        engine_ok = hasattr(studio_core, "pricing") and hasattr(
            studio_core.pricing, "monte_carlo_best_of_call_price_copula")
        st.markdown(f"{'✅' if engine_ok else '❌'} Python pricing engine "
                    "(`monte_carlo_*` API present)")
        st.markdown("✅ Gaussian copula sampler")
        st.markdown("✅ Student-t copula sampler (shared χ² scale)")
        st.markdown("✅ Tail-dependence estimators (theoretical + empirical)")
        st.markdown("✅ Analytic Black–Scholes single-asset Greeks (reference)")
        st.markdown("❌ Rainbow-contract finite-difference Greeks — not implemented")
        st.markdown("❌ Variance-reduction engine estimators — offline scripts only")
        st.markdown("❌ C++20 engine on this cloud instance — local workflow only (`copuladeriv/cpp/`)")

    st.markdown("#### Interactive simulation limits")
    st.info(f"Maximum interactive path count N = {studio_core.MAX_PATHS_INTERACTIVE:,} "
            f"(chunked streaming at {studio_core.CHUNK_SIZE:,} paths per chunk for "
            "convergence studies). Limits are chosen conservatively for the free "
            "Streamlit Community Cloud tier.")

    st.markdown("#### Experiment diagnostics log")
    records = studio_core.load_experiments()
    if records:
        st.markdown(f"Last experiment: `{records[0]['experiment_id']}` — "
                    f"{records[0].get('experiment_kind', 'single_pricing')} — "
                    f"{records[0].get('timestamp_utc', '?')} UTC")
    else:
        st.markdown("No experiments executed in this instance yet.")

    st.markdown("#### Validation status")
    st.markdown(
        "- Engine parity, paired-CRN statistics, tail-dependence reference formula, CI "
        "arithmetic and input validation are covered by automated tests — see "
        "`docs/studio_validation.md` and run "
        "`python -m pytest copuladeriv/tests/ -q` from `copuladeriv/python`.")
    st.title("Methodology, Assumptions & Known Limitations")
    st.markdown("#### Statistical contract of this dashboard")
    st.markdown(r"""
1. Identical risk-neutral lognormal marginals whenever copulas are compared — only the
   copula changes; $E_Q[S_i(T)] = S_i(0)\,e^{(r-q_i)T}$ is untouched.
2. The Student-t copula uses **one shared χ²(ν) draw per multivariate observation**
   (per-asset independent scales are invalid and are never used).
3. ρ is the *latent copula parameter*; it is **not** the Pearson correlation of the
   terminal asset prices.
4. Paired comparisons use common random numbers and report the paired SE from the
   pathwise discounted payoff differences (ddof=1)/√N — never two independent SEs combined.
5. Every displayed metric carries N, seed and runtime; every CI is the normal-approximation
   95% interval $\hat V_0\pm1.96\,SE$.
6. Significance language is restricted: differences within 2 paired SEs are labelled
   "not statistically distinguishable" in this experiment; stronger claims need repeated
   independent runs.
7. Model risk (copula/marginal choice) and Monte Carlo sampling uncertainty are *reported
   separately* — the CI measures the second only.
""")

    st.markdown("#### Modelling assumptions")
    st.markdown(r"""
- Risk-neutral GBM marginals: $dS_i/S_i=(r-q_i)\,dt+\sigma_i\,dW_i$, terminal solution
  $S_i(T)=S_i(0)\,e^{(r-q_i-\sigma_i^2/2)T+\sigma_i\sqrt T Z_i}$ — exact (no time stepping).
- Terminal-dependence specification: the copula couples the marginals **at maturity**.
  This is *not* a dynamically consistent stochastic process with Student-t dynamics, and
  the Student-t experiment is not automatically a diffusion-based dynamics model.
- No historical calibration: illustrative parameters are user-supplied, not fitted to
  market data. Physical-measure dependence is never reused as the risk-neutral dependence.
- rho validation: $[-1,1]$; ν validation: positive (engine-enforced, plus UI sanity limits).
""")

    st.markdown("#### Numerical limitations disclosed")
    st.markdown(r"""
- Normal-approximation CIs: for heavy-tailed Student-t cases (small ν) payoffs are
  heavy-tailed; the CLT-based SE is still asymptotically valid but finite-sample intervals
  can be optimistic. No bootstrap intervals are implemented.
- The χ²(ν) draw is floored at machine epsilon to avoid division by zero; ν below ~0.05 is
  rejected as unreliable.
- empirical tail-dependence estimates with <30 joint exceedances are flagged; with fewer
  than a handful they should not be quoted at all.
- Free-tier deployment caps: interactive runs ≤ 500 000 paths; convergence trials use
  100k-path chunks to keep memory bounded; runtime on a shared cloud CPU is slower than
  your local machine and varies between runs.
- Variance-reduction estimators (antithetic, control variate) and importance sampling are
  **not part of the deployed estimator set** — they exist only as offline Day-10
  experiments in the repository. Rainbow-contract finite-difference Greeks are **not
  implemented**.
- Monte Carlo throughput (paths/second) measured here has **no bearing on HFT
  event-processing latency**; they are different engineering metrics.
""")

    st.markdown("#### Environment of this deployment")
    st.code(
        f"Python {platform.python_version()}\n"
        f"NumPy {np.__version__} · SciPy {__import__('scipy').__version__} · "
        f"pandas {pd.__version__} · Streamlit {st.__version__}\n"
        f"OS: {platform.platform()}", language="text")
