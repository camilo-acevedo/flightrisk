from __future__ import annotations

from collections.abc import Iterable

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from flightrisk.utils.paths import get_paths

_BG = "rgba(0,0,0,0)"
_GRID = "rgba(148,163,184,0.10)"
_AXIS = "rgba(148,163,184,0.35)"
_FONT_FAMILY = "Inter, Segoe UI, system-ui, sans-serif"
_TEXT = "#E2E8F0"
_MUTED = "#94A3B8"
_DARK_PALETTE = {
    "risk": "#60A5FA",
    "uplift": "#34D399",
    "survival": "#C084FC",
    "random": "#64748B",
    "perfect": "#E2E8F0",
    "accent": "#F59E0B",
    "muted": "#94A3B8",
}


def _list_runs(track: str) -> list[str]:
    """List run subdirectories under ``reports/<track>/`` sorted by recency.

    :param track: Track identifier.
    :returns: List of run-id strings (most recent first).
    """
    base = get_paths().reports / track
    if not base.exists():
        return []
    runs = [p for p in base.iterdir() if (p / "model.joblib").exists()]
    return [p.name for p in sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)]


def _load_model(track: str, run_id: str) -> object:
    """Load a model artifact for the requested track / run.

    :param track: Track identifier.
    :param run_id: Run subdirectory name.
    :returns: Deserialised model.
    """
    path = get_paths().reports / track / run_id / "model.joblib"
    return joblib.load(path)


def _load_calibration(run_id: str) -> pd.DataFrame | None:
    """Load the calibration CSV emitted by the risk trainer if it exists.

    :param run_id: Risk run id.
    :returns: Calibration frame or ``None`` if missing.
    """
    path = get_paths().reports / "risk" / run_id / "calibration.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _load_qini_curve(run_id: str) -> pd.DataFrame | None:
    """Load the Qini curve CSV emitted by the uplift trainer if it exists.

    :param run_id: Uplift run id.
    :returns: Qini curve frame or ``None`` if missing.
    """
    path = get_paths().reports / "uplift" / run_id / "qini_curve.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _load_survival_matrix(run_id: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load the saved S(t) matrix and its horizons for a survival run.

    :param run_id: Survival run id.
    :returns: ``(matrix, horizons)`` or ``None`` if missing.
    """
    base = get_paths().reports / "survival" / run_id
    matrix_path = base / "survival_at_horizons.npy"
    if not matrix_path.exists():
        return None
    matrix = np.load(matrix_path)
    horizons = np.linspace(1, max(int(matrix.shape[1]), 1), matrix.shape[1])
    return matrix, horizons


def _shap_explanation(
    model: object, sample: pd.DataFrame, feature_names: Iterable[str]
) -> pd.DataFrame:
    """Compute SHAP values for a small sample if SHAP is available.

    :param model: Calibrated risk model wrapping a tree-based base.
    :param sample: Feature frame to explain (kept small for the demo).
    :param feature_names: Feature ordering.
    :returns: Tidy frame of mean absolute SHAP values per feature.
    """
    base = getattr(model, "base", model)
    estimator = getattr(base, "estimator", base)
    feature_list = list(feature_names)
    try:
        import shap

        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(sample[feature_list])
        if isinstance(values, list):
            values = values[1] if len(values) == 2 else values[-1]
        importance = np.abs(values).mean(axis=0)
        return (
            pd.DataFrame({"feature": feature_list, "mean_abs_shap": importance})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
    except Exception:
        return pd.DataFrame(
            {"feature": feature_list, "mean_abs_shap": [np.nan] * len(feature_list)}
        )


def _apply_layout_template(fig) -> None:
    """Apply the dark-mode flightrisk plotly layout template to a figure.

    :param fig: A plotly :class:`Figure`.
    """
    fig.update_layout(
        template="plotly_dark",
        font=dict(family=_FONT_FAMILY, color=_TEXT, size=13),
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        margin=dict(l=12, r=12, t=44, b=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(color=_MUTED, size=12),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            gridcolor=_GRID,
            zerolinecolor=_GRID,
            linecolor=_AXIS,
            tickcolor=_AXIS,
            showline=False,
            title=dict(font=dict(color=_MUTED, size=12)),
            tickfont=dict(color=_MUTED, size=11),
        ),
        yaxis=dict(
            gridcolor=_GRID,
            zerolinecolor=_GRID,
            linecolor=_AXIS,
            tickcolor=_AXIS,
            showline=False,
            title=dict(font=dict(color=_MUTED, size=12)),
            tickfont=dict(color=_MUTED, size=11),
        ),
        title=dict(font=dict(color=_TEXT, size=14), x=0, xanchor="left"),
    )


def _kpi_row(items: list[tuple[str, str, str | None]]) -> None:
    """Render a row of metric cards.

    :param items: List of ``(label, value, helper)`` tuples; ``helper`` may be
        ``None``.
    """
    cols = st.columns(len(items))
    for col, (label, value, helper) in zip(cols, items, strict=True):
        with col:
            st.metric(label=label, value=value, delta=helper)


def _hero() -> None:
    """Render the page hero with title, tagline, and dark theme styling."""
    st.markdown(
        """
        <style>
            html, body, [class*="css"]  { font-family: "Inter", "Segoe UI", system-ui, sans-serif; }
            .stApp { background: #0B1020; }
            .block-container { padding-top: 2.2rem !important; }

            .flightrisk-hero {
                position: relative;
                padding: 28px 32px;
                border-radius: 18px;
                background:
                    radial-gradient(1200px 240px at 0% 0%, rgba(96,165,250,0.18), transparent 60%),
                    radial-gradient(900px 220px at 100% 100%, rgba(52,211,153,0.16), transparent 55%),
                    linear-gradient(180deg, #131A2C 0%, #0F1525 100%);
                border: 1px solid rgba(96,165,250,0.18);
                color: #E2E8F0;
                margin-bottom: 22px;
                box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset, 0 12px 40px rgba(0,0,0,0.35);
            }
            .flightrisk-hero h1 {
                margin: 0;
                font-weight: 700;
                font-size: 30px;
                letter-spacing: -0.02em;
                background: linear-gradient(90deg, #E2E8F0 0%, #60A5FA 60%, #34D399 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .flightrisk-hero p {
                margin: 8px 0 0 0;
                color: #94A3B8;
                font-size: 14px;
                max-width: 760px;
                line-height: 1.55;
            }

            div[data-testid="stMetric"] {
                background: rgba(19, 26, 44, 0.72);
                border: 1px solid rgba(148,163,184,0.10);
                border-radius: 14px;
                padding: 16px 18px;
                box-shadow: 0 1px 0 rgba(255,255,255,0.03) inset;
                backdrop-filter: blur(8px);
                -webkit-backdrop-filter: blur(8px);
                transition: border-color .2s ease, transform .2s ease;
            }
            div[data-testid="stMetric"]:hover {
                border-color: rgba(96,165,250,0.35);
                transform: translateY(-1px);
            }
            div[data-testid="stMetricLabel"] p {
                font-size: 11px !important;
                font-weight: 600;
                color: #94A3B8 !important;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }
            div[data-testid="stMetricValue"] {
                color: #F1F5F9 !important;
                font-weight: 700;
                font-size: 28px !important;
                letter-spacing: -0.01em;
            }
            div[data-testid="stMetricDelta"] {
                color: #64748B !important;
                font-weight: 500;
                font-size: 12px !important;
            }

            section[data-testid="stSidebar"] {
                background: #0E1424 !important;
                border-right: 1px solid rgba(148,163,184,0.08);
            }
            section[data-testid="stSidebar"] * {
                color: #CBD5E1 !important;
            }
            section[data-testid="stSidebar"] code {
                color: #60A5FA !important;
                background: rgba(96,165,250,0.08) !important;
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 12px;
            }
            section[data-testid="stSidebar"] a {
                color: #60A5FA !important;
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 4px;
                border-bottom: 1px solid rgba(148,163,184,0.12);
            }
            .stTabs [data-baseweb="tab"] {
                color: #94A3B8;
                background: transparent;
                padding: 10px 14px;
                font-weight: 500;
            }
            .stTabs [aria-selected="true"] {
                color: #60A5FA !important;
                background: rgba(96,165,250,0.08);
                border-radius: 8px 8px 0 0;
            }

            .stMarkdown h3, .stMarkdown h4, .stMarkdown h5 {
                color: #E2E8F0;
                font-weight: 600;
            }
            .stCaption, .st-emotion-cache-1jicfl2 p { color: #94A3B8 !important; }

            .stSelectbox label, .stSlider label { color: #94A3B8 !important; font-weight: 500; }

            div[data-testid="stExpander"] {
                background: rgba(19,26,44,0.55);
                border: 1px solid rgba(148,163,184,0.10);
                border-radius: 12px;
            }
            div[data-testid="stExpander"] summary { color: #CBD5E1; }

            .stDataFrame, .stDataFrame * { background: transparent !important; }
        </style>
        <div class="flightrisk-hero">
            <h1>flightrisk</h1>
            <p>Identify customers on the verge before they take off &mdash; risk, survival, and uplift, scored in dollars.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_risk_tab() -> None:
    """Render the risk tab with metric cards, reliability chart, and SHAP."""
    st.subheader("Track A — Risk")
    st.caption(
        "Calibrated `P(churn)` from a LightGBM (or XGBoost) wrapper with isotonic / Platt calibration."
    )
    runs = _list_runs("risk")
    if not runs:
        st.info("No risk runs found. Run `flightrisk train risk` first.")
        return
    run_id = st.selectbox("Run id", runs, key="risk_run")

    calib = _load_calibration(run_id)
    if calib is not None and not calib.empty:
        n_total = int(calib["count"].sum())
        weighted_pred = float((calib["mean_predicted"] * calib["count"]).sum() / max(n_total, 1))
        weighted_emp = float((calib["empirical_rate"] * calib["count"]).sum() / max(n_total, 1))
        gap = weighted_emp - weighted_pred
        _kpi_row(
            [
                ("Test rows", f"{n_total:,}", None),
                ("Mean predicted", f"{weighted_pred:.3f}", None),
                ("Mean empirical", f"{weighted_emp:.3f}", f"{gap:+.3f} vs predicted"),
            ]
        )

        st.markdown("##### Reliability diagram")
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color=_DARK_PALETTE["perfect"], dash="dash", width=1.4),
                name="perfect",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=calib["mean_predicted"],
                y=calib["empirical_rate"],
                mode="markers",
                marker=dict(
                    size=np.clip(calib["count"], a_min=8, a_max=42),
                    color=_DARK_PALETTE["risk"],
                    line=dict(color="rgba(11,16,32,0.9)", width=1.6),
                    opacity=0.92,
                ),
                hovertemplate=(
                    "predicted: %{x:.3f}<br>empirical: %{y:.3f}<br>count: %{marker.size:.0f}<extra></extra>"
                ),
                name="bin",
            )
        )
        _apply_layout_template(fig)
        fig.update_layout(
            xaxis=dict(title="Mean predicted probability", range=[0, 1]),
            yaxis=dict(title="Empirical churn rate", range=[0, 1]),
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Per-bin calibration table", expanded=False):
            st.dataframe(calib.round(4), hide_index=True, use_container_width=True)

    feature_path = get_paths().data_features / "kkbox" / "features.parquet"
    if not feature_path.exists():
        st.info("Build features (`flightrisk features build`) to enable SHAP local explanations.")
        return
    sample = pd.read_parquet(feature_path).drop(columns=["msno"], errors="ignore").head(80)
    model = _load_model("risk", run_id)
    feature_names = getattr(model, "feature_names", None) or sample.columns.tolist()
    table = _shap_explanation(model, sample, feature_names).head(15).iloc[::-1]
    if table["mean_abs_shap"].notna().any():
        st.markdown("##### Top 15 features by mean |SHAP|")
        import plotly.express as px

        fig_shap = px.bar(
            table,
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            color_discrete_sequence=[_DARK_PALETTE["risk"]],
        )
        _apply_layout_template(fig_shap)
        fig_shap.update_layout(
            xaxis_title="mean |SHAP|",
            yaxis_title=None,
            height=480,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_shap, use_container_width=True)


def _render_survival_tab() -> None:
    """Render the survival tab with metric cards and per-row S(t) curves."""
    st.subheader("Track B — Survival")
    st.caption(
        "Random Survival Forest (or Cox PH) returning the full `S(t)` and per-horizon hazards."
    )
    runs = _list_runs("survival")
    if not runs:
        st.info("No survival runs found. Run `flightrisk train survival` first.")
        return
    run_id = st.selectbox("Run id", runs, key="survival_run")
    payload = _load_survival_matrix(run_id)
    if payload is None:
        st.info("S(t) matrix not found for this run.")
        return
    matrix, horizons = payload

    median_curve = np.median(matrix, axis=0)
    s_at_horizon = float(median_curve[-1])
    _kpi_row(
        [
            ("Customers in test", f"{matrix.shape[0]:,}", None),
            ("Horizons evaluated", f"{len(horizons)}", None),
            (
                f"Median S(t={int(horizons[-1])}d)",
                f"{s_at_horizon:.3f}",
                f"hazard ≈ {1 - s_at_horizon:.3f}",
            ),
        ]
    )

    n_curves = st.slider("Sampled curves", min_value=10, max_value=200, value=60)
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(matrix.shape[0], size=min(n_curves, matrix.shape[0]), replace=False)

    import plotly.graph_objects as go

    fig = go.Figure()
    for i in sample_idx:
        fig.add_trace(
            go.Scatter(
                x=horizons,
                y=matrix[i],
                mode="lines",
                line=dict(color=_DARK_PALETTE["survival"], width=1.2),
                opacity=0.18,
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=horizons,
            y=median_curve,
            mode="lines",
            line=dict(color=_DARK_PALETTE["accent"], width=2.6),
            name="cohort median",
        )
    )
    _apply_layout_template(fig)
    fig.update_layout(
        xaxis_title="Days since cutoff",
        yaxis=dict(title="S(t)", range=[0, 1.05]),
        height=440,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_uplift_tab() -> None:
    """Render the uplift tab with metric cards, Qini curve, and decile bar."""
    st.subheader("Track C — Uplift")
    st.caption("Per-customer treatment effect `τ(x)` from T-/X-/DR-learners or a CausalForestDML.")
    runs = _list_runs("uplift")
    if not runs:
        st.info("No uplift runs found. Run `flightrisk train uplift` first.")
        return
    run_id = st.selectbox("Run id", runs, key="uplift_run")
    curve = _load_qini_curve(run_id)
    if curve is None:
        st.info("Qini curve not found for this run.")
        return

    final_qini = float(curve["qini"].iloc[-1])
    n_test = len(curve)
    treated_share = (
        float(curve["treatment"].mean()) if "treatment" in curve.columns else float("nan")
    )
    _kpi_row(
        [
            ("Test rows", f"{n_test:,}", None),
            (
                "Treated share",
                f"{treated_share:.3f}" if treated_share == treated_share else "n/a",
                None,
            ),
            ("Final Qini value", f"{final_qini:,.1f}", None),
        ]
    )

    st.markdown("##### Qini curve")
    import plotly.graph_objects as go

    fig = go.Figure()
    random_line = np.linspace(0, final_qini, len(curve))
    fig.add_trace(
        go.Scatter(
            x=curve["rank"],
            y=random_line,
            mode="lines",
            name="random",
            line=dict(color=_DARK_PALETTE["random"], dash="dash", width=1.6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=curve["rank"],
            y=curve["qini"],
            mode="lines",
            name="model",
            line=dict(color=_DARK_PALETTE["uplift"], width=2.6),
            fill="tonexty",
            fillcolor="rgba(52,211,153,0.16)",
        )
    )
    _apply_layout_template(fig)
    fig.update_layout(
        xaxis_title="Ranked observations (best first)",
        yaxis_title="Incremental retained outcomes",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    if "treatment" in curve.columns and "outcome" in curve.columns:
        decile_size = max(len(curve) // 10, 1)
        rows = []
        for i in range(10):
            start = i * decile_size
            end = (i + 1) * decile_size if i < 9 else len(curve)
            slice_ = curve.iloc[start:end]
            t = slice_["treatment"] == 1
            if t.sum() == 0 or (~t).sum() == 0:
                rows.append({"decile": i + 1, "uplift": 0.0})
            else:
                rows.append(
                    {
                        "decile": i + 1,
                        "uplift": float(
                            slice_.loc[t, "outcome"].mean() - slice_.loc[~t, "outcome"].mean()
                        ),
                    }
                )
        decile_df = pd.DataFrame(rows)
        st.markdown("##### Per-decile uplift (top-1 = best by `τ̂`)")
        import plotly.express as px

        fig_dec = px.bar(
            decile_df,
            x="decile",
            y="uplift",
            color="uplift",
            color_continuous_scale=["#1E293B", _DARK_PALETTE["uplift"]],
        )
        _apply_layout_template(fig_dec)
        fig_dec.update_layout(
            xaxis=dict(title="Decile (1 = best)", tickmode="linear"),
            yaxis_title="Treated minus control outcome rate",
            coloraxis_showscale=False,
            height=380,
        )
        st.plotly_chart(fig_dec, use_container_width=True)


def _render_simulator_tab() -> None:
    """Render the campaign ROI tab with KPI cards and the headline chart."""
    st.subheader("Campaign ROI simulator")
    st.caption(
        "Same population, same budget, three policies. Bars are mean retained revenue; whiskers are 95% bootstrap CIs."
    )
    comparison_path = get_paths().reports / "simulator" / "policy_comparison.csv"
    chart_path = get_paths().reports / "simulator" / "roi_chart.png"
    if not comparison_path.exists():
        st.info("Run `flightrisk simulate` to populate this tab.")
        return
    df = pd.read_csv(comparison_path)

    largest = df.loc[df["budget"].idxmax()]
    pivot = df.pivot_table(
        index="budget", columns="policy", values="expected_revenue"
    ).reset_index()
    if {"risk", "uplift"}.issubset(pivot.columns):
        pivot["uplift_lift"] = pivot["uplift"] - pivot["risk"]
        best_row = pivot.iloc[pivot["uplift_lift"].idxmax()]
        budget_best = float(best_row["budget"])
        lift_best = float(best_row["uplift_lift"])
    else:
        budget_best, lift_best = float("nan"), float("nan")

    _kpi_row(
        [
            ("Budgets evaluated", f"{df['budget'].nunique()}", None),
            (
                "Best uplift gain",
                f"${lift_best:,.0f}" if lift_best == lift_best else "n/a",
                f"@ ${budget_best:,.0f} budget" if budget_best == budget_best else None,
            ),
            (
                "Largest budget",
                f"${largest['budget']:,.0f}",
                f"{int(largest['n_targeted']):,} targeted",
            ),
        ]
    )

    import plotly.graph_objects as go

    pretty = {"risk": "Top-k by P(churn)", "uplift": "Top-k by τ̂", "random": "Random"}
    colours = {
        "risk": _DARK_PALETTE["risk"],
        "uplift": _DARK_PALETTE["uplift"],
        "random": _DARK_PALETTE["random"],
    }
    fig = go.Figure()
    for policy in ("risk", "uplift", "random"):
        sub = df[df["policy"] == policy].sort_values("budget")
        if sub.empty:
            continue
        upper = sub["ci_upper"] - sub["expected_revenue"]
        lower = sub["expected_revenue"] - sub["ci_lower"]
        fig.add_trace(
            go.Bar(
                x=[f"${int(b):,}" for b in sub["budget"]],
                y=sub["expected_revenue"],
                name=pretty.get(policy, policy),
                marker_color=colours.get(policy, _DARK_PALETTE["muted"]),
                error_y=dict(
                    type="data",
                    array=upper.clip(lower=0),
                    arrayminus=lower.clip(lower=0),
                    color=_AXIS,
                    thickness=1.4,
                ),
                text=[f"${v / 1000:,.0f}k" for v in sub["expected_revenue"]],
                textposition="outside",
                textfont=dict(color=_TEXT, size=11),
                marker_line_color="rgba(11,16,32,0.6)",
                marker_line_width=1.0,
                hovertemplate=(
                    "<b>%{x}</b><br>%{fullData.name}<br>revenue: %{y:$,.0f}<extra></extra>"
                ),
            )
        )
    _apply_layout_template(fig)
    fig.update_layout(
        barmode="group",
        bargap=0.18,
        xaxis_title="Campaign budget",
        yaxis_title="Expected retained revenue",
        height=460,
        yaxis=dict(tickformat="$,.0f", gridcolor=_GRID),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Comparison table", expanded=False):
        st.dataframe(
            df.assign(policy=df["policy"].map(pretty)).round(2),
            hide_index=True,
            use_container_width=True,
        )

    if chart_path.exists():
        with st.expander("Static export (matplotlib version)", expanded=False):
            st.image(str(chart_path), caption="reports/simulator/roi_chart.png")


def _sidebar() -> None:
    """Render the static sidebar with project metadata and quick links."""
    paths = get_paths()
    runs = {
        "Risk": _list_runs("risk"),
        "Survival": _list_runs("survival"),
        "Uplift": _list_runs("uplift"),
    }
    st.sidebar.markdown("### Repo state")
    st.sidebar.write(f"**Root:** `{paths.root}`")
    st.sidebar.write(f"**Tracks with runs:** {sum(1 for v in runs.values() if v)} / 3")
    for track, run_list in runs.items():
        if run_list:
            st.sidebar.write(f"- {track}: latest `{run_list[0][:12]}` ({len(run_list)} total)")
        else:
            st.sidebar.write(f"- {track}: _no runs yet_")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Links**\n\n"
        "- [Repo](https://github.com/camilo-acevedo/flightrisk)\n"
        "- [Docs](https://camilo-acevedo.github.io/flightrisk/)\n"
        "- [Walkthrough notebook](https://github.com/camilo-acevedo/flightrisk/blob/main/notebooks/00_walkthrough.ipynb)"
    )


def main() -> None:
    """Streamlit app entry point.

    Configures the page, paints the hero, and renders one tab per track plus
    the simulator.
    """
    st.set_page_config(
        page_title="flightrisk demo",
        page_icon="🛬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _hero()
    _sidebar()
    risk_tab, survival_tab, uplift_tab, sim_tab = st.tabs(
        ["Risk", "Survival", "Uplift", "ROI simulator"]
    )
    with risk_tab:
        _render_risk_tab()
    with survival_tab:
        _render_survival_tab()
    with uplift_tab:
        _render_uplift_tab()
    with sim_tab:
        _render_simulator_tab()


if __name__ == "__main__":
    main()
