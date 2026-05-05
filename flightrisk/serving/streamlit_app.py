from __future__ import annotations

from collections.abc import Iterable

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from flightrisk.utils.paths import get_paths


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
    try:
        import shap

        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(sample[list(feature_names)])
        if isinstance(values, list):
            values = values[1] if len(values) == 2 else values[-1]
        importance = np.abs(values).mean(axis=0)
        return pd.DataFrame(
            {"feature": list(feature_names), "mean_abs_shap": importance}
        ).sort_values("mean_abs_shap", ascending=False)
    except Exception as exc:
        return pd.DataFrame(
            {
                "feature": list(feature_names),
                "mean_abs_shap": [np.nan] * len(list(feature_names)),
                "error": [str(exc)] * len(list(feature_names)),
            }
        )


def _render_risk_tab() -> None:
    """Render the risk tab with metrics, calibration plot, and SHAP."""
    st.header("Risk track (Track A)")
    runs = _list_runs("risk")
    if not runs:
        st.info("No risk runs found. Run `flightrisk train risk` first.")
        return
    run_id = st.selectbox("Run id", runs, key="risk_run")
    calib = _load_calibration(run_id)
    if calib is not None:
        st.subheader("Reliability diagram")
        st.scatter_chart(calib.set_index("mean_predicted")["empirical_rate"])
        st.dataframe(calib, hide_index=True, use_container_width=True)

    feature_path = get_paths().data_features / "kkbox" / "features.parquet"
    if not feature_path.exists():
        st.info("Build features (`flightrisk features build`) to enable SHAP local explanations.")
        return
    sample = pd.read_parquet(feature_path).drop(columns=["msno"], errors="ignore").head(50)
    model = _load_model("risk", run_id)
    feature_names = getattr(model, "feature_names", None) or sample.columns.tolist()
    st.subheader("Top features by mean |SHAP|")
    table = _shap_explanation(model, sample, feature_names)
    st.dataframe(table.head(15), hide_index=True, use_container_width=True)


def _render_survival_tab() -> None:
    """Render the survival tab with sampled S(t) curves."""
    st.header("Survival track (Track B)")
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
    n_curves = st.slider("Curves to plot", min_value=10, max_value=200, value=50)
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(matrix.shape[0], size=min(n_curves, matrix.shape[0]), replace=False)
    chart_df = pd.DataFrame(matrix[sample_idx].T, index=horizons.astype(int))
    st.line_chart(chart_df)
    st.caption("Each line is one customer's predicted S(t).")


def _render_uplift_tab() -> None:
    """Render the uplift tab with the Qini curve and decile bar."""
    st.header("Uplift track (Track C)")
    runs = _list_runs("uplift")
    if not runs:
        st.info("No uplift runs found. Run `flightrisk train uplift` first.")
        return
    run_id = st.selectbox("Run id", runs, key="uplift_run")
    curve = _load_qini_curve(run_id)
    if curve is None:
        st.info("Qini curve not found for this run.")
        return
    st.subheader("Qini curve")
    st.line_chart(curve.set_index("rank")["qini"])

    decile_size = max(len(curve) // 10, 1)
    deciles = []
    for i in range(10):
        start = i * decile_size
        end = (i + 1) * decile_size if i < 9 else len(curve)
        slice_ = curve.iloc[start:end]
        treated = (slice_["treatment"] == 1) if "treatment" in slice_.columns else None
        outcome = slice_.get("outcome")
        if treated is None or outcome is None or treated.sum() == 0 or (~treated).sum() == 0:
            deciles.append({"decile": i + 1, "uplift": np.nan})
            continue
        deciles.append(
            {
                "decile": i + 1,
                "uplift": outcome[treated].mean() - outcome[~treated].mean(),
            }
        )
    decile_df = pd.DataFrame(deciles)
    st.subheader("Per-decile uplift")
    st.bar_chart(decile_df.set_index("decile"))


def _render_simulator_tab() -> None:
    """Render the campaign ROI tab from the persisted comparison CSV."""
    st.header("Campaign ROI simulator")
    comparison_path = get_paths().reports / "simulator" / "policy_comparison.csv"
    chart_path = get_paths().reports / "simulator" / "roi_chart.png"
    if not comparison_path.exists():
        st.info("Run `flightrisk simulate` to populate this tab.")
        return
    df = pd.read_csv(comparison_path)
    st.dataframe(df, hide_index=True, use_container_width=True)
    if chart_path.exists():
        st.image(str(chart_path), caption="Headline ROI chart")


def main() -> None:
    """Streamlit app entry point.

    Configures the page and renders one tab per track plus the simulator.
    """
    st.set_page_config(page_title="flightrisk demo", layout="wide")
    st.title("flightrisk")
    st.caption("Identify customers on the verge before they take off.")
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
