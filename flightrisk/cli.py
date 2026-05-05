from __future__ import annotations

import click

from flightrisk.utils.logging import get_logger

_log = get_logger(__name__)


@click.group(help="flightrisk command-line interface.")
def main() -> None:
    """Top-level CLI group; subcommands attach below."""


@main.group("data", help="Dataset ingestion and validation.")
def data_group() -> None:
    """Data subcommands."""


@data_group.command("pull", help="Pull raw datasets via DVC, with Kaggle fallback for KKBox.")
@click.option(
    "--kaggle-fallback/--no-kaggle-fallback",
    default=True,
    help="If DVC is unavailable, attempt to pull KKBox via the Kaggle CLI.",
)
def data_pull(kaggle_fallback: bool) -> None:
    """Materialise the raw data layer.

    Tries ``dvc pull`` first; on failure or absence of DVC, falls back to the
    Kaggle CLI for the KKBox bundle if ``--kaggle-fallback`` is enabled.

    :param kaggle_fallback: Whether to try Kaggle when DVC is unavailable.
    """
    from flightrisk.data import ingest

    try:
        rc = ingest.dvc_pull()
        if rc == 0:
            _log.info("dvc pull completed successfully")
            return
        _log.warning("dvc pull returned non-zero exit code %s", rc)
    except RuntimeError as exc:
        _log.warning("%s", exc)

    if kaggle_fallback:
        _log.info("attempting Kaggle fallback for KKBox")
        ingest.kaggle_download_kkbox()
        _log.info("kaggle download complete; you may need to unzip downloaded archives")
    else:
        _log.error("data pull failed and Kaggle fallback is disabled")
        raise click.exceptions.Exit(1)


@data_group.command("validate", help="Load raw datasets and validate against pandera schemas.")
@click.option("--sample-frac", type=float, default=None, help="Optional KKBox subsample fraction.")
def data_validate(sample_frac: float | None) -> None:
    """Run schema validation against the raw data layer.

    :param sample_frac: If provided, sub-sample KKBox by user ID for speed.
    """
    from flightrisk.data.loaders import load_kkbox, load_orange_belgium

    kk = load_kkbox(sample_frac=sample_frac)
    _log.info(
        "kkbox: members=%d transactions=%d user_logs=%d labels=%d",
        len(kk.members),
        len(kk.transactions),
        len(kk.user_logs),
        len(kk.labels),
    )
    try:
        orange = load_orange_belgium()
        _log.info(
            "orange-belgium: rows=%d features=%d treated_share=%.3f",
            len(orange.features),
            orange.features.shape[1],
            float(orange.treatment.mean()),
        )
    except FileNotFoundError as exc:
        _log.warning("orange-belgium not available: %s", exc)


@main.group("features", help="Feature engineering pipelines.")
def features_group() -> None:
    """Feature subcommands."""


@features_group.command("build", help="Build feature matrices for KKBox and Orange Belgium.")
@click.option("--cutoff", default="2017-02-28", show_default=True, help="KKBox build cutoff date.")
@click.option("--sample-frac", type=float, default=None, help="KKBox subsample fraction.")
@click.option(
    "--skip-orange/--with-orange",
    default=False,
    help="Skip the Orange Belgium pipeline if its raw file is missing.",
)
def features_build(cutoff: str, sample_frac: float | None, skip_orange: bool) -> None:
    """Build and persist feature bundles for both datasets.

    :param cutoff: KKBox build cutoff (any pandas-parsable date string).
    :param sample_frac: Optional subsample fraction for KKBox.
    :param skip_orange: When set, do not attempt to load Orange Belgium.
    """
    from flightrisk.data.loaders import load_kkbox, load_orange_belgium
    from flightrisk.features.pipeline import (
        build_kkbox_bundle,
        build_orange_bundle,
        write_kkbox_bundle,
        write_orange_bundle,
    )

    kk = load_kkbox(sample_frac=sample_frac)
    bundle = build_kkbox_bundle(kk, cutoff=cutoff)
    write_kkbox_bundle(bundle)

    if skip_orange:
        _log.info("skipping orange-belgium pipeline as requested")
        return
    try:
        orange = load_orange_belgium()
    except FileNotFoundError as exc:
        _log.warning("orange-belgium not available: %s", exc)
        return
    write_orange_bundle(build_orange_bundle(orange))


@main.group("train", help="Model training tracks.")
def train_group() -> None:
    """Training subcommands."""


@train_group.command("risk", help="Train Track A (classical risk) on KKBox features.")
@click.option(
    "--estimator",
    type=click.Choice(["lightgbm", "xgboost"]),
    default="lightgbm",
    show_default=True,
)
@click.option(
    "--calibration",
    type=click.Choice(["isotonic", "platt", "none"]),
    default="isotonic",
    show_default=True,
)
@click.option("--train-cutoff", default="2017-01-31", show_default=True)
@click.option("--val-cutoff", default="2017-02-15", show_default=True)
@click.option("--seed", type=int, default=1337, show_default=True)
def train_risk(
    estimator: str, calibration: str, train_cutoff: str, val_cutoff: str, seed: int
) -> None:
    """Train Track A and log artifacts to MLflow.

    :param estimator: Either ``lightgbm`` or ``xgboost``.
    :param calibration: ``isotonic``, ``platt``, or ``none`` to skip.
    :param train_cutoff: Last date kept in the train slice.
    :param val_cutoff: Last date kept in the validation slice.
    :param seed: Reproducibility seed.
    """
    import joblib
    import mlflow
    import pandas as pd

    from flightrisk.data.splits import temporal_split
    from flightrisk.models.risk.trainer import save_calibration_plot, train_risk_model
    from flightrisk.utils import seed_everything
    from flightrisk.utils.mlflow_helpers import configure_mlflow, start_run
    from flightrisk.utils.paths import get_paths

    seed_everything(seed)
    paths = get_paths()
    base = paths.data_features / "kkbox"
    features = pd.read_parquet(base / "features.parquet")
    labels = pd.read_parquet(base / "labels.parquet")
    cutoff = pd.Timestamp((base / "cutoff.txt").read_text().strip())

    pseudo_dates = pd.Series(
        pd.to_datetime(features["msno"].astype(str), errors="coerce").fillna(cutoff)
    )
    if pseudo_dates.is_monotonic_increasing is False:
        pseudo_dates = pd.Series(
            pd.date_range(end=cutoff, periods=len(features), freq="D")
        )

    split = temporal_split(
        pseudo_dates,
        train_cutoff=train_cutoff,
        val_cutoff=val_cutoff,
    )

    calibration_arg = None if calibration == "none" else calibration
    configure_mlflow()
    with start_run(run_name=f"risk-{estimator}") as run:
        mlflow.log_params(
            {
                "estimator": estimator,
                "calibration": calibration,
                "seed": seed,
                "train_cutoff": train_cutoff,
                "val_cutoff": val_cutoff,
                "n_features": features.shape[1] - 1,
            }
        )
        result = train_risk_model(
            features,
            labels["is_churn"].values,
            train_idx=split.train_idx,
            val_idx=split.val_idx,
            test_idx=split.test_idx,
            estimator=estimator,
            calibration=calibration_arg,
        )
        mlflow.log_metrics(dict(result.metrics.as_dict()))

        out_dir = paths.reports / "risk" / run.info.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        save_calibration_plot(result.calibration, output=out_dir / "calibration.png")
        result.calibration.to_csv(out_dir / "calibration.csv", index=False)
        joblib.dump(result.model, out_dir / "model.joblib")
        mlflow.log_artifacts(str(out_dir))
        _log.info("risk artifacts saved to %s", out_dir)


@train_group.command("survival", help="Train Track B (survival) on KKBox features.")
@click.option(
    "--estimator",
    type=click.Choice(["cox", "rsf"]),
    default="rsf",
    show_default=True,
)
@click.option("--horizon-days", type=int, default=90, show_default=True)
@click.option("--train-frac", type=float, default=0.8, show_default=True)
@click.option("--seed", type=int, default=1337, show_default=True)
def train_survival(estimator: str, horizon_days: int, train_frac: float, seed: int) -> None:
    """Train Track B and log artifacts to MLflow.

    :param estimator: ``cox`` or ``rsf``.
    :param horizon_days: Maximum follow-up window in days.
    :param train_frac: Fraction of rows reserved for training.
    :param seed: Reproducibility seed.
    """
    import joblib
    import mlflow
    import numpy as np
    import pandas as pd

    from flightrisk.data.loaders import load_kkbox
    from flightrisk.models.survival.labels import build_survival_labels
    from flightrisk.models.survival.trainer import (
        save_survival_curve_plot,
        train_survival_model,
    )
    from flightrisk.utils import seed_everything
    from flightrisk.utils.mlflow_helpers import configure_mlflow, start_run
    from flightrisk.utils.paths import get_paths

    seed_everything(seed)
    paths = get_paths()
    base = paths.data_features / "kkbox"
    features = pd.read_parquet(base / "features.parquet")
    cutoff = pd.Timestamp((base / "cutoff.txt").read_text().strip())

    _log.info("rebuilding survival labels at cutoff=%s, horizon=%d", cutoff.date(), horizon_days)
    raw = load_kkbox()
    labels = build_survival_labels(
        raw.transactions, cutoff=cutoff, horizon_days=horizon_days
    ).to_frame()

    aligned = features.merge(labels, on="msno", how="inner")
    feat_cols = [c for c in features.columns if c != "msno"]
    feature_frame = aligned[["msno"] + feat_cols]
    durations = aligned["duration_days"].values
    events = aligned["event_observed"].values

    n = len(aligned)
    perm = np.random.default_rng(seed).permutation(n)
    cut = int(n * train_frac)
    train_idx = perm[:cut]
    test_idx = perm[cut:]

    horizons = (max(horizon_days // 3, 7), max(2 * horizon_days // 3, 14), horizon_days)

    configure_mlflow()
    with start_run(run_name=f"survival-{estimator}") as run:
        mlflow.log_params(
            {
                "estimator": estimator,
                "horizon_days": horizon_days,
                "n_samples": n,
                "train_frac": train_frac,
                "seed": seed,
                "n_features": len(feat_cols),
            }
        )
        result = train_survival_model(
            feature_frame,
            durations,
            events,
            train_idx=train_idx,
            test_idx=test_idx,
            estimator=estimator,
            horizons_days=horizons,
        )
        mlflow.log_metrics(dict(result.metrics.as_dict()))

        out_dir = paths.reports / "survival" / run.info.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        save_survival_curve_plot(
            result.survival_at_horizons, result.horizons_days, output=out_dir / "survival_curves.png"
        )
        joblib.dump(result.model, out_dir / "model.joblib")
        np.save(out_dir / "survival_at_horizons.npy", result.survival_at_horizons)
        mlflow.log_artifacts(str(out_dir))
        _log.info("survival artifacts saved to %s", out_dir)


@train_group.command("uplift")
def train_uplift() -> None:
    """Train Track C (uplift). Lands in step 5."""
    _log.info("train uplift is not yet implemented; coming in step 5.")


@main.command("simulate", help="Run the campaign ROI simulator.")
def simulate() -> None:
    """Run the campaign ROI simulator. Lands in step 6."""
    _log.info("simulate is not yet implemented; coming in step 6.")


if __name__ == "__main__":
    main()
