from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from flightrisk.data.loaders import KKBoxArtifacts, OrangeBelgiumArtifacts
from flightrisk.features.kkbox import KKBoxFeatureConfig, build_kkbox_feature_matrix
from flightrisk.features.orange import OrangeFeatureConfig, prepare_orange_features
from flightrisk.utils.logging import get_logger
from flightrisk.utils.paths import get_paths

_log = get_logger(__name__)


@dataclass(frozen=True)
class KKBoxFeatureBundle:
    """Output of the KKBox feature pipeline.

    :param features: Wide feature matrix keyed by ``msno``.
    :param labels: Churn labels aligned to ``features`` by ``msno``.
    :param cutoff: Build cutoff used to assemble the matrix.
    """

    features: pd.DataFrame
    labels: pd.DataFrame
    cutoff: pd.Timestamp


def build_kkbox_bundle(
    artifacts: KKBoxArtifacts,
    *,
    cutoff: str | pd.Timestamp,
    config: KKBoxFeatureConfig | None = None,
) -> KKBoxFeatureBundle:
    """Run the full KKBox feature pipeline and align it with labels.

    :param artifacts: Validated raw frames from
        :func:`flightrisk.data.loaders.load_kkbox`.
    :param cutoff: Build cutoff (string or :class:`pandas.Timestamp`).
    :param config: Optional feature configuration.
    :returns: A :class:`KKBoxFeatureBundle` ready for modeling.
    """
    cutoff_ts = pd.Timestamp(cutoff)
    cfg = config or KKBoxFeatureConfig()
    _log.info("building kkbox feature matrix at cutoff=%s", cutoff_ts.date())
    feature_matrix = build_kkbox_feature_matrix(
        artifacts.members,
        artifacts.transactions,
        artifacts.user_logs,
        cutoff=cutoff_ts,
        config=cfg,
    )
    labels = artifacts.labels[["msno", "is_churn"]].copy()
    aligned = feature_matrix.merge(labels, on="msno", how="inner")
    features = aligned.drop(columns=["is_churn"])
    aligned_labels = aligned[["msno", "is_churn"]]
    _log.info(
        "kkbox features: %d rows, %d feature columns",
        len(features),
        features.shape[1] - 1,
    )
    return KKBoxFeatureBundle(features=features, labels=aligned_labels, cutoff=cutoff_ts)


@dataclass(frozen=True)
class OrangeFeatureBundle:
    """Output of the Orange Belgium feature pipeline.

    :param features: Cleaned feature frame.
    :param treatment: 0/1 treatment column from the original RCT.
    :param outcome: 0/1 outcome column.
    """

    features: pd.DataFrame
    treatment: pd.Series
    outcome: pd.Series


def build_orange_bundle(
    artifacts: OrangeBelgiumArtifacts,
    *,
    config: OrangeFeatureConfig | None = None,
) -> OrangeFeatureBundle:
    """Prepare the Orange Belgium dataset for uplift modelling.

    :param artifacts: Validated raw frame from
        :func:`flightrisk.data.loaders.load_orange_belgium`.
    :param config: Optional feature configuration.
    :returns: An :class:`OrangeFeatureBundle`.
    """
    cfg = config or OrangeFeatureConfig()
    _log.info("preparing orange-belgium features")
    features = prepare_orange_features(artifacts.features, config=cfg)
    return OrangeFeatureBundle(
        features=features,
        treatment=artifacts.treatment,
        outcome=artifacts.outcome,
    )


def write_kkbox_bundle(bundle: KKBoxFeatureBundle, *, name: str = "kkbox") -> Path:
    """Persist a KKBox bundle to ``data/features/{name}/``.

    :param bundle: Bundle returned by :func:`build_kkbox_bundle`.
    :param name: Subdirectory name under ``data/features``.
    :returns: The directory the bundle was written to.
    """
    base = get_paths().data_features / name
    base.mkdir(parents=True, exist_ok=True)
    bundle.features.to_parquet(base / "features.parquet", index=False)
    bundle.labels.to_parquet(base / "labels.parquet", index=False)
    (base / "cutoff.txt").write_text(bundle.cutoff.isoformat())
    _log.info("wrote kkbox bundle to %s", base)
    return base


def write_orange_bundle(bundle: OrangeFeatureBundle, *, name: str = "orange") -> Path:
    """Persist an Orange Belgium bundle to ``data/features/{name}/``.

    :param bundle: Bundle returned by :func:`build_orange_bundle`.
    :param name: Subdirectory name under ``data/features``.
    :returns: The directory the bundle was written to.
    """
    base = get_paths().data_features / name
    base.mkdir(parents=True, exist_ok=True)
    bundle.features.to_parquet(base / "features.parquet", index=False)
    pd.DataFrame(
        {
            "treatment": bundle.treatment.reset_index(drop=True),
            "outcome": bundle.outcome.reset_index(drop=True),
        }
    ).to_parquet(base / "labels.parquet", index=False)
    _log.info("wrote orange bundle to %s", base)
    return base
