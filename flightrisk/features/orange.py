from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OrangeFeatureConfig:
    """Configuration for Orange Belgium feature post-processing.

    :param drop_low_variance_threshold: Minimum variance to keep a column. Set
        to 0 to disable filtering.
    :param fill_strategy: How to fill missing numeric values: ``"median"`` or
        ``"zero"``.
    """

    drop_low_variance_threshold: float = 0.0
    fill_strategy: str = "median"


def prepare_orange_features(
    features: pd.DataFrame,
    *,
    config: OrangeFeatureConfig | None = None,
) -> pd.DataFrame:
    """Prepare the 178 Orange Belgium features for downstream models.

    The dataset already provides anonymised, model-ready columns, so the
    pipeline is intentionally minimal: drop low-variance columns and impute
    missing values to keep tree models happy.

    :param features: Raw feature frame from
        :func:`flightrisk.data.loaders.load_orange_belgium`.
    :param config: Optional feature configuration.
    :returns: A new frame ready for fitting. The original is not modified.
    :raises ValueError: If ``fill_strategy`` is unknown.
    """
    cfg = config or OrangeFeatureConfig()
    out = features.copy()

    numeric = out.select_dtypes(include=[np.number])
    if cfg.drop_low_variance_threshold > 0.0:
        variances = numeric.var(numeric_only=True)
        keep = variances[variances > cfg.drop_low_variance_threshold].index.tolist()
        non_numeric = [c for c in out.columns if c not in numeric.columns]
        out = out[non_numeric + keep]

    numeric = out.select_dtypes(include=[np.number])
    if cfg.fill_strategy == "median":
        fill_values = numeric.median(numeric_only=True)
        out[numeric.columns] = numeric.fillna(fill_values)
    elif cfg.fill_strategy == "zero":
        out[numeric.columns] = numeric.fillna(0.0)
    else:
        raise ValueError(f"unknown fill_strategy: {cfg.fill_strategy!r}")

    return out
