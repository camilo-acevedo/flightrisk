from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from flightrisk.models.risk.tune import SweepSpace, load_sweep_space, run_risk_sweep


def _synthetic(n: int = 800, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    f0 = rng.normal(size=n)
    f1 = rng.normal(size=n)
    logits = 0.7 * f0 - 0.4 * f1 - 0.5
    y = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-logits))).astype(int)
    features = pd.DataFrame({"msno": [f"u{i}" for i in range(n)], "f0": f0, "f1": f1})
    return features, y


def test_load_sweep_space_parses_yaml(tmp_path: Path) -> None:
    yml = tmp_path / "sweep.yaml"
    yml.write_text(
        "name: t\n"
        "direction: maximize\n"
        "metric: auc\n"
        "n_trials: 3\n"
        "search_space:\n"
        "  learning_rate:\n"
        "    type: float\n"
        "    low: 0.01\n"
        "    high: 0.2\n"
        "    log: true\n"
        "  num_leaves:\n"
        "    type: int\n"
        "    low: 15\n"
        "    high: 63\n"
    )
    space = load_sweep_space(yml)
    assert space.name == "t"
    assert space.metric == "auc"
    assert space.n_trials == 3
    assert "learning_rate" in space.search_space


def test_run_risk_sweep_returns_best_params() -> None:
    features, y = _synthetic()
    n = len(features)
    train_idx = np.arange(0, int(0.7 * n))
    val_idx = np.arange(int(0.7 * n), n)

    space = SweepSpace(
        name="test",
        direction="maximize",
        metric="auc",
        n_trials=3,
        timeout_seconds=None,
        search_space={
            "learning_rate": {"type": "float", "low": 0.05, "high": 0.1, "log": False},
            "num_leaves": {"type": "int", "low": 15, "high": 31},
            "n_estimators": {"type": "int", "low": 50, "high": 100},
        },
    )
    result = run_risk_sweep(features, y, train_idx=train_idx, val_idx=val_idx, space=space)
    assert "learning_rate" in result.best_params
    assert 0.0 < result.best_metric <= 1.0
    assert result.study.trials and len(result.study.trials) == 3
