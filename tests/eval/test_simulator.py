from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flightrisk.eval.plots import save_roi_chart
from flightrisk.eval.simulator import (
    SimulatorConfig,
    compare_policies,
    simulate_random_policy,
    simulate_risk_policy,
    simulate_uplift_policy,
)


def _scenario(n: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    risk = rng.uniform(0, 1, size=n)
    base = 1.0 - risk
    uplift = np.clip(rng.uniform(0, 0.3, size=n) + 0.4 * (risk > 0.7), 0.0, 1.0)
    lift = np.clip(0.05 + 0.4 * (risk > 0.7) * (uplift > 0.4), 0.0, 0.6)
    return risk, uplift, base, lift


def test_simulate_risk_policy_respects_budget() -> None:
    risk, _, base, lift = _scenario()
    cfg = SimulatorConfig(cost_per_treated=10.0, bootstrap_iters=50)
    out = simulate_risk_policy(
        risk_scores=risk, treatment_lift=lift, base_outcome=base, budget=200.0, config=cfg
    )
    assert out.n_targeted == 20
    assert out.policy == "risk"
    assert out.expected_revenue >= out.ci_lower
    assert out.expected_revenue <= out.ci_upper or abs(out.ci_upper - out.expected_revenue) < 1e-6


def test_simulate_uplift_policy_targets_high_lift_customers() -> None:
    risk, uplift, base, lift = _scenario()
    cfg = SimulatorConfig(cost_per_treated=10.0, bootstrap_iters=50)
    risk_result = simulate_risk_policy(
        risk_scores=risk, treatment_lift=lift, base_outcome=base, budget=2000.0, config=cfg
    )
    uplift_result = simulate_uplift_policy(
        uplift_scores=uplift, treatment_lift=lift, base_outcome=base, budget=2000.0, config=cfg
    )
    assert uplift_result.expected_retained >= risk_result.expected_retained


def test_simulate_random_policy_produces_finite_revenue() -> None:
    _, _, base, lift = _scenario()
    cfg = SimulatorConfig(cost_per_treated=10.0, bootstrap_iters=50)
    out = simulate_random_policy(
        n=len(base), treatment_lift=lift, base_outcome=base, budget=500.0, config=cfg
    )
    assert np.isfinite(out.expected_revenue)
    assert out.policy == "random"


def test_compare_policies_returns_one_row_per_combo() -> None:
    risk, uplift, base, lift = _scenario(n=600)
    cfg = SimulatorConfig(cost_per_treated=10.0, bootstrap_iters=20)
    table = compare_policies(
        risk_scores=risk,
        uplift_scores=uplift,
        treatment_lift=lift,
        base_outcome=base,
        budgets=(500.0, 1000.0),
        config=cfg,
    )
    assert len(table) == 6
    assert set(table["policy"]) == {"risk", "uplift", "random"}
    assert set(table["budget"]) == {500.0, 1000.0}


def test_simulator_validates_cost(tmp_path: Path) -> None:
    risk, _, base, lift = _scenario(n=200)
    with pytest.raises(ValueError):
        simulate_risk_policy(
            risk_scores=risk,
            treatment_lift=lift,
            base_outcome=base,
            budget=100.0,
            config=SimulatorConfig(cost_per_treated=0.0),
        )


def test_save_roi_chart_writes_png(tmp_path: Path) -> None:
    risk, uplift, base, lift = _scenario(n=400)
    cfg = SimulatorConfig(cost_per_treated=10.0, bootstrap_iters=20)
    table = compare_policies(
        risk_scores=risk,
        uplift_scores=uplift,
        treatment_lift=lift,
        base_outcome=base,
        budgets=(500.0,),
        config=cfg,
    )
    output = tmp_path / "chart.png"
    save_roi_chart(table, output=output)
    assert output.exists() and output.stat().st_size > 0
