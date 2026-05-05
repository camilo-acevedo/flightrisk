from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulatorConfig:
    """Configuration for the campaign ROI simulator.

    :param cost_per_treated: Marginal cost of treating one customer.
    :param revenue_per_retained: Revenue captured by retaining one customer.
    :param bootstrap_iters: Number of bootstrap resamples for confidence bands.
    :param alpha: Two-sided significance level for the confidence interval.
    :param random_state: Seed used by the bootstrap resampler.
    """

    cost_per_treated: float = 5.0
    revenue_per_retained: float = 50.0
    bootstrap_iters: int = 1000
    alpha: float = 0.05
    random_state: int = 1337


@dataclass(frozen=True)
class SimulationResult:
    """Outcome of a budgeted targeting policy simulation.

    :param budget: Budget cap in dollars.
    :param n_targeted: Number of customers actually treated under the policy.
    :param expected_retained: Mean retained customers across bootstrap samples.
    :param expected_revenue: Mean retained revenue net of treatment cost.
    :param ci_lower: Lower bound of the bootstrap CI for revenue.
    :param ci_upper: Upper bound of the bootstrap CI for revenue.
    :param policy: Identifier (``"risk"``, ``"uplift"``, ``"random"``).
    """

    budget: float
    n_targeted: int
    expected_retained: float
    expected_revenue: float
    ci_lower: float
    ci_upper: float
    policy: str

    def as_dict(self) -> Mapping[str, float | int | str]:
        """Return the result as a flat dict suitable for MLflow.

        :returns: Mapping of fields to values.
        """
        return {
            "budget": float(self.budget),
            "n_targeted": int(self.n_targeted),
            "expected_retained": float(self.expected_retained),
            "expected_revenue": float(self.expected_revenue),
            "ci_lower": float(self.ci_lower),
            "ci_upper": float(self.ci_upper),
            "policy": self.policy,
        }


def _sort_by_score(scores: np.ndarray, *, descending: bool = True) -> np.ndarray:
    """Stable sort of indices by score.

    :param scores: 1-D numeric array.
    :param descending: If True, largest scores come first.
    :returns: Array of indices.
    """
    order = np.argsort(scores, kind="stable")
    return order[::-1] if descending else order


def _affordable(n: int, *, budget: float, cost_per_treated: float) -> int:
    """Return how many customers fit under ``budget`` at ``cost_per_treated``.

    :param n: Population size.
    :param budget: Budget cap in dollars.
    :param cost_per_treated: Cost per treated customer.
    :returns: Affordable count, never larger than ``n``.
    :raises ValueError: If ``cost_per_treated`` is non-positive.
    """
    if cost_per_treated <= 0:
        raise ValueError("cost_per_treated must be positive")
    return min(n, int(budget // cost_per_treated))


def simulate_risk_policy(
    *,
    risk_scores: np.ndarray,
    treatment_lift: np.ndarray,
    base_outcome: np.ndarray,
    budget: float,
    config: SimulatorConfig | None = None,
) -> SimulationResult:
    """Simulate a top-by-P(churn) campaign at the given ``budget``.

    The policy treats the top-``k`` customers by ``risk_scores`` where ``k`` is
    determined by ``budget / cost_per_treated``. Each treated customer's
    retention probability is increased by ``treatment_lift`` (clipped to
    ``[0, 1 - base_outcome]``); each untreated customer is assumed to retain at
    ``base_outcome``.

    :param risk_scores: Per-row predicted churn probability (Track A output).
    :param treatment_lift: Per-row uplift in retention if treated (e.g. RCT
        average lift; can be a constant repeated).
    :param base_outcome: Baseline retention probability per customer.
    :param budget: Total dollars available.
    :param config: Optional simulator configuration.
    :returns: A :class:`SimulationResult` with bootstrapped CI bands.
    """
    return _simulate_policy(
        scores=risk_scores,
        treatment_lift=treatment_lift,
        base_outcome=base_outcome,
        budget=budget,
        config=config,
        policy_name="risk",
    )


def simulate_uplift_policy(
    *,
    uplift_scores: np.ndarray,
    treatment_lift: np.ndarray,
    base_outcome: np.ndarray,
    budget: float,
    config: SimulatorConfig | None = None,
) -> SimulationResult:
    """Simulate a top-by-uplift campaign at the given ``budget``.

    The policy treats the top-``k`` customers by ``uplift_scores`` (Track C
    output).

    :param uplift_scores: Per-row uplift estimate.
    :param treatment_lift: Per-row uplift in retention if treated.
    :param base_outcome: Baseline retention probability per customer.
    :param budget: Total dollars available.
    :param config: Optional simulator configuration.
    :returns: A :class:`SimulationResult` with bootstrapped CI bands.
    """
    return _simulate_policy(
        scores=uplift_scores,
        treatment_lift=treatment_lift,
        base_outcome=base_outcome,
        budget=budget,
        config=config,
        policy_name="uplift",
    )


def simulate_random_policy(
    *,
    n: int,
    treatment_lift: np.ndarray,
    base_outcome: np.ndarray,
    budget: float,
    config: SimulatorConfig | None = None,
) -> SimulationResult:
    """Simulate a uniformly random targeting policy at the given ``budget``.

    :param n: Population size.
    :param treatment_lift: Per-row uplift in retention if treated.
    :param base_outcome: Baseline retention probability per customer.
    :param budget: Total dollars available.
    :param config: Optional simulator configuration.
    :returns: A :class:`SimulationResult` with bootstrapped CI bands.
    """
    cfg = config or SimulatorConfig()
    rng = np.random.default_rng(cfg.random_state)
    random_scores = rng.uniform(0, 1, size=n)
    return _simulate_policy(
        scores=random_scores,
        treatment_lift=treatment_lift,
        base_outcome=base_outcome,
        budget=budget,
        config=cfg,
        policy_name="random",
    )


def _simulate_policy(
    *,
    scores: np.ndarray,
    treatment_lift: np.ndarray,
    base_outcome: np.ndarray,
    budget: float,
    config: SimulatorConfig | None,
    policy_name: str,
) -> SimulationResult:
    """Shared simulation core for the three policy variants.

    :param scores: Per-row policy score (higher = treated first).
    :param treatment_lift: Per-row retention lift if treated.
    :param base_outcome: Baseline retention probability per customer.
    :param budget: Budget cap in dollars.
    :param config: Optional simulator configuration.
    :param policy_name: Tag stored on the result.
    :returns: A :class:`SimulationResult` bundle.
    """
    cfg = config or SimulatorConfig()
    n = len(scores)
    k = _affordable(n, budget=budget, cost_per_treated=cfg.cost_per_treated)

    order = _sort_by_score(np.asarray(scores))
    treated_idx = order[:k]
    base = np.asarray(base_outcome).astype(float)
    lift = np.clip(np.asarray(treatment_lift).astype(float), 0.0, None)

    realised = base.copy()
    realised[treated_idx] = np.minimum(base[treated_idx] + lift[treated_idx], 1.0)

    expected_retained = float(realised.sum())
    expected_revenue = expected_retained * cfg.revenue_per_retained - k * cfg.cost_per_treated

    rng = np.random.default_rng(cfg.random_state)
    bootstrapped = np.empty(cfg.bootstrap_iters, dtype=float)
    for i in range(cfg.bootstrap_iters):
        idx = rng.integers(0, n, size=n)
        b_realised = realised[idx]
        b_treated = np.intersect1d(idx, treated_idx, assume_unique=False)
        b_cost = len(b_treated) * cfg.cost_per_treated
        bootstrapped[i] = b_realised.sum() * cfg.revenue_per_retained - b_cost
    lower, upper = np.quantile(bootstrapped, [cfg.alpha / 2, 1 - cfg.alpha / 2])

    return SimulationResult(
        budget=float(budget),
        n_targeted=int(k),
        expected_retained=expected_retained,
        expected_revenue=expected_revenue,
        ci_lower=float(lower),
        ci_upper=float(upper),
        policy=policy_name,
    )


def compare_policies(
    *,
    risk_scores: np.ndarray,
    uplift_scores: np.ndarray,
    treatment_lift: np.ndarray,
    base_outcome: np.ndarray,
    budgets: tuple[float, ...],
    config: SimulatorConfig | None = None,
) -> pd.DataFrame:
    """Simulate risk, uplift, and random policies across multiple budgets.

    :param risk_scores: Per-row Track A scores.
    :param uplift_scores: Per-row Track C scores.
    :param treatment_lift: Per-row retention lift if treated.
    :param base_outcome: Baseline retention probability per customer.
    :param budgets: Budgets to evaluate.
    :param config: Optional simulator configuration.
    :returns: Tidy frame ready for plotting.
    """
    rows = []
    n = len(risk_scores)
    for budget in budgets:
        for result in (
            simulate_risk_policy(
                risk_scores=risk_scores,
                treatment_lift=treatment_lift,
                base_outcome=base_outcome,
                budget=budget,
                config=config,
            ),
            simulate_uplift_policy(
                uplift_scores=uplift_scores,
                treatment_lift=treatment_lift,
                base_outcome=base_outcome,
                budget=budget,
                config=config,
            ),
            simulate_random_policy(
                n=n,
                treatment_lift=treatment_lift,
                base_outcome=base_outcome,
                budget=budget,
                config=config,
            ),
        ):
            rows.append(result.as_dict())
    return pd.DataFrame(rows)
