from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_roi_chart(comparison: pd.DataFrame, *, output: Path) -> Path:
    """Render the headline ROI chart comparing policies across budgets.

    :param comparison: Output of
        :func:`flightrisk.eval.simulator.compare_policies`.
    :param output: Destination PNG path.
    :returns: The path written.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.25
    budgets = sorted(comparison["budget"].unique())
    policies = ["risk", "uplift", "random"]
    colours = {"risk": "tab:blue", "uplift": "tab:green", "random": "tab:gray"}

    x_positions = {b: i for i, b in enumerate(budgets)}
    for offset, policy in enumerate(policies):
        sub = comparison[comparison["policy"] == policy].sort_values("budget")
        xs = [x_positions[b] + (offset - 1) * width for b in sub["budget"]]
        revenues = sub["expected_revenue"].values
        lower = revenues - sub["ci_lower"].values
        upper = sub["ci_upper"].values - revenues
        ax.bar(
            xs,
            revenues,
            width=width,
            color=colours[policy],
            label=policy,
            yerr=[lower.clip(min=0), upper.clip(min=0)],
            capsize=4,
        )
    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels([f"${int(b):,}" for b in budgets])
    ax.set_xlabel("budget")
    ax.set_ylabel("expected retained revenue")
    ax.set_title("Campaign ROI: risk vs uplift vs random")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output
