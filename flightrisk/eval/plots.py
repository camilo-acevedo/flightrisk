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

    from flightrisk.eval.style import PALETTE, annotate_axis, apply_style

    apply_style()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    width = 0.26
    budgets = sorted(comparison["budget"].unique())
    policies = ["risk", "uplift", "random"]
    colours = {
        "risk": PALETTE["risk"],
        "uplift": PALETTE["uplift"],
        "random": PALETTE["random"],
    }
    pretty = {"risk": "Top-k by P(churn)", "uplift": "Top-k by τ̂", "random": "Random"}

    x_positions = {b: i for i, b in enumerate(budgets)}
    for offset, policy in enumerate(policies):
        sub = comparison[comparison["policy"] == policy].sort_values("budget")
        xs = [x_positions[b] + (offset - 1) * width for b in sub["budget"]]
        revenues = sub["expected_revenue"].values
        lower = revenues - sub["ci_lower"].values
        upper = sub["ci_upper"].values - revenues
        bars = ax.bar(
            xs,
            revenues,
            width=width,
            color=colours[policy],
            label=pretty[policy],
            yerr=[lower.clip(min=0), upper.clip(min=0)],
            capsize=4,
            edgecolor="white",
            linewidth=1.0,
            alpha=0.92,
        )
        for rect, value in zip(bars, revenues, strict=True):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height(),
                f"${value / 1000:,.0f}k",
                ha="center",
                va="bottom",
                fontsize=9,
                color=colours[policy],
                fontweight="semibold",
            )

    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels([f"${int(b):,}" for b in budgets])
    ax.set_xlabel("Campaign budget")
    ax.set_ylabel("Expected retained revenue (USD)")
    ax.set_title("Headline ROI — risk vs uplift vs random under fixed budgets", loc="left")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _pos: f"${v / 1000:,.0f}k"))
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", framealpha=0.9, ncols=3)
    annotate_axis(
        ax,
        source="flightrisk · ROI simulator",
        footer="Bars: mean retained revenue. Error bars: 95% bootstrap CI.",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output)
    plt.close(fig)
    return output
