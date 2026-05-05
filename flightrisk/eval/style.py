from __future__ import annotations

from typing import Any

PALETTE: dict[str, str] = {
    "risk": "#1F6FEB",
    "uplift": "#2DA44E",
    "survival": "#9333EA",
    "random": "#A0A0A0",
    "treated": "#2DA44E",
    "control": "#1F6FEB",
    "perfect": "#0F172A",
    "accent": "#F97316",
    "muted": "#64748B",
    "grid": "#E2E8F0",
}

BACKGROUND = "#FFFFFF"
TEXT = "#0F172A"


def _rcparams() -> dict[str, Any]:
    """Return the matplotlib ``rcParams`` overrides for the flightrisk theme.

    :returns: Mapping ready to pass to :func:`matplotlib.rcParams.update`.
    """
    return {
        "figure.facecolor": BACKGROUND,
        "figure.edgecolor": BACKGROUND,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": BACKGROUND,
        "axes.facecolor": BACKGROUND,
        "axes.edgecolor": PALETTE["muted"],
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "axes.titleweight": "semibold",
        "axes.titlesize": 13,
        "axes.titlepad": 12,
        "axes.labelsize": 11,
        "axes.labelweight": "regular",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.prop_cycle": _prop_cycle(),
        "grid.color": PALETTE["grid"],
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.6,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "font.family": ["DejaVu Sans", "Segoe UI", "Helvetica", "Arial", "sans-serif"],
        "font.size": 11,
    }


def _prop_cycle() -> Any:
    """Build a colour cycle prioritising the flightrisk palette.

    :returns: A matplotlib ``cycler`` over the brand colours.
    """
    from cycler import cycler

    return cycler(
        color=[
            PALETTE["risk"],
            PALETTE["uplift"],
            PALETTE["survival"],
            PALETTE["accent"],
            PALETTE["muted"],
        ]
    )


def apply_style() -> None:
    """Apply the flightrisk theme to the global matplotlib state.

    Idempotent: safe to call from notebooks, scripts, and entry points; the
    only side effect is updating :mod:`matplotlib.rcParams`.
    """
    import matplotlib as mpl

    mpl.rcParams.update(_rcparams())


def annotate_axis(ax: Any, *, source: str | None = None, footer: str | None = None) -> None:
    """Add a subtle source / footer line to a plot.

    :param ax: A matplotlib :class:`Axes` (or any object with a ``figure`` attr).
    :param source: Optional source attribution rendered bottom-left of the figure.
    :param footer: Optional centered footer text.
    """
    fig = ax.figure
    if source:
        fig.text(0.01, 0.005, source, fontsize=8, color=PALETTE["muted"], ha="left")
    if footer:
        fig.text(0.5, 0.005, footer, fontsize=8, color=PALETTE["muted"], ha="center")
