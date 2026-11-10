"""Shared matplotlib styling for the CG programs."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_2,
    "axes.titlecolor": TEXT,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT_2,
    "lines.linewidth": 2.0,
    "legend.frameon": False,
    "legend.labelcolor": TEXT,
    "font.size": 10,
})


def cg_axis(ax, title: str):
    """Axis for CG vs time: positive aft, so invert so that 'up' = forward."""
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("time since ignition (s)")
    ax.set_ylabel("CG position, m aft of ox-tank top  (up = forward)")
    ax.invert_yaxis()
