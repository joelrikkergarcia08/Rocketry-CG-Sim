#!/usr/bin/env python3
"""Overlay the CG traces of the three flow models.

    python compare.py [vehicle.toml] [--out results/]

Left: CG vs time. Right: CG vs propellant mass expelled — if the curves
collapse onto one another, the flow model only re-maps the time axis and the
remaining spread is mixture-ratio drift.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import cg_blowdown
import cg_constant_flow
from plotting import SERIES, cg_axis, plt
from vehicle import VehicleModel, load_config


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="vehicle.toml")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    veh = VehicleModel(cfg)

    cases = []
    rows, reason = cg_constant_flow.run(cfg, veh)
    cases.append(("constant flow", rows))
    rows, reason, _ = cg_blowdown.run(cfg, veh, "empirical")
    cases.append((f"empirical decay to {100 * cfg['blowdown']['empirical_end_fraction']:.0f} %", rows))
    rows, reason, _ = cg_blowdown.run(cfg, veh, "equilibrium")
    cases.append(("equilibrium blowdown", rows))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True, gridspec_kw={"wspace": 0.08})
    cg_axis(ax1, "CG vs time")
    ax2.set_title("CG vs propellant expelled", loc="left", fontweight="bold")
    ax2.set_xlabel("propellant mass expelled (kg)")
    for (label, rows), color in zip(cases, SERIES):
        t = np.array([r["t"] for r in rows])
        cg = np.array([r["cg"] for r in rows])
        m = np.array([r["total_mass"] for r in rows])
        ax1.plot(t, cg, color=color, label=label)
        ax2.plot(m[0] - m, cg, color=color, label=label)
        ax1.annotate(f"{t[-1]:.1f} s", (t[-1], cg[-1]), xytext=(4, 0), textcoords="offset points",
                     fontsize=9, color=color, va="center")
    ax1.legend(loc="lower left", fontsize=9)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "cg_comparison.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out / 'cg_comparison.png'}")


if __name__ == "__main__":
    main()
