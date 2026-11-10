#!/usr/bin/env python3
"""Program 1 — centre of gravity vs time with CONSTANT propellant mass flows.

    python cg_constant_flow.py [vehicle.toml] [--out results/]

Model
-----
* Oxidizer and fuel leave at the constant rates in [constant_flow].
* Liquid N2O settles aft in the annulus; fuel is pushed aft by the piston.
* N2O vapour (optional): the free volume — annulus ullage, top gap, and the
  inner tube above the piston — is filled with saturated vapour at the loaded
  temperature. Its mass comes out of the liquid (isothermal; no pressure decay).
* Stops when either propellant is exhausted or t_max is reached.

Outputs: <out>/cg_constant_flow.csv and .png
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np

import nitrous
from plotting import SERIES, cg_axis, plt
from vehicle import VehicleModel, load_config


def run(cfg: dict, veh: VehicleModel):
    cf = cfg["constant_flow"]
    pr = cfg["propellants"]
    T0 = pr["ox_temperature"]
    rho_l = nitrous.rho_liquid(T0)
    rho_v = nitrous.rho_vapour(T0) if cf.get("include_vapour", True) else 0.0
    rho_f = pr["fuel_density"]

    m_liq0 = pr["ox_mass_liquid"]
    m_fuel = pr["fuel_mass"]
    veh.check_ox_load(m_liq0, rho_l)
    # N2O inventory = configured liquid + the vapour that fills the ullage at t = 0
    # (same starting state as the blowdown program)
    m_ox_total = m_liq0 + rho_v * veh.vapour_volume(m_liq0, rho_l, m_fuel, rho_f)

    def split_vapour(m_ox_remaining, m_fuel_now):
        """Partition remaining N2O into liquid + vapour so the vapour fills the free volume."""
        if rho_v == 0.0:
            return m_ox_remaining, 0.0
        # m_l/rho_l + m_v/rho_v = V_free(m_l)   with   m_l + m_v = m_ox_remaining
        # V_free depends weakly on m_l through the liquid level; two fixed-point passes suffice.
        m_l = m_ox_remaining
        for _ in range(3):
            V_free = veh.vapour_volume(m_l, rho_l, m_fuel_now, rho_f) + m_l / rho_l  # total N2O volume
            m_l = (V_free - m_ox_remaining / rho_v) / (1 / rho_l - 1 / rho_v)
            m_l = min(max(m_l, 0.0), m_ox_remaining)
        return m_l, m_ox_remaining - m_l

    dt, t_max = cf["dt"], cf["t_max"]
    mdot_ox, mdot_f = cf["mdot_ox"], cf["mdot_fuel"]
    rows = []
    t = 0.0
    m_ox = m_ox_total
    cg0 = None
    stop_reason = "t_max reached"
    while True:
        m_l, m_v = split_vapour(m_ox, m_fuel)
        mp = veh.mass_properties(m_l, rho_l, m_v, m_fuel, rho_f)
        cg0 = mp.cg if cg0 is None else cg0
        rows.append(dict(t=t, total_mass=mp.total_mass, cg=mp.cg, cg_from_nose=mp.cg + veh.nose_offset,
                         static_margin_cal=veh.static_margin(mp.cg, cg0),
                         m_ox_liquid=m_l, m_ox_vapour=m_v, m_fuel=m_fuel,
                         ox_level=mp.ox_liquid_level, fuel_level=mp.fuel_level))
        if m_fuel <= 1e-9:
            stop_reason = "fuel depleted"
            break
        if m_l <= 1e-9:
            stop_reason = "liquid N2O depleted"
            break
        if t >= t_max - 1e-12:
            break
        step = min(dt, t_max - t)
        # clamp the last step so we land exactly on depletion
        step = min(step, m_fuel / mdot_f if mdot_f > 0 else step, m_l / mdot_ox if mdot_ox > 0 else step)
        m_fuel = max(m_fuel - mdot_f * step, 0.0)
        m_ox = max(m_ox - mdot_ox * step, 0.0)
        t += step
    return rows, stop_reason


def write_outputs(rows, stop_reason, veh, out_dir: Path, cfg):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "cg_constant_flow.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    t = np.array([r["t"] for r in rows])
    cg = np.array([r["cg"] for r in rows])
    m = np.array([r["total_mass"] for r in rows])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12})
    cg_axis(ax1, "Centre of gravity — constant mass flow")
    ax1.plot(t, cg, color=SERIES[0])
    ax1.axhline(veh.dry_cg, color=SERIES[1], linewidth=1.2, linestyle="--")
    ax1.annotate(f"dry CG {veh.dry_cg:.3f} m (placeholder)", (t[0], veh.dry_cg), ha="left", va="bottom",
                 fontsize=9, color=SERIES[1], xytext=(4, 3), textcoords="offset points")
    ax1.annotate(f"liftoff {cg[0]:.3f} m", (t[0], cg[0]), xytext=(6, -10), textcoords="offset points", fontsize=9)
    ax1.annotate(f"burnout {cg[-1]:.3f} m  ({stop_reason}, t = {t[-1]:.2f} s)", (t[-1], cg[-1]),
                 xytext=(-6, 8), textcoords="offset points", ha="right", fontsize=9)
    ax1.set_xlabel("")

    ax2.plot(t, m, color=SERIES[0], label="total")
    ax2.plot(t, [r["m_ox_liquid"] for r in rows], color=SERIES[2], label="N2O liquid")
    ax2.plot(t, [r["m_fuel"] for r in rows], color=SERIES[3], label="fuel")
    ax2.plot(t, [r["m_ox_vapour"] for r in rows], color=SERIES[4], label="N2O vapour")
    ax2.set_ylabel("mass (kg)")
    ax2.set_xlabel("time since ignition (s)")
    ax2.legend(ncol=4, loc="upper right", fontsize=9)
    fig.savefig(out_dir / "cg_constant_flow.png", dpi=150, bbox_inches="tight")
    return csv_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default="vehicle.toml")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    veh = VehicleModel(cfg)
    print(veh.describe())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rows, stop_reason = run(cfg, veh)
        for w in caught:
            print(f"WARNING: {w.message}")
    csv_path = write_outputs(rows, stop_reason, veh, Path(args.out), cfg)

    r0, r1 = rows[0], rows[-1]
    print(f"\nliftoff : mass {r0['total_mass']:.2f} kg, CG {r0['cg']:.3f} m")
    print(f"burnout : mass {r1['total_mass']:.2f} kg, CG {r1['cg']:.3f} m at t = {r1['t']:.2f} s ({stop_reason})")
    print(f"CG shift: {1e3 * (r1['cg'] - r0['cg']):+.0f} mm (+ = aft)")
    if r0["static_margin_cal"] is not None:
        mins = min(rows, key=lambda r: r["static_margin_cal"])
        print(f"static margin (fixed CP): {r0['static_margin_cal']:.2f} cal liftoff -> {r1['static_margin_cal']:.2f} cal burnout, "
              f"min {mins['static_margin_cal']:.2f} cal at t = {mins['t']:.2f} s")
    print(f"wrote {csv_path} and {csv_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
