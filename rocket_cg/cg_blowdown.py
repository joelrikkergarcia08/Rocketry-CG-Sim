#!/usr/bin/env python3
"""Program 2 — centre of gravity vs time with mass flows COMPUTED from the
self-pressurising nitrous tank and the piston-coupled fuel feed.

    python cg_blowdown.py [vehicle.toml] [--out results/] [--tank equilibrium|empirical]

Model
-----
Tank      : saturated N2O liquid + vapour in equilibrium (energy balance). The
            vapour space is the annulus ullage + top gap + inner tube above the
            piston, so every litre of fuel that leaves adds a litre of vapour
            volume the nitrous must fill (and the vapour does P·dV work on the
            piston). Both effects cool the tank and pull the pressure down.
Fuel side : P_fuel = P_tank − piston friction. Fuel flow through its injector,
            with feed-line loss ∝ mdot².
Ox side   : P_tank → feed-line loss ∝ mdot² → injector (SPI, effective Cd).
Chamber   : Pc = (mdot_ox + mdot_fuel) · c*_eff / A_t, solved each step.
Stops when the configured propellant is exhausted (or t_max).

Outputs: <out>/cg_blowdown.csv and .png
"""
from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np

import nitrous
from feed_system import Branch, Engine, Injector
from nitrous import EmpiricalTank, EquilibriumTank
from plotting import SERIES, TEXT_2, cg_axis, plt
from vehicle import VehicleModel, load_config

ATM = 101325.0


def build_feed(cfg: dict):
    fs, en, pst = cfg["feed_system"], cfg["engine"], cfg["piston"]
    ox_inj = Injector(**fs["ox_injector"])
    f_inj = Injector(**fs["fuel_injector"])
    ox = Branch.from_nominal_loss(ox_inj, fs["ox_feed_loss"]["dp"], fs["ox_feed_loss"]["mdot"])
    fuel = Branch.from_nominal_loss(f_inj, fs["fuel_feed_loss"]["dp"], fs["fuel_feed_loss"]["mdot"],
                                    dp_fixed=pst["friction_dp"])
    engine = Engine(en["throat_diameter"], en["cstar_eff"], en["thrust_coefficient"])
    return ox, fuel, engine


def run(cfg: dict, veh: VehicleModel, tank_model: str | None = None):
    bd, pr = cfg["blowdown"], cfg["propellants"]
    tank_model = tank_model or bd.get("tank_model", "equilibrium")
    T0, rho_f = pr["ox_temperature"], pr["fuel_density"]
    m_liq0, m_fuel = pr["ox_mass_liquid"], pr["fuel_mass"]
    rho_l0 = nitrous.rho_liquid(T0)
    veh.check_ox_load(m_liq0, rho_l0)

    # total volume available to N2O (liquid + vapour) at t = 0
    V_free0 = veh.vapour_volume(m_liq0, rho_l0, m_fuel, rho_f) + m_liq0 / rho_l0
    if tank_model == "equilibrium":
        tank = EquilibriumTank(T0, m_liq0, V_free0)
        tank.gamma = bd.get("gas_phase_gamma", 1.27)
    elif tank_model == "empirical":
        tank = EmpiricalTank(T0, m_liq0, V_free0, bd.get("empirical_end_fraction", 0.7))
    else:
        raise ValueError(f"unknown tank_model {tank_model!r}")

    ox, fuel, engine = build_feed(cfg)
    dt, t_max, stop_on = bd["dt"], bd["t_max"], bd.get("stop_on", "either")

    rows, t = [], 0.0
    cg0 = None
    stop_reason = "t_max reached"
    while True:
        p_tank = tank.P
        pc, mo, mf = engine.solve_chamber(p_tank, tank.rho_l, ox, p_tank, rho_f, fuel)
        mp = veh.mass_properties(tank.m_liq, tank.rho_l, tank.m_vap, m_fuel, rho_f)
        cg0 = mp.cg if cg0 is None else cg0
        rows.append(dict(
            t=t, total_mass=mp.total_mass, cg=mp.cg, cg_from_nose=mp.cg + veh.nose_offset,
            static_margin_cal=veh.static_margin(mp.cg, cg0),
            m_ox_liquid=tank.m_liq, m_ox_vapour=tank.m_vap, m_fuel=m_fuel,
            p_tank=p_tank, p_fuel=p_tank - fuel.dp_fixed, p_chamber=pc,
            mdot_ox=mo, mdot_fuel=mf, of_ratio=(mo / mf if mf > 0 else float("nan")),
            thrust=engine.thrust(pc), tank_T=tank.T, ox_level=mp.ox_liquid_level, fuel_level=mp.fuel_level,
        ))
        fuel_out = m_fuel <= 1e-9
        ox_out = tank.m_liq <= 1e-9
        if (stop_on in ("fuel", "either") and fuel_out) or (stop_on in ("ox", "either") and ox_out):
            stop_reason = "fuel depleted" if fuel_out else "liquid N2O depleted"
            break
        if stop_on == "ox" and fuel_out:
            stop_reason = "fuel depleted (ox still liquid)"
            break
        if pc <= 0.0:
            stop_reason = "no flow (tank pressure collapsed)"
            break
        if t >= t_max - 1e-12:
            break
        step = min(dt, t_max - t)
        if mf > 0:
            step = min(step, m_fuel / mf)
        if mo > 0 and tank.liquid_phase:
            step = min(step, tank.m_liq / mo)
        dm_f = mf * step
        tank.step(mo * step, dm_f / rho_f)        # piston descends -> vapour volume grows
        m_fuel = max(m_fuel - dm_f, 0.0)
        t += step
    return rows, stop_reason, tank_model


def summarize(rows, stop_reason):
    t = np.array([r["t"] for r in rows])
    F = np.array([r["thrust"] for r in rows])
    r0, r1 = rows[0], rows[-1]
    return {
        "burn_time_s": t[-1],
        "stop_reason": stop_reason,
        "p_tank_initial_atm": r0["p_tank"] / ATM, "p_tank_final_atm": r1["p_tank"] / ATM,
        "p_chamber_initial_atm": r0["p_chamber"] / ATM, "p_chamber_final_atm": r1["p_chamber"] / ATM,
        "thrust_initial_N": r0["thrust"], "thrust_final_N": r1["thrust"],
        "total_impulse_Ns": float(np.trapezoid(F, t)),
        "mdot_initial": r0["mdot_ox"] + r0["mdot_fuel"], "of_initial": r0["of_ratio"], "of_final": r1["of_ratio"],
        "tank_T_initial_K": r0["tank_T"], "tank_T_final_K": r1["tank_T"],
        "ox_liquid_left_kg": r1["m_ox_liquid"], "ox_vapour_left_kg": r1["m_ox_vapour"], "fuel_left_kg": r1["m_fuel"],
        "mass_liftoff": r0["total_mass"], "mass_burnout": r1["total_mass"],
        "cg_liftoff": r0["cg"], "cg_burnout": r1["cg"],
    }


def write_outputs(rows, stop_reason, tank_model, veh, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "cg_blowdown.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    t = np.array([r["t"] for r in rows])
    cg = np.array([r["cg"] for r in rows])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10.5), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 2, 2], "hspace": 0.14})
    cg_axis(ax1, f"Centre of gravity — nitrous blowdown ({tank_model} tank model)")
    ax1.plot(t, cg, color=SERIES[0])
    ax1.axhline(veh.dry_cg, color=SERIES[1], linewidth=1.2, linestyle="--")
    ax1.annotate(f"dry CG {veh.dry_cg:.3f} m (placeholder)", (t[0], veh.dry_cg), ha="left", va="bottom",
                 fontsize=9, color=SERIES[1], xytext=(4, 3), textcoords="offset points")
    ax1.annotate(f"liftoff {cg[0]:.3f} m", (t[0], cg[0]), xytext=(6, -10), textcoords="offset points", fontsize=9)
    ax1.annotate(f"burnout {cg[-1]:.3f} m  ({stop_reason}, t = {t[-1]:.2f} s)", (t[-1], cg[-1]),
                 xytext=(-6, 8), textcoords="offset points", ha="right", fontsize=9)
    ax1.set_xlabel("")

    ax2.plot(t, [r["p_tank"] / ATM for r in rows], color=SERIES[0], label="N2O tank")
    ax2.plot(t, [r["p_fuel"] / ATM for r in rows], color=SERIES[2], label="fuel (below piston)")
    ax2.plot(t, [r["p_chamber"] / ATM for r in rows], color=SERIES[1], label="chamber")
    ax2.set_ylabel("pressure (atm)")
    ax2.set_ylim(bottom=0)
    ax2.legend(ncol=3, loc="lower left", fontsize=9)

    ax3.plot(t, [r["mdot_ox"] for r in rows], color=SERIES[0], label="N2O")
    ax3.plot(t, [r["mdot_fuel"] for r in rows], color=SERIES[2], label="fuel")
    ax3.plot(t, [r["mdot_ox"] + r["mdot_fuel"] for r in rows], color=TEXT_2, linewidth=1.2, label="total")
    ax3.set_ylabel("mass flow (kg/s)")
    ax3.set_xlabel("time since ignition (s)")
    ax3.set_ylim(bottom=0)
    ax3.legend(ncol=3, loc="lower left", fontsize=9)
    fig.savefig(out_dir / "cg_blowdown.png", dpi=150, bbox_inches="tight")
    return csv_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default="vehicle.toml")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tank", choices=["equilibrium", "empirical"], default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    veh = VehicleModel(cfg)
    print(veh.describe())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rows, stop_reason, tank_model = run(cfg, veh, args.tank)
        for w in caught:
            print(f"WARNING: {w.message}")
    csv_path = write_outputs(rows, stop_reason, tank_model, veh, Path(args.out))

    s = summarize(rows, stop_reason)
    print(f"\ntank model     : {tank_model}")
    print(f"burn time      : {s['burn_time_s']:.2f} s  ({s['stop_reason']})")
    print(f"tank pressure  : {s['p_tank_initial_atm']:.1f} -> {s['p_tank_final_atm']:.1f} atm   "
          f"(tank {s['tank_T_initial_K'] - 273.15:.1f} -> {s['tank_T_final_K'] - 273.15:.1f} C)")
    print(f"chamber        : {s['p_chamber_initial_atm']:.1f} -> {s['p_chamber_final_atm']:.1f} atm")
    print(f"thrust         : {s['thrust_initial_N']:.0f} -> {s['thrust_final_N']:.0f} N, impulse {s['total_impulse_Ns']:.0f} N s")
    print(f"mass flow      : {s['mdot_initial']:.3f} kg/s initial, O/F {s['of_initial']:.3f} -> {s['of_final']:.3f}")
    print(f"left at stop   : N2O liquid {s['ox_liquid_left_kg']:.2f} kg, vapour {s['ox_vapour_left_kg']:.2f} kg, fuel {s['fuel_left_kg']:.2f} kg")
    print(f"liftoff        : mass {s['mass_liftoff']:.2f} kg, CG {s['cg_liftoff']:.3f} m")
    print(f"burnout        : mass {s['mass_burnout']:.2f} kg, CG {s['cg_burnout']:.3f} m")
    print(f"CG shift       : {1e3 * (s['cg_burnout'] - s['cg_liftoff']):+.0f} mm (+ = aft)")
    if rows[0]["static_margin_cal"] is not None:
        mins = min(rows, key=lambda r: r["static_margin_cal"])
        print(f"static margin  : {rows[0]['static_margin_cal']:.2f} cal liftoff -> {rows[-1]['static_margin_cal']:.2f} cal burnout, "
              f"min {mins['static_margin_cal']:.2f} cal at t = {mins['t']:.2f} s  (CP held fixed)")
    print(f"wrote {csv_path} and {csv_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
