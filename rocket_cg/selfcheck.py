#!/usr/bin/env python3
"""Self-checks for the CG programs. Run: python selfcheck.py
No test framework needed; prints PASS/FAIL per check and exits non-zero on failure."""
from __future__ import annotations

import math
import sys

import nitrous
from feed_system import Branch, Engine, Injector
from geometry import AxialProfile, annulus_area, circle_area
from nitrous import EquilibriumTank
from vehicle import VehicleModel, load_config

failures = 0


def check(name, ok, detail=""):
    global failures
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        failures += 1


# --- geometry ------------------------------------------------------------
p = AxialProfile(0.0, 2.0, 8001).add_cylinder(0.0, 2.0, 0.1)
fr = p.fill_from_aft(circle_area(0.2) * 0.5)         # fill the bottom 0.5 m of a 2 m cylinder
check("cylinder fill level", abs(fr.z_top - 1.5) < 1e-4, f"z_top={fr.z_top:.5f} (expect 1.5)")
check("cylinder centroid", abs(fr.centroid - 1.75) < 1e-4, f"c={fr.centroid:.5f} (expect 1.75)")

a = AxialProfile(0.0, 1.0, 8001).add_cylinder(0.0, 1.0, 0.07, 0.04)
check("annulus volume", abs(a.total_volume - annulus_area(0.14, 0.08)) < 1e-9,
      f"{a.total_volume:.6e} vs {annulus_area(0.14, 0.08):.6e}")
check("partial centroid", abs(a.centroid_between(0.2, 0.6) - 0.4) < 1e-4)

# --- nitrous properties against known points -----------------------------
check("N2O Psat 60F ~45.6 bar", abs(nitrous.p_sat(288.71) / 1e5 - 45.6) < 0.4, f"{nitrous.p_sat(288.71)/1e5:.2f} bar")
check("spec: tank pressure 44 atm at configured T", abs(nitrous.p_sat(load_config("vehicle.toml")["propellants"]["ox_temperature"]) / 101325 - 44) < 0.05)
check("N2O rho_l 60F ~817", abs(nitrous.rho_liquid(288.71) - 817) < 5, f"{nitrous.rho_liquid(288.71):.1f}")
check("T_sat inverts P_sat", abs(nitrous.t_sat(nitrous.p_sat(280.0)) - 280.0) < 1e-3)

# --- equilibrium tank -----------------------------------------------------
tank = EquilibriumTank(288.71, 10.0, 0.015)
T0, P0 = tank.T, tank.P
tank.step(0.0, 0.0)
check("closed tank keeps state", abs(tank.T - T0) < 1e-4 and abs(tank.P - P0) < 10, f"dT={tank.T-T0:.2e}")
m0 = tank.m
tank.step(0.5, 0.0)
check("mass conserved on outflow", abs(tank.m - (m0 - 0.5)) < 1e-9 and abs(tank.m_liq + tank.m_vap - tank.m) < 1e-9)
check("outflow cools tank", tank.T < T0, f"T {T0:.2f} -> {tank.T:.2f}")
T1 = tank.T
tank.step(0.0, 0.0005)                                # piston moves: vapour volume grows
check("volume growth alone cools tank", tank.T < T1, f"T {T1:.3f} -> {tank.T:.3f}")

# --- feed system reproduces the design sheet ------------------------------
cfg = load_config("vehicle.toml")
fs, en = cfg["feed_system"], cfg["engine"]
ox = Branch.from_nominal_loss(Injector(**fs["ox_injector"]), fs["ox_feed_loss"]["dp"], fs["ox_feed_loss"]["mdot"])
fuel = Branch.from_nominal_loss(Injector(**fs["fuel_injector"]), fs["fuel_feed_loss"]["dp"], fs["fuel_feed_loss"]["mdot"],
                                dp_fixed=cfg["piston"]["friction_dp"])
engine = Engine(en["throat_diameter"], en["cstar_eff"], en["thrust_coefficient"])
T0 = cfg["propellants"]["ox_temperature"]
P = nitrous.p_sat(T0)
pc, mo, mf = engine.solve_chamber(P, nitrous.rho_liquid(T0), ox, P, 786.0, fuel)
check("spec: Pc ~20 atm", abs(pc / 101325 - 20) < 0.5, f"{pc/101325:.2f} atm")
check("spec: mdot_ox ~1.977", abs(mo - 1.9767) < 0.03, f"{mo:.3f} kg/s")
check("spec: mdot_fuel ~0.791", abs(mf - 0.79107) < 0.02, f"{mf:.3f} kg/s")
check("spec: thrust ~3842 N", abs(engine.thrust(pc) - 3842) < 60, f"{engine.thrust(pc):.0f} N")

# --- vehicle mass bookkeeping ---------------------------------------------
veh = VehicleModel(cfg)
mp = veh.mass_properties(15.085, nitrous.rho_liquid(T0), 0.0, 5.708, 786.0)
check("fuel capacity matches sheet 5.708 kg", abs(veh.fuel_capacity * 786 - 5.708) < 0.01, f"{veh.fuel_capacity*786:.3f} kg")
check("wet mass = dry(21.285 incl. piston) + props", abs(mp.total_mass - (21.285 + 15.085 + 5.708)) < 1e-6, f"{mp.total_mass:.3f} kg")
# CG must lie between the forward-most and aft-most item
zs = [i.z for i in mp.items]
check("CG inside item span", min(zs) <= mp.cg <= max(zs))
# with props removed CG -> dry+piston CG
mp0 = veh.mass_properties(0.0, 800.0, 0.0, 0.0, 786.0)
exp = (sum(i.mass * i.z for i in veh.dry_items) + veh.piston_mass * (veh.tube_bottom - 0.5 * veh.piston_thk)) / (veh.dry_mass + veh.piston_mass)
check("empty CG = dry + piston", abs(mp0.cg - exp) < 1e-6, f"{mp0.cg:.4f} vs {exp:.4f}")

print("\nALL PASS" if failures == 0 else f"\n{failures} FAILURE(S)")
sys.exit(1 if failures else 0)
