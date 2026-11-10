# Liquid-rocket centre-of-gravity programs (concentric N2O / IPA piston-fed tanks)

Two programs sharing one core. Python >= 3.11, numpy, matplotlib.

```
python cg_constant_flow.py            # Program 1: constant mass flows from the design sheet
python cg_blowdown.py                 # Program 2: flows computed from the nitrous tank + piston feed
python cg_blowdown.py --tank empirical   # Program 2 with the sheet-style linear pressure decay instead
python compare.py                     # overlay the three cases
python selfcheck.py                   # geometry / property / sheet-reproduction checks
```

Outputs go to `results/` as CSV + PNG. All numbers come from `vehicle.toml`;
lines marked **PLACEHOLDER** are guesses to replace when measured.

## Coordinate convention
`z` in metres, **zero at the forward (top) inside face of the oxidizer tank, positive aft**.
Set `meta.nose_to_tank_top` once the forward section exists and the CSVs' `cg_from_nose` column
becomes nose-referenced.

## Geometry model (`geometry.py`, `vehicle.py`)
Every fluid region is a cross-sectional-area profile A(z) on a fine grid. Liquids occupy the aft end
of their region (settled under thrust / pushed by the piston); volume and centroid are cumulative
integrals, so domes or other profile changes are a one-line addition.

Regions:
* **N2O liquid** — annulus between ox-tank wall and fuel-tube OD, plus the full-circle top gap above the tube.
* **N2O vapour** — everything not liquid: annulus ullage + top gap + inner tube above the piston. Uniform density.
* **Fuel** — aft end of the inner tube up to the piston underside.
* **Piston** — moving point mass riding on the fuel surface.
* **Dry** — list of fixed point masses (currently one lumped PLACEHOLDER).

## Program 1 — `cg_constant_flow.py`
Constant `mdot_ox`, `mdot_fuel`. Vapour is isothermal at the loaded temperature; the mass needed to
fill the growing ullage comes out of the liquid. No pressures.

## Program 2 — `cg_blowdown.py`
* `nitrous.py` — ESDU 91022 saturated-N2O correlations (checked against CoolProp: < 1 % across 250–300 K)
  and an **equilibrium tank**: state = (total mass, internal energy, free volume); each step removes
  liquid at the outlet, adds the fuel volume that left (piston descends → vapour volume grows) and the
  P·dV work the vapour does on the piston, then re-solves T from the energy + volume constraints.
  Once liquid is gone the vapour blows down polytropically. An `EmpiricalTank` option prescribes a
  linear pressure decay with liquid mass (sheet-style "decay constant") on the saturation line.
* `feed_system.py` — fuel supply = P_tank − piston friction; each branch has a feed loss ∝ mdot² fitted
  to the sheet's nominal loss and an SPI injector with the sheet's Cd; chamber pressure solved from
  Pc = Σmdot · c*_eff / A_t each step. `cstar_eff` and `thrust_coefficient` are derived from the sheet.

## Reproduction of the spec sheet (initial conditions)
| quantity | spec | this code |
|---|---|---|
| tank pressure | 44 atm | 44.0 atm (T = 14.5 °C) |
| chamber pressure | 20 atm | 20.0 atm |
| mdot ox / fuel | 1.977 / 0.791 kg/s | 1.977 / 0.790 kg/s |
| O/F | 2.5 | 2.505 |
| thrust | 3842 N | 3844 N |
| wet mass | 42.813 kg | 42.37 kg (dry 21.285 + 15.085 + 5.708 + 0.29 vapour) |
| burn time | 8.04 s | 8.04 s equilibrium / 7.30 s empirical — but ended by **liquid N2O depletion**, with 0.5–0.6 kg IPA stranded |

Static margin: `[stability]` holds CP fixed at the position implied by the spec's initial 4.16 cal, so
the programs report the CG-driven change in margin over the burn (minimum ≈ 3.4 cal mid-burn).

## Known simplifications
* Flat tank ends. Tank-wall heat transfer ignored (adiabatic). Nitrous injector is single-phase with an
  effective Cd (swap in Dyer/NHNE in `Branch.mass_flow` if wanted). No slosh, no coast-phase propellant
  positioning. Piston inertia ignored.
* The 15.085 kg N2O load puts the liquid surface 29 mm below the inner-tube top (max 15.5 kg at 14.5 °C).
* Dry-mass layout: the 4 kg
  "piston + bulkhead" is split 1 / 3 kg; the 7.3 kg remainder is lumped into forward and aft sections
  at guessed positions; `nose_to_tank_top` is a guess. All marked PLACEHOLDER in `vehicle.toml`.
