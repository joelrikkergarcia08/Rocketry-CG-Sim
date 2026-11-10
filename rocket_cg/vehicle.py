"""Vehicle model: loads the TOML config and converts propellant masses into a
centre of gravity.

Mass items tracked
------------------
* liquid N2O      - settled aft in the annulus (and in the top gap if overfilled)
* N2O vapour      - fills the remaining annulus/top-gap volume AND the inner tube
                    above the piston (uniform density)
* liquid fuel     - aft end of the inner tube, up to the piston face
* piston          - moving point mass riding on the fuel column
* dry components  - fixed point masses

Call `VehicleModel.mass_properties(m_ox_liq, rho_ox_liq, m_vap, m_fuel)`.
"""
from __future__ import annotations

import math
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib  # Python >= 3.11
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from geometry import AxialProfile, FillResult, circle_area


@dataclass
class MassItem:
    name: str
    mass: float
    z: float


@dataclass
class MassProperties:
    total_mass: float
    cg: float
    items: list[MassItem]
    ox_liquid_level: float      # z of the liquid N2O surface
    fuel_level: float           # z of the fuel surface (piston underside)
    vapour_volume: float        # m^3 available to vapour


class VehicleModel:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        ox = cfg["ox_tank"]
        ft = cfg["fuel_tube"]
        pst = cfg["piston"]

        if ox.get("end_shape", "flat") != "flat":
            raise NotImplementedError("Only flat tank ends are implemented")

        self.ox_id = ox["inner_diameter"]
        self.ox_len = ox["length"]
        self.tube_od = ft["outer_diameter"]
        self.tube_id = ft["inner_diameter"]
        self.tube_top = ft["top_gap"]
        self.tube_bottom = ft["top_gap"] + ft["length"]
        self.stroke = ft["stroke"]
        self.piston_mass = pst["mass"]
        self.piston_thk = pst["thickness"]

        if self.tube_top >= self.ox_len:
            raise ValueError("Inner tube top is below the oxidizer tank bottom")

        z_max = max(self.ox_len, self.tube_bottom)
        n = 8000

        # Region available to N2O outside the inner tube: full circle in the top
        # gap, annulus alongside the tube.
        self.ox_region = AxialProfile(0.0, z_max, n)
        self.ox_region.add_cylinder(0.0, self.tube_top, self.ox_id / 2)
        self.ox_region.add_cylinder(self.tube_top, self.ox_len, self.ox_id / 2, self.tube_od / 2)

        # Interior of the inner tube (fuel below the piston, vapour above it).
        self.tube_region = AxialProfile(0.0, z_max, n)
        self.tube_region.add_cylinder(self.tube_top, self.tube_bottom, self.tube_id / 2)

        self.tube_area = circle_area(self.tube_id)
        self.fuel_capacity = self.tube_area * self.stroke          # m^3
        self.ox_region_volume = self.ox_region.total_volume

        self.dry_items = [MassItem(c["name"], c["mass"], c["z"]) for c in cfg["dry"]["components"]]
        self.dry_mass = sum(i.mass for i in self.dry_items)
        self.dry_cg = sum(i.mass * i.z for i in self.dry_items) / self.dry_mass

        self.nose_offset = cfg["meta"].get("nose_to_tank_top", 0.0)

        st = cfg.get("stability")
        self.ref_diameter = st["reference_diameter"] if st else None
        self.initial_margin = st["initial_margin_cal"] if st else None

    def static_margin(self, cg: float, cg0: float) -> float | None:
        """Static margin in calibers for a CP fixed at the position implied by the
        initial margin: margin(t) = margin0 - (cg - cg0)/D. None if not configured."""
        if self.ref_diameter is None:
            return None
        return self.initial_margin - (cg - cg0) / self.ref_diameter

    # ------------------------------------------------------------------
    @classmethod
    def from_toml(cls, path: str | Path) -> "VehicleModel":
        with open(path, "rb") as f:
            cfg = tomllib.load(f)
        return cls(cfg)

    # ------------------------------------------------------------------
    def fuel_fill(self, m_fuel: float, rho_fuel: float) -> FillResult:
        V = max(m_fuel, 0.0) / rho_fuel
        if V > self.fuel_capacity * (1 + 1e-6):
            raise ValueError(f"Fuel load {m_fuel:.3f} kg exceeds tube stroke capacity "
                             f"({self.fuel_capacity * rho_fuel:.3f} kg)")
        return self.tube_region.fill_from_aft(V)

    def vapour_volume(self, m_ox_liq: float, rho_ox_liq: float, m_fuel: float, rho_fuel: float) -> float:
        """Free volume available to N2O vapour for a given liquid state."""
        V_liq = max(m_ox_liq, 0.0) / rho_ox_liq
        V_out = self.ox_region_volume - V_liq
        ff = self.fuel_fill(m_fuel, rho_fuel)
        piston_top = ff.z_top - self.piston_thk
        V_tube = self.tube_region.volume_between(self.tube_top, max(piston_top, self.tube_top))
        return V_out + V_tube

    def mass_properties(self, m_ox_liq: float, rho_ox_liq: float, m_vap: float,
                        m_fuel: float, rho_fuel: float) -> MassProperties:
        items: list[MassItem] = list(self.dry_items)

        # --- liquid oxidizer -------------------------------------------
        ox_fill = self.ox_region.fill_from_aft(max(m_ox_liq, 0.0) / rho_ox_liq)
        if m_ox_liq > 0:
            items.append(MassItem("N2O liquid", m_ox_liq, ox_fill.centroid))

        # --- fuel + piston -----------------------------------------------
        ff = self.fuel_fill(m_fuel, rho_fuel)
        if m_fuel > 0:
            items.append(MassItem("fuel", m_fuel, ff.centroid))
        piston_top = ff.z_top - self.piston_thk
        items.append(MassItem("piston", self.piston_mass, ff.z_top - 0.5 * self.piston_thk))

        # --- vapour (uniform density over two sub-regions) -----------------
        V_out = self.ox_region.volume_between(0.0, ox_fill.z_top)
        c_out = self.ox_region.centroid_between(0.0, ox_fill.z_top)
        z_hi = max(piston_top, self.tube_top)
        V_tube = self.tube_region.volume_between(self.tube_top, z_hi)
        c_tube = self.tube_region.centroid_between(self.tube_top, z_hi)
        V_vap = V_out + V_tube
        if m_vap > 0 and V_vap > 0:
            c_vap = (V_out * c_out + V_tube * c_tube) / V_vap
            items.append(MassItem("N2O vapour", m_vap, c_vap))

        total = sum(i.mass for i in items)
        cg = sum(i.mass * i.z for i in items) / total
        return MassProperties(total, cg, items, ox_fill.z_top, ff.z_top, V_vap)

    # ------------------------------------------------------------------
    def check_ox_load(self, m_ox_liq: float, rho_ox_liq: float) -> None:
        """Warn if the liquid N2O surface rises above the inner tube top."""
        fill = self.ox_region.fill_from_aft(m_ox_liq / rho_ox_liq)
        if fill.z_top < self.tube_top:
            annulus_cap = self.ox_region.volume_between(self.tube_top, self.ox_len) * rho_ox_liq
            warnings.warn(
                f"Liquid N2O surface is {1e3 * (self.tube_top - fill.z_top):.0f} mm ABOVE the inner tube top: "
                f"liquid would flow into the tube above the piston. Max load with the surface at the "
                f"tube top is {annulus_cap:.2f} kg at this density (configured {m_ox_liq:.3f} kg).",
                stacklevel=2,
            )

    def describe(self) -> str:
        lines = [
            f"Vehicle: {self.cfg['meta']['name']}",
            f"  datum: ox tank forward inside face, +z aft; nose offset {self.nose_offset:.3f} m",
            f"  ox region volume {self.ox_region_volume * 1e3:.2f} L  (annulus+gap)",
            f"  fuel capacity    {self.fuel_capacity * 1e3:.2f} L  (stroke {self.stroke:.3f} m)",
            f"  inner tube       z = {self.tube_top:.3f} .. {self.tube_bottom:.3f} m",
            f"  dry mass {self.dry_mass:.2f} kg at z = {self.dry_cg:.3f} m",
        ]
        return "\n".join(lines)


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)
