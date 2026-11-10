"""Injector / feed-line / chamber coupling for the blowdown program.

Each propellant branch: supply pressure -> fixed losses (piston friction) ->
feed-line loss (k * mdot^2) -> injector (single-phase incompressible orifice,
Cd from the design sheet) -> chamber. With the quadratic feed loss the branch
flow has a closed form:

    mdot = CdA * sqrt( 2*rho*(P_supply - dP_fixed - Pc) / (1 + 2*rho*k*CdA^2) )

Chamber pressure is found from Pc = (mdot_ox + mdot_fuel) * c*_eff / A_t by
bisection; the residual is monotonic in Pc.

Note: the nitrous injector is really two-phase; the sheet's Cd = 0.45 is an
effective value that absorbs that. Swapping in a Dyer/NHNE model later only
touches `Branch.mass_flow`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Injector:
    n_holes: int
    hole_diameter: float
    cd: float

    @property
    def cda(self) -> float:
        return self.cd * self.n_holes * math.pi * self.hole_diameter**2 / 4.0


@dataclass
class Branch:
    injector: Injector
    k_loss: float = 0.0      # Pa / (kg/s)^2  feed-line loss coefficient
    dp_fixed: float = 0.0    # Pa  fixed drop (e.g. piston friction)

    @classmethod
    def from_nominal_loss(cls, injector: Injector, dp_nom: float, mdot_nom: float, dp_fixed: float = 0.0) -> "Branch":
        return cls(injector, k_loss=dp_nom / mdot_nom**2, dp_fixed=dp_fixed)

    def mass_flow(self, p_supply: float, p_chamber: float, rho: float) -> float:
        dp = p_supply - self.dp_fixed - p_chamber
        if dp <= 0.0:
            return 0.0
        cda = self.injector.cda
        return cda * math.sqrt(2.0 * rho * dp / (1.0 + 2.0 * rho * self.k_loss * cda**2))

    def feed_loss(self, mdot: float) -> float:
        return self.k_loss * mdot**2


@dataclass
class Engine:
    throat_diameter: float
    cstar_eff: float
    thrust_coefficient: float

    @property
    def throat_area(self) -> float:
        return math.pi * self.throat_diameter**2 / 4.0

    def solve_chamber(self, p_ox: float, rho_ox: float, ox: Branch,
                      p_fuel: float, rho_fuel: float, fuel: Branch) -> tuple[float, float, float]:
        """Return (Pc, mdot_ox, mdot_fuel) consistent with both injectors and c*."""
        At = self.throat_area

        def flows(pc):
            return ox.mass_flow(p_ox, pc, rho_ox), fuel.mass_flow(p_fuel, pc, rho_fuel)

        def residual(pc):
            mo, mf = flows(pc)
            return (mo + mf) * self.cstar_eff / At - pc

        lo, hi = 0.0, max(p_ox, p_fuel)
        if residual(lo) <= 0.0:          # no flow possible
            return 0.0, 0.0, 0.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if residual(mid) > 0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1.0:            # 1 Pa
                break
        pc = 0.5 * (lo + hi)
        mo, mf = flows(pc)
        return pc, mo, mf

    def thrust(self, pc: float) -> float:
        return self.thrust_coefficient * pc * self.throat_area
