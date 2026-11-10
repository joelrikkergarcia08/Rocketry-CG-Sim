"""Saturated nitrous oxide properties and a self-pressurising tank model.

Properties: ESDU 91022 correlations (as reproduced in the Aspire Space paper
"Modelling the nitrous run tank emptying"), valid from the triple point
(182.3 K) to the critical point (309.57 K). Inputs in kelvin, outputs in SI.

Tank model: `EquilibriumTank` — liquid and vapour in thermodynamic equilibrium
at a common temperature. State is (total mass, total internal energy, free
volume). Each step removes liquid through the outlet, applies the P·dV work
done on the fuel piston, and re-solves the temperature from the energy and
volume constraints. This is the classic "equilibrium" blowdown model: it tends
to under-predict pressure slightly versus a real tank, which is somewhat
superheated.
"""
from __future__ import annotations

import math

T_CRIT = 309.57      # K
P_CRIT = 7.251e6     # Pa
RHO_CRIT = 452.0     # kg/m^3
T_TRIPLE = 182.3     # K


def _check(T: float) -> float:
    if not (T_TRIPLE - 1 <= T < T_CRIT):
        raise ValueError(f"N2O temperature {T:.2f} K outside saturated range")
    return T / T_CRIT


def p_sat(T: float) -> float:
    """Vapour pressure, Pa."""
    Tr = _check(T)
    x = 1 - Tr
    b = (-6.71893, 1.35966, -1.3779, -4.051)
    return P_CRIT * math.exp((b[0] * x + b[1] * x**1.5 + b[2] * x**2.5 + b[3] * x**5) / Tr)


def rho_liquid(T: float) -> float:
    Tr = _check(T)
    x = 1 - Tr
    b = (1.72328, -0.83950, 0.51060, -0.10412)
    return RHO_CRIT * math.exp(b[0] * x ** (1 / 3) + b[1] * x ** (2 / 3) + b[2] * x + b[3] * x ** (4 / 3))


def rho_vapour(T: float) -> float:
    Tr = _check(T)
    x = 1 / Tr - 1
    b = (-1.00900, -6.28792, 7.50332, -7.90463, 0.629427)
    return RHO_CRIT * math.exp(b[0] * x ** (1 / 3) + b[1] * x ** (2 / 3) + b[2] * x
                               + b[3] * x ** (4 / 3) + b[4] * x ** (5 / 3))


def h_liquid(T: float) -> float:
    """Specific enthalpy of saturated liquid, J/kg (arbitrary reference)."""
    Tr = _check(T)
    x = 1 - Tr
    b = (-200.0, 116.043, -917.225, 794.779, -589.587)
    return 1e3 * (b[0] + b[1] * x ** (1 / 3) + b[2] * x ** (2 / 3) + b[3] * x + b[4] * x ** (4 / 3))


def h_vapour(T: float) -> float:
    """Specific enthalpy of saturated vapour, J/kg (same reference as h_liquid)."""
    Tr = _check(T)
    x = 1 - Tr
    b = (-200.0, 440.055, -459.701, 434.081, -485.338)
    return 1e3 * (b[0] + b[1] * x ** (1 / 3) + b[2] * x ** (2 / 3) + b[3] * x + b[4] * x ** (4 / 3))


def h_fg(T: float) -> float:
    return h_vapour(T) - h_liquid(T)


def cp_liquid(T: float) -> float:
    """Isobaric specific heat of saturated liquid, J/(kg K)."""
    Tr = _check(T)
    x = 1 - Tr
    b = (2.49973, 0.023454, -3.80136, 13.0945, -14.5180)
    return 1e3 * b[0] * (1 + b[1] / x + b[2] * x + b[3] * x**2 + b[4] * x**3)


def u_liquid(T: float) -> float:
    return h_liquid(T) - p_sat(T) / rho_liquid(T)


def u_vapour(T: float) -> float:
    return h_vapour(T) - p_sat(T) / rho_vapour(T)


def _bisect(f, lo, hi, tol=1e-7, max_iter=200):
    flo, fhi = f(lo), f(hi)
    if flo == 0:
        return lo
    if fhi == 0:
        return hi
    if flo * fhi > 0:
        raise ValueError(f"Root not bracketed: f({lo})={flo:.4g}, f({hi})={fhi:.4g}")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if fm == 0 or (hi - lo) < tol:
            return mid
        if flo * fm < 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


class EquilibriumTank:
    """Saturated N2O tank emptying with a moving free volume."""

    T_MIN = T_TRIPLE + 0.5
    T_MAX = T_CRIT - 4.0

    def __init__(self, T0: float, m_liq0: float, V_free0: float):
        rl, rv = rho_liquid(T0), rho_vapour(T0)
        V_liq = m_liq0 / rl
        if V_liq > V_free0:
            raise ValueError(f"Tank overfilled: liquid {V_liq * 1e3:.2f} L > free volume {V_free0 * 1e3:.2f} L")
        m_vap0 = (V_free0 - V_liq) * rv
        self.T = T0
        self.V = V_free0
        self.m = m_liq0 + m_vap0
        self.U = m_liq0 * u_liquid(T0) + m_vap0 * u_vapour(T0)
        self.m_liq = m_liq0
        self.m_vap = m_vap0
        self.liquid_phase = True
        self.P = p_sat(T0)
        self.gamma = 1.27  # for the gas-phase fallback

    # --------------------------------------------------------------
    def _m_liq_for(self, T: float) -> float:
        rl, rv = rho_liquid(T), rho_vapour(T)
        return (self.V - self.m / rv) / (1 / rl - 1 / rv)

    def _energy_residual(self, T: float) -> float:
        ml = self._m_liq_for(T)
        return ml * u_liquid(T) + (self.m - ml) * u_vapour(T) - self.U

    def _solve_state(self) -> None:
        # Energy residual is monotonic increasing in T for a closed (m, V) system.
        try:
            T = _bisect(self._energy_residual, self.T_MIN, self.T_MAX)
        except ValueError:
            # Could not bracket: the contents are no longer two-phase.
            self._enter_gas_phase()
            return
        ml = self._m_liq_for(T)
        if ml <= 0.0:
            self._enter_gas_phase()
            return
        if ml > self.m:
            raise ValueError("Tank is liquid-full (compressed liquid); reduce load or free volume")
        self.T = T
        self.m_liq = ml
        self.m_vap = self.m - ml
        self.P = p_sat(T)

    def _enter_gas_phase(self) -> None:
        self.liquid_phase = False
        self.m_liq = 0.0
        self.m_vap = self.m

    # --------------------------------------------------------------
    def step(self, m_out: float, dV: float) -> None:
        """Remove `m_out` kg of liquid through the outlet; free volume grows by dV (m^3)."""
        if self.liquid_phase:
            work = self.P * dV
            self.U -= m_out * h_liquid(self.T) + work
            self.m -= m_out
            self.V += dV
            self._solve_state()
        else:
            # Polytropic (isentropic ideal-gas) blowdown of the remaining vapour.
            m_new = max(self.m - m_out, 1e-9)
            V_new = self.V + dV
            rho_old = self.m / self.V
            rho_new = m_new / V_new
            self.P *= (rho_new / rho_old) ** self.gamma
            self.m = m_new
            self.m_vap = m_new
            self.V = V_new

    @property
    def rho_l(self) -> float:
        return rho_liquid(self.T)

    @property
    def rho_v(self) -> float:
        return rho_vapour(self.T) if self.liquid_phase else self.m / self.V


def t_sat(P: float) -> float:
    """Saturation temperature for a given pressure, K."""
    return _bisect(lambda T: p_sat(T) - P, T_TRIPLE + 0.5, T_CRIT - 1.0)


class EmpiricalTank:
    """Prescribed pressure decay: P falls linearly with liquid mass from P0 to
    f_end*P0 at liquid depletion (mimics a spreadsheet-style "decay constant").

    The contents are assumed to sit on the saturation line at that pressure, so
    liquid and vapour densities follow T_sat(P). Mass is conserved: vapour
    needed to fill the free volume comes out of the liquid.
    """

    def __init__(self, T0: float, m_liq0: float, V_free0: float, end_fraction: float):
        self.T = T0
        self.P0 = p_sat(T0)
        self.P = self.P0
        self.f_end = end_fraction
        self.m_liq0 = m_liq0
        self.V = V_free0
        rl, rv = rho_liquid(T0), rho_vapour(T0)
        if m_liq0 / rl > V_free0:
            raise ValueError("Tank overfilled")
        self.m_liq = m_liq0
        self.m_vap = (V_free0 - m_liq0 / rl) * rv
        self.m = self.m_liq + self.m_vap
        self.liquid_phase = True

    def _partition(self) -> None:
        rl, rv = rho_liquid(self.T), rho_vapour(self.T)
        # m_liq/rl + m_vap/rv = V  and  m_liq + m_vap = m
        self.m_liq = (self.V - self.m / rv) / (1 / rl - 1 / rv)
        self.m_liq = min(max(self.m_liq, 0.0), self.m)
        self.m_vap = self.m - self.m_liq
        if self.m_liq <= 0.0:
            self.liquid_phase = False

    def step(self, m_out: float, dV: float) -> None:
        self.V += dV
        self.m -= m_out
        # two passes: pressure depends on liquid mass, densities on pressure
        for _ in range(2):
            self._partition()
            frac = self.m_liq / self.m_liq0
            self.P = self.P0 * (self.f_end + (1 - self.f_end) * frac)
            self.T = t_sat(self.P)

    @property
    def rho_l(self) -> float:
        return rho_liquid(self.T)

    @property
    def rho_v(self) -> float:
        return rho_vapour(self.T)
