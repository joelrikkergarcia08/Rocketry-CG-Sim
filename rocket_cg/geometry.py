"""Axisymmetric geometry helpers for propellant centre-of-gravity calculations.

Convention
----------
z is the axial coordinate in metres, increasing AFT (toward the engine).
z = 0 is the datum defined in the config (default: forward inside face of the
oxidizer tank). Liquids are assumed to occupy the AFT-most part of their
region (settled under thrust, or pushed there by a piston).

Every fluid region is described by its cross-sectional area A(z) sampled on a
uniform grid. Volumes and centroids are then simple cumulative integrals, which
handles annular regions, piecewise geometry and (later) domes without special
cases.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class FillResult:
    volume: float        # m^3 actually placed
    z_top: float         # forward (upper) surface of the liquid column
    z_bottom: float      # aft end of the region
    centroid: float      # z of the liquid's centre of mass


class AxialProfile:
    """Cross-sectional area A(z) of one fluid region on a uniform z grid."""

    def __init__(self, z_min: float, z_max: float, n: int = 6000):
        if z_max <= z_min:
            raise ValueError("z_max must exceed z_min")
        self.z = np.linspace(z_min, z_max, n)
        self.dz = self.z[1] - self.z[0]
        self.area = np.zeros_like(self.z)

    # -- construction -----------------------------------------------------
    def add_cylinder(self, z0: float, z1: float, r_outer: float, r_inner: float = 0.0) -> "AxialProfile":
        """Add a (possibly annular) cylindrical region between z0 and z1."""
        if z1 < z0:
            z0, z1 = z1, z0
        if r_inner >= r_outer:
            raise ValueError("r_inner must be smaller than r_outer")
        mask = (self.z >= z0 - 1e-12) & (self.z <= z1 + 1e-12)
        self.area[mask] += math.pi * (r_outer**2 - r_inner**2)
        return self

    # -- integrals --------------------------------------------------------
    def _cum_from_aft(self, integrand: np.ndarray) -> np.ndarray:
        """Cumulative integral from z_max back toward z_min (trapezoid)."""
        seg = 0.5 * (integrand[1:] + integrand[:-1]) * self.dz
        out = np.zeros_like(integrand)
        out[:-1] = np.cumsum(seg[::-1])[::-1]
        return out

    @property
    def total_volume(self) -> float:
        return float(np.trapezoid(self.area, self.z))

    def volume_between(self, z0: float, z1: float) -> float:
        if z1 < z0:
            z0, z1 = z1, z0
        zz = np.clip(self.z, z0, z1)
        # integrate area over [z0, z1] using interpolation at the bounds
        a0 = float(np.interp(z0, self.z, self.area))
        a1 = float(np.interp(z1, self.z, self.area))
        inner = (self.z > z0) & (self.z < z1)
        zs = np.concatenate(([z0], self.z[inner], [z1]))
        As = np.concatenate(([a0], self.area[inner], [a1]))
        return float(np.trapezoid(As, zs))

    def centroid_between(self, z0: float, z1: float) -> float:
        if z1 < z0:
            z0, z1 = z1, z0
        a0 = float(np.interp(z0, self.z, self.area))
        a1 = float(np.interp(z1, self.z, self.area))
        inner = (self.z > z0) & (self.z < z1)
        zs = np.concatenate(([z0], self.z[inner], [z1]))
        As = np.concatenate(([a0], self.area[inner], [a1]))
        V = float(np.trapezoid(As, zs))
        if V <= 0:
            return 0.5 * (z0 + z1)
        return float(np.trapezoid(As * zs, zs) / V)

    def fill_from_aft(self, volume: float) -> FillResult:
        """Place `volume` of liquid at the aft end of the region.

        Returns the liquid surface position and centroid. Raises if the region
        cannot hold the volume.
        """
        z_bottom = float(self.z[-1])
        if volume <= 0:
            return FillResult(0.0, z_bottom, z_bottom, z_bottom)
        cumV = self._cum_from_aft(self.area)            # volume aft of each z
        if volume > cumV[0] * (1 + 1e-9):
            raise ValueError(f"Region overfilled: asked {volume:.5f} m^3, capacity {cumV[0]:.5f} m^3")
        # cumV decreases with index; interpolate on reversed arrays
        z_top = float(np.interp(volume, cumV[::-1], self.z[::-1]))
        cumM = self._cum_from_aft(self.area * self.z)   # first moment aft of each z
        M = float(np.interp(z_top, self.z, cumM))
        return FillResult(volume, z_top, z_bottom, M / volume)


def circle_area(d: float) -> float:
    return math.pi * d * d / 4.0


def annulus_area(d_outer: float, d_inner: float) -> float:
    return math.pi * (d_outer**2 - d_inner**2) / 4.0
