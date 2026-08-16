"""Conservative 2D shallow-water baseline for projected 3D particle scenes."""

from __future__ import annotations

import numpy as np


class ProjectedSWESolver:
    """Project particles to an x/z height field and evolve conservative SWE.

    Water-3D uses dataset-coordinate displacement per step rather than SI
    velocity. Accordingly ``gravity`` is the per-step acceleration magnitude
    from metadata, while the same finite-volume equations are retained.
    """

    def __init__(self, bounds: np.ndarray, gravity: float, resolution: int = 64, cfl: float = 0.35):
        self.bounds = np.asarray(bounds, np.float64)
        self.nx = self.nz = int(resolution)
        self.dx = float((self.bounds[0, 1] - self.bounds[0, 0]) / self.nx)
        self.dz = float((self.bounds[2, 1] - self.bounds[2, 0]) / self.nz)
        self.gravity = max(float(abs(gravity)), 1e-8)
        self.cfl = float(cfl)
        shape = (self.nz, self.nx)
        self.h = np.zeros(shape, np.float64)
        self.hu = np.zeros(shape, np.float64)
        self.hv = np.zeros(shape, np.float64)
        self.bed = np.full(shape, self.bounds[1, 0], np.float64)
        self.initial_volume = 0.0
        self.particle_position: np.ndarray | None = None
        self.vertical_fraction: np.ndarray | None = None
        self.fluid_mask: np.ndarray | None = None

    def _cells(self, position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ix = np.floor((position[:, 0] - self.bounds[0, 0]) / self.dx).astype(np.int64)
        iz = np.floor((position[:, 2] - self.bounds[2, 0]) / self.dz).astype(np.int64)
        return np.clip(ix, 0, self.nx - 1), np.clip(iz, 0, self.nz - 1)

    def initialize(self, position: np.ndarray, velocity: np.ndarray, fluid_mask: np.ndarray, boundary_position: np.ndarray | None = None) -> None:
        position = np.asarray(position, np.float64)
        velocity = np.asarray(velocity, np.float64)
        fluid_mask = np.asarray(fluid_mask, bool)
        if boundary_position is not None and len(boundary_position):
            boundary = np.asarray(boundary_position, np.float64)
            bx, bz = self._cells(boundary)
            interior = (boundary[:, 1] > self.bounds[1, 0] + 1e-6) & (boundary[:, 1] < self.bounds[1, 1] - 1e-6)
            np.maximum.at(self.bed, (bz[interior], bx[interior]), boundary[interior, 1])
        fluid = position[fluid_mask]
        fluid_velocity = velocity[fluid_mask]
        ix, iz = self._cells(fluid)
        count = np.zeros_like(self.h)
        np.add.at(count, (iz, ix), 1.0)
        surface = self.bed.copy()
        np.maximum.at(surface, (iz, ix), fluid[:, 1])
        observed_depth = np.maximum(surface - self.bed, 0.0)
        observed_volume = float(np.sum(observed_depth * self.dx * self.dz))
        particle_volume = observed_volume / max(len(fluid), 1)
        if particle_volume <= 1e-12:
            particle_volume = 0.125 * min(self.dx, self.dz) ** 3
        np.add.at(self.h, (iz, ix), particle_volume / (self.dx * self.dz))
        np.add.at(self.hu, (iz, ix), particle_volume / (self.dx * self.dz) * fluid_velocity[:, 0])
        np.add.at(self.hv, (iz, ix), particle_volume / (self.dx * self.dz) * fluid_velocity[:, 2])
        depth_at_particle = np.maximum(observed_depth[iz, ix], 1e-9)
        fraction = np.clip((fluid[:, 1] - self.bed[iz, ix]) / depth_at_particle, 0.0, 1.0)
        self.particle_position = position.copy()
        self.vertical_fraction = fraction
        self.fluid_mask = fluid_mask.copy()
        self.initial_volume = self.volume

    @property
    def volume(self) -> float:
        return float(np.sum(self.h) * self.dx * self.dz)

    def _physical_flux(self, h: np.ndarray, normal: np.ndarray, transverse: np.ndarray):
        safe = np.maximum(h, 1e-12)
        u = np.where(h > 1e-10, normal / safe, 0.0)
        vt = np.where(h > 1e-10, transverse / safe, 0.0)
        return normal, normal * u + 0.5 * self.gravity * h * h, normal * vt

    def _rusanov(self, hl, mnl, mtl, hr, mnr, mtr):
        fl, fr = self._physical_flux(hl, mnl, mtl), self._physical_flux(hr, mnr, mtr)
        ul = np.where(hl > 1e-10, mnl / np.maximum(hl, 1e-12), 0.0)
        ur = np.where(hr > 1e-10, mnr / np.maximum(hr, 1e-12), 0.0)
        wave = np.maximum(np.abs(ul) + np.sqrt(self.gravity * hl), np.abs(ur) + np.sqrt(self.gravity * hr))
        return tuple(0.5 * (a + b) - 0.5 * wave * (qr - ql)
                     for a, b, ql, qr in zip(fl, fr, (hl, mnl, mtl), (hr, mnr, mtr)))

    def stable_dt(self, maximum: float = 1.0) -> float:
        wet = self.h > 1e-10
        if not np.any(wet):
            return maximum
        u = np.where(wet, self.hu / np.maximum(self.h, 1e-12), 0.0)
        v = np.where(wet, self.hv / np.maximum(self.h, 1e-12), 0.0)
        c = np.sqrt(self.gravity * self.h)
        rate = np.max((np.abs(u) + c) / self.dx + (np.abs(v) + c) / self.dz)
        return min(maximum, self.cfl / max(float(rate), 1e-12))

    def step(self, dt: float) -> None:
        old_volume = self.volume
        h, hu, hv = self.h, self.hu, self.hv
        dh = np.zeros_like(h); dhu = np.zeros_like(h); dhv = np.zeros_like(h)
        fx = self._rusanov(h[:, :-1], hu[:, :-1], hv[:, :-1], h[:, 1:], hu[:, 1:], hv[:, 1:])
        for field, flux in zip((dh, dhu, dhv), fx):
            applied = flux * dt / self.dx
            field[:, :-1] -= applied; field[:, 1:] += applied
        fz_h, fz_hv, fz_hu = self._rusanov(h[:-1], hv[:-1], hu[:-1], h[1:], hv[1:], hu[1:])
        for field, flux in ((dh, fz_h), (dhv, fz_hv), (dhu, fz_hu)):
            applied = flux * dt / self.dz
            field[:-1] -= applied; field[1:] += applied
        bed_dz, bed_dx = np.gradient(self.bed, self.dz, self.dx)
        dhu += dt * h * np.clip(-self.gravity * bed_dx, -self.gravity, self.gravity)
        dhv += dt * h * np.clip(-self.gravity * bed_dz, -self.gravity, self.gravity)
        new_h = np.maximum(h + dh, 0.0)
        new_volume = float(np.sum(new_h) * self.dx * self.dz)
        if new_volume > 0.0:
            new_h *= old_volume / new_volume
        wet = new_h > 1e-10
        self.hu = np.where(wet, (hu + dhu) * 0.999, 0.0)
        self.hv = np.where(wet, (hv + dhv) * 0.999, 0.0)
        self.h = new_h

    def advance(self, duration: float = 1.0) -> None:
        remaining = float(duration)
        while remaining > 1e-10:
            dt = min(remaining, self.stable_dt(remaining))
            self.step(dt)
            remaining -= dt

    def advance_particles(self, duration: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        if self.particle_position is None or self.fluid_mask is None or self.vertical_fraction is None:
            raise RuntimeError("initialize must be called first")
        self.advance(duration)
        output_position = self.particle_position.copy()
        output_velocity = np.zeros_like(output_position)
        fluid_position = output_position[self.fluid_mask]
        ix, iz = self._cells(fluid_position)
        wet_depth = np.maximum(self.h[iz, ix], 1e-12)
        ux = self.hu[iz, ix] / wet_depth
        uz = self.hv[iz, ix] / wet_depth
        fluid_position[:, 0] = np.clip(fluid_position[:, 0] + ux * duration, self.bounds[0, 0], self.bounds[0, 1])
        fluid_position[:, 2] = np.clip(fluid_position[:, 2] + uz * duration, self.bounds[2, 0], self.bounds[2, 1])
        next_ix, next_iz = self._cells(fluid_position)
        fluid_position[:, 1] = self.bed[next_iz, next_ix] + self.vertical_fraction * self.h[next_iz, next_ix]
        output_velocity[self.fluid_mask] = fluid_position - output_position[self.fluid_mask]
        output_position[self.fluid_mask] = fluid_position
        self.particle_position = output_position
        return output_position.astype(np.float32), output_velocity.astype(np.float32)
