"""Finite-volume shallow-water solver with conservative waterfall extraction."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image


GRAVITY = 9.81


@dataclass(frozen=True)
class TerrainData:
    height: np.ndarray
    cliff: np.ndarray
    channel: np.ndarray
    source: np.ndarray
    dx: float
    dz: float
    width_m: float
    length_m: float
    source_flow_m3s: float
    source_velocity_xz: tuple[float, float]

    @classmethod
    def load(cls, directory: Path | str) -> "TerrainData":
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        spec = metadata["spec"]
        dx, dz = map(float, metadata["cell_size_m"])

        def mask(name: str) -> np.ndarray:
            return np.asarray(Image.open(directory / name), dtype=np.uint8) > 0

        velocity = metadata["water_source"]["initial_velocity_mps"]
        return cls(
            height=np.load(directory / "height_meters.npy").astype(np.float64),
            cliff=mask("cliff_mask.png"),
            channel=mask("channel_mask.png"),
            source=mask("source_mask.png"),
            dx=dx,
            dz=dz,
            width_m=float(spec["world_width_m"]),
            length_m=float(spec["world_length_m"]),
            source_flow_m3s=float(metadata["water_source"]["flow_rate_m3s"]),
            source_velocity_xz=(float(velocity[0]), float(velocity[2])),
        )


@dataclass(frozen=True)
class WaterfallFlux:
    """Flux leaving the 2D domain through an upstream-to-cliff face."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    discharge_m3s: np.ndarray
    velocity_xyz: np.ndarray
    time_s: float = 0.0
    duration_s: float = 0.0

    @property
    def total_discharge_m3s(self) -> float:
        return float(np.sum(self.discharge_m3s))


def _physical_flux(h: np.ndarray, momentum: np.ndarray, transverse: np.ndarray) -> tuple[np.ndarray, ...]:
    safe_h = np.maximum(h, 1e-9)
    velocity = np.where(h > 1e-8, momentum / safe_h, 0.0)
    transverse_velocity = np.where(h > 1e-8, transverse / safe_h, 0.0)
    return (
        momentum,
        momentum * velocity + 0.5 * GRAVITY * h * h,
        momentum * transverse_velocity,
    )


def _rusanov(
    h_l: np.ndarray,
    mn_l: np.ndarray,
    mt_l: np.ndarray,
    h_r: np.ndarray,
    mn_r: np.ndarray,
    mt_r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f_l = _physical_flux(h_l, mn_l, mt_l)
    f_r = _physical_flux(h_r, mn_r, mt_r)
    u_l = np.where(h_l > 1e-8, mn_l / np.maximum(h_l, 1e-9), 0.0)
    u_r = np.where(h_r > 1e-8, mn_r / np.maximum(h_r, 1e-9), 0.0)
    wave = np.maximum(np.abs(u_l) + np.sqrt(GRAVITY * h_l), np.abs(u_r) + np.sqrt(GRAVITY * h_r))
    return tuple(
        0.5 * (left + right) - 0.5 * wave * (q_r - q_l)
        for left, right, q_l, q_r in zip(f_l, f_r, (h_l, mn_l, mt_l), (h_r, mn_r, mt_r))
    )


class ShallowWaterSolver:
    """First-order conservative SWE solver intended as the Phase 2 baseline."""

    def __init__(self, terrain: TerrainData, initial_depth_m: float = 0.0, cfl: float = 0.35):
        self.terrain = terrain
        self.cfl = float(cfl)
        self.h = np.where(terrain.channel & ~terrain.cliff, initial_depth_m, 0.0).astype(np.float64)
        self.hu = np.zeros_like(self.h)
        self.hv = np.zeros_like(self.h)
        self.active = ~terrain.cliff
        self.time = 0.0
        self.injected_volume = 0.0
        self.waterfall_volume = 0.0
        self.boundary_volume = 0.0
        self.last_waterfall_flux = self._empty_flux()

    def _empty_flux(self) -> WaterfallFlux:
        empty = np.empty(0, dtype=np.float64)
        return WaterfallFlux(empty, empty, empty, empty, np.empty((0, 3), dtype=np.float64), self.time, 0.0)

    @property
    def volume(self) -> float:
        return float(np.sum(self.h) * self.terrain.dx * self.terrain.dz)

    def stable_dt(self, maximum: float = 0.02) -> float:
        wet = self.h > 1e-5
        if not np.any(wet):
            return maximum
        u = np.zeros_like(self.h)
        v = np.zeros_like(self.h)
        u[wet] = self.hu[wet] / self.h[wet]
        v[wet] = self.hv[wet] / self.h[wet]
        c = np.sqrt(GRAVITY * self.h)
        rate = np.max((np.abs(u) + c) / self.terrain.dx + (np.abs(v) + c) / self.terrain.dz)
        return min(maximum, self.cfl / max(float(rate), 1e-9))

    def advance(self, duration: float, maximum_dt: float = 0.02) -> list[WaterfallFlux]:
        remaining = float(duration)
        events: list[WaterfallFlux] = []
        while remaining > 1e-10:
            dt = min(remaining, self.stable_dt(maximum_dt))
            flux = self.step(dt)
            if flux.total_discharge_m3s > 0.0:
                events.append(flux)
            remaining -= dt
        return events

    def _inject_source(self, dt: float) -> None:
        cells = self.terrain.source & self.active
        count = int(np.count_nonzero(cells))
        if count == 0:
            return
        volume = self.terrain.source_flow_m3s * dt
        depth = volume / (count * self.terrain.dx * self.terrain.dz)
        self.h[cells] += depth
        vx, vz = self.terrain.source_velocity_xz
        self.hu[cells] += depth * vx
        self.hv[cells] += depth * vz
        self.injected_volume += volume

    def step(self, dt: float) -> WaterfallFlux:
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self._inject_source(dt)
        h, hu, hv = self.h, self.hu, self.hv
        dh = np.zeros_like(h)
        dhu = np.zeros_like(h)
        dhv = np.zeros_like(h)

        # x faces: momentum normal to the face is hu.
        fx = _rusanov(h[:, :-1], hu[:, :-1], hv[:, :-1], h[:, 1:], hu[:, 1:], hv[:, 1:])
        valid_x = self.active[:, :-1] & self.active[:, 1:]
        for field, flux in zip((dh, dhu, dhv), fx):
            applied = np.where(valid_x, flux, 0.0) * dt / self.terrain.dx
            field[:, :-1] -= applied
            field[:, 1:] += applied

        # z faces: momentum normal to the face is hv; swap transverse momentum.
        fz_h, fz_hv, fz_hu = _rusanov(h[:-1], hv[:-1], hu[:-1], h[1:], hv[1:], hu[1:])
        upstream = self.active[:-1] & self.terrain.cliff[1:]
        regular_z = self.active[:-1] & self.active[1:]
        for field, flux in ((dh, fz_h), (dhv, fz_hv), (dhu, fz_hu)):
            applied = np.where(regular_z, flux, 0.0) * dt / self.terrain.dz
            field[:-1] -= applied
            field[1:] += applied

        # Extract only positive downstream discharge at a cliff. It leaves the
        # 2D control volume and becomes a 3D waterfall source.
        raw_discharge = np.maximum(fz_h, 0.0)
        cliff_discharge = np.where(upstream, raw_discharge, 0.0)
        dh[:-1] -= cliff_discharge * dt / self.terrain.dz

        # Bed-slope source. Limiting prevents a single steep/noisy cell from
        # injecting more momentum than gravity can produce over one step.
        bed_dz, bed_dx = np.gradient(self.terrain.height, self.terrain.dz, self.terrain.dx)
        ax = np.clip(-GRAVITY * bed_dx, -GRAVITY, GRAVITY)
        az = np.clip(-GRAVITY * bed_dz, -GRAVITY, GRAVITY)
        dhu += dt * h * ax * self.active
        dhv += dt * h * az * self.active

        new_h = h + dh
        negative = new_h < 0.0
        new_h[negative] = 0.0
        new_h[~self.active] = 0.0
        wet = (new_h > 1e-5) & self.active
        next_hu = np.where(wet, hu + dhu, 0.0)
        next_hv = np.where(wet, hv + dhv, 0.0)
        speed = np.zeros_like(new_h)
        speed[wet] = np.hypot(next_hu[wet], next_hv[wet]) / new_h[wet]
        limiter = np.ones_like(new_h)
        limiter[wet] = np.minimum(1.0, 20.0 / np.maximum(speed[wet], 1e-9))
        self.hu = next_hu * limiter
        self.hv = next_hv * limiter
        new_h[~wet] = 0.0
        self.h = new_h

        # Simple open downstream boundary; removed volume is tracked.
        outlet_depth = self.h[-1].copy()
        outgoing = np.maximum(self.hv[-1], 0.0)
        removed = np.minimum(outlet_depth, outgoing * dt / self.terrain.dz)
        self.h[-1] -= removed
        self.boundary_volume += float(np.sum(removed) * self.terrain.dx * self.terrain.dz)

        rows, cols = np.nonzero(cliff_discharge > 0.0)
        if rows.size:
            q = cliff_discharge[rows, cols] * self.terrain.dx
            local_h = np.maximum(h[rows, cols], 1e-8)
            vx = hu[rows, cols] / local_h
            vz = hv[rows, cols] / local_h
            x = -0.5 * self.terrain.width_m + (cols + 0.5) * self.terrain.dx
            z = (rows + 1.0) * self.terrain.dz
            y = self.terrain.height[rows, cols] + h[rows, cols]
            velocity = np.column_stack((vx, np.zeros_like(vx), np.maximum(vz, 0.0)))
            result = WaterfallFlux(x, y, z, q, velocity, self.time + dt, dt)
            self.waterfall_volume += result.total_discharge_m3s * dt
        else:
            result = self._empty_flux()
        self.last_waterfall_flux = result
        self.time += dt
        return result

    def mass_balance_error(self, initial_volume: float = 0.0) -> float:
        expected = initial_volume + self.injected_volume - self.waterfall_volume - self.boundary_volume
        return self.volume - expected
