"""Write fixed-shape debug trajectories from SWE waterfall flux.

This is a pipeline teacher, not the final high-resolution SPH/MPM teacher.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

try:  # Package execution: python -m phase2.archive.prototypes.generate_debug_teachers
    from phase2.shallow_water import FluxParticleEmitter, ShallowWaterSolver, TerrainData
except ModuleNotFoundError:  # Direct execution from the repository root.
    from shallow_water import FluxParticleEmitter, ShallowWaterSolver, TerrainData


@dataclass(frozen=True)
class TrajectoryConfig:
    frames: int = 120
    dt: float = 1.0 / 30.0
    max_particles: int = 4096
    particle_mass_kg: float = 2.0
    air_drag_rate: float = 0.08
    restitution: float = 0.18
    max_age_s: float = 4.0
    seed: int = 0


class DebugTeacherWriter:
    def __init__(self, terrain: TerrainData, terrain_id: str, config: TrajectoryConfig):
        self.terrain = terrain
        self.terrain_id = terrain_id
        self.config = config
        self.swe = ShallowWaterSolver(terrain, initial_depth_m=0.015)
        self.emitter = FluxParticleEmitter(config.particle_mass_kg, seed=config.seed)
        n = config.max_particles
        self.position = np.zeros((n, 3), dtype=np.float64)
        self.velocity = np.zeros((n, 3), dtype=np.float64)
        self.active = np.zeros(n, dtype=bool)
        self.age = np.zeros(n, dtype=np.float64)
        self.impacts = np.zeros(n, dtype=np.uint8)
        self.next_slot = 0
        self.dropped_particles = 0

    def _sample_height_normal(self, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        col = np.clip((x + self.terrain.width_m * 0.5) / self.terrain.dx, 0, self.terrain.height.shape[1] - 1)
        row = np.clip(z / self.terrain.dz, 0, self.terrain.height.shape[0] - 1)
        c0 = np.floor(col).astype(int)
        r0 = np.floor(row).astype(int)
        c1 = np.minimum(c0 + 1, self.terrain.height.shape[1] - 1)
        r1 = np.minimum(r0 + 1, self.terrain.height.shape[0] - 1)
        tx, tz = col - c0, row - r0
        bed = (
            self.terrain.height[r0, c0] * (1 - tx) * (1 - tz)
            + self.terrain.height[r0, c1] * tx * (1 - tz)
            + self.terrain.height[r1, c0] * (1 - tx) * tz
            + self.terrain.height[r1, c1] * tx * tz
        )
        dz, dx = np.gradient(self.terrain.height, self.terrain.dz, self.terrain.dx)
        gx = dx[r0, c0]
        gz = dz[r0, c0]
        normal = np.column_stack((-gx, np.ones_like(gx), -gz))
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
        return bed, normal

    def _allocate(self, count: int) -> np.ndarray:
        if count <= 0:
            return np.empty(0, dtype=np.int64)
        free = np.flatnonzero(~self.active)
        chosen = free[:count]
        if chosen.size < count:
            self.dropped_particles += count - chosen.size
        return chosen

    def _emit(self, events) -> None:
        for event in events:
            batch = self.emitter.emit(event, event.duration_s)
            slots = self._allocate(batch.position.shape[0])
            if slots.size == 0:
                continue
            take = slots.size
            self.position[slots] = batch.position[:take]
            self.velocity[slots] = batch.velocity[:take]
            self.active[slots] = True
            self.age[slots] = 0.0
            self.impacts[slots] = 0

    def _advance_particles(self, dt: float) -> np.ndarray:
        splash_roi = np.zeros(self.config.max_particles, dtype=bool)
        ids = np.flatnonzero(self.active)
        if ids.size == 0:
            return splash_roi
        velocity = self.velocity[ids]
        velocity[:, 1] -= 9.81 * dt
        velocity *= np.exp(-self.config.air_drag_rate * dt)
        self.position[ids] += velocity * dt
        self.velocity[ids] = velocity
        self.age[ids] += dt

        bed, normal = self._sample_height_normal(self.position[ids, 0], self.position[ids, 2])
        hit = self.position[ids, 1] <= bed
        hit_ids = ids[hit]
        if hit_ids.size:
            n = normal[hit]
            v = self.velocity[hit_ids]
            vn = np.sum(v * n, axis=1)
            impact = np.maximum(-vn, 0.0)
            approaching = vn < 0.0
            v[approaching] -= ((1.0 + self.config.restitution) * vn[approaching])[:, None] * n[approaching]
            v[:, [0, 2]] *= 0.72
            self.velocity[hit_ids] = v
            self.position[hit_ids, 1] = bed[hit] + 0.01
            energetic = impact > 1.0
            splash_roi[hit_ids[energetic]] = True
            self.impacts[hit_ids] += 1
            settled = (self.impacts[hit_ids] >= 2) | (impact < 0.35)
            self.active[hit_ids[settled]] = False

        clearance = self.position[ids, 1] - bed
        splash_roi[ids[(clearance < 0.5) & (np.linalg.norm(self.velocity[ids], axis=1) > 2.0)]] = True
        outside = (
            (np.abs(self.position[ids, 0]) > self.terrain.width_m * 0.5)
            | (self.position[ids, 2] < 0.0)
            | (self.position[ids, 2] > self.terrain.length_m)
            | (self.age[ids] > self.config.max_age_s)
        )
        self.active[ids[outside]] = False
        return splash_roi

    def run(self, output: Path | str) -> dict[str, float | int | str]:
        cfg = self.config
        positions = np.zeros((cfg.frames, cfg.max_particles, 3), dtype=np.float32)
        velocities = np.zeros_like(positions)
        active_mask = np.zeros((cfg.frames, cfg.max_particles), dtype=bool)
        splash_roi = np.zeros_like(active_mask)
        flow_rate = np.full((cfg.frames, 1), self.terrain.source_flow_m3s, dtype=np.float32)
        initial_volume = self.swe.volume

        for frame in range(cfg.frames):
            events = self.swe.advance(cfg.dt)
            self._emit(events)
            roi = self._advance_particles(cfg.dt)
            positions[frame] = self.position
            velocities[frame] = self.velocity
            active_mask[frame] = self.active
            splash_roi[frame] = roi & self.active

        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": 1,
            "teacher_kind": "debug_swe_ballistic",
            "not_final_training_teacher": True,
            "terrain_id": self.terrain_id,
            "config": asdict(cfg),
            "dropped_particles": self.dropped_particles,
            "swe_mass_balance_error_m3": self.swe.mass_balance_error(initial_volume),
        }
        np.savez_compressed(
            output,
            positions=positions,
            velocities=velocities,
            particle_id=np.arange(cfg.max_particles, dtype=np.int32),
            particle_type=np.zeros(cfg.max_particles, dtype=np.uint8),
            active_mask=active_mask,
            splash_roi=splash_roi,
            terrain_id=np.asarray(self.terrain_id),
            flow_rate=flow_rate,
            dt=np.asarray(cfg.dt, dtype=np.float32),
            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        )
        return metadata
