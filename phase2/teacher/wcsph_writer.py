"""SWE-fed WCSPH trajectory writer using the common GNN dataset schema."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np

try:
    from phase2.shallow_water import FluxParticleEmitter, ShallowWaterSolver, TerrainData
except ModuleNotFoundError:
    from shallow_water import FluxParticleEmitter, ShallowWaterSolver, TerrainData

from .trajectory_writer import TrajectoryConfig
from .wcsph_solver import WCSPHSolver


class WCSPHTeacherWriter:
    def __init__(self, terrain: TerrainData, terrain_id: str, config: TrajectoryConfig):
        self.terrain, self.terrain_id, self.config = terrain, terrain_id, config
        self.swe = ShallowWaterSolver(terrain, initial_depth_m=0.015)
        self.emitter = FluxParticleEmitter(config.particle_mass_kg, seed=config.seed)
        # Keep roughly the same kernel support in particle-spacing units as the
        # parcel mass changes; the floor avoids an impractically large CPU grid.
        smoothing_radius = max(0.20, 2.2 * math.pow(config.particle_mass_kg / 1000.0, 1.0 / 3.0))
        self.sph = WCSPHSolver(
            terrain.height,
            terrain.dx,
            terrain.dz,
            terrain.width_m,
            terrain.length_m,
            config.max_particles,
            config.particle_mass_kg,
            smoothing_radius_m=smoothing_radius,
            air_drag_rate=config.air_drag_rate,
            restitution=config.restitution,
        )
        self.dropped_particles = 0

    def _emit(self, events) -> None:
        _, _, active = self.sph.snapshot()
        free = np.flatnonzero(~active)
        cursor = 0
        for event in events:
            batch = self.emitter.emit(event, event.duration_s)
            take = min(batch.position.shape[0], free.size - cursor)
            if take <= 0:
                self.dropped_particles += batch.position.shape[0]
                continue
            ids = free[cursor : cursor + take].astype(np.int32)
            self.sph.activate(ids, batch.position[:take].astype(np.float32), batch.velocity[:take].astype(np.float32), take)
            cursor += take
            self.dropped_particles += batch.position.shape[0] - take

    def run(self, output: Path | str) -> dict:
        cfg = self.config
        positions = np.zeros((cfg.frames, cfg.max_particles, 3), np.float32)
        velocities = np.zeros_like(positions)
        active_mask = np.zeros((cfg.frames, cfg.max_particles), bool)
        splash_roi = np.zeros_like(active_mask)
        initial_volume = self.swe.volume
        for frame in range(cfg.frames):
            self._emit(self.swe.advance(cfg.dt))
            self.sph.step(cfg.dt, substeps=3, max_age=cfg.max_age_s)
            p, v, active = self.sph.snapshot()
            positions[frame], velocities[frame], active_mask[frame] = p, v, active
            ids = np.flatnonzero(active)
            if ids.size:
                bed, _ = self._terrain_sample(p[ids, 0], p[ids, 2])
                clearance = p[ids, 1] - bed
                splash_roi[frame, ids] = (clearance < 0.45) & (np.linalg.norm(v[ids], axis=1) > 1.0)
        metadata = {
            "schema_version": 1,
            "teacher_kind": "wcsph",
            "teacher_status": "candidate_requires_calibration",
            "not_final_training_teacher": True,
            "terrain_id": self.terrain_id,
            "config": asdict(cfg),
            "dropped_particles": self.dropped_particles,
            "swe_mass_balance_error_m3": self.swe.mass_balance_error(initial_volume),
        }
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            positions=positions,
            velocities=velocities,
            particle_id=np.arange(cfg.max_particles, dtype=np.int32),
            particle_type=np.zeros(cfg.max_particles, dtype=np.uint8),
            active_mask=active_mask,
            splash_roi=splash_roi,
            terrain_id=np.asarray(self.terrain_id),
            flow_rate=np.full((cfg.frames, 1), self.terrain.source_flow_m3s, np.float32),
            dt=np.asarray(cfg.dt, np.float32),
            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        )
        return metadata

    def _terrain_sample(self, x: np.ndarray, z: np.ndarray):
        col = np.clip(np.rint((x + self.terrain.width_m * 0.5) / self.terrain.dx).astype(int), 0, self.terrain.height.shape[1] - 1)
        row = np.clip(np.rint(z / self.terrain.dz).astype(int), 0, self.terrain.height.shape[0] - 1)
        return self.terrain.height[row, col], (row, col)
