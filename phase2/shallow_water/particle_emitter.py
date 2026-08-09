"""Convert conservative waterfall flux into fixed-mass 3D particles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .solver import WaterfallFlux


@dataclass(frozen=True)
class ParticleBatch:
    position: np.ndarray
    velocity: np.ndarray
    mass_kg: np.ndarray

    @property
    def total_mass_kg(self) -> float:
        return float(np.sum(self.mass_kg))


class FluxParticleEmitter:
    """Stateful volume accumulator; no water is lost to fractional particles."""

    def __init__(self, particle_mass_kg: float = 0.25, water_density_kgm3: float = 1000.0, seed: int = 0):
        if particle_mass_kg <= 0.0 or water_density_kgm3 <= 0.0:
            raise ValueError("particle mass and density must be positive")
        self.particle_mass_kg = float(particle_mass_kg)
        self.water_density_kgm3 = float(water_density_kgm3)
        self.residual_mass_kg = 0.0
        self.rng = np.random.default_rng(seed)

    def emit(self, flux: WaterfallFlux, dt: float, spacing_m: float = 0.04) -> ParticleBatch:
        empty = ParticleBatch(np.empty((0, 3)), np.empty((0, 3)), np.empty(0))
        if flux.x.size == 0 or dt <= 0.0:
            return empty
        face_mass = flux.discharge_m3s * float(dt) * self.water_density_kgm3
        available = float(np.sum(face_mass)) + self.residual_mass_kg
        count = int(available // self.particle_mass_kg)
        self.residual_mass_kg = available - count * self.particle_mass_kg
        if count == 0:
            return empty

        weights = face_mass / max(float(np.sum(face_mass)), 1e-12)
        faces = self.rng.choice(flux.x.size, size=count, p=weights)
        jitter = self.rng.uniform(-spacing_m * 0.5, spacing_m * 0.5, size=(count, 2))
        position = np.column_stack((flux.x[faces] + jitter[:, 0], flux.y[faces], flux.z[faces] + jitter[:, 1]))
        velocity = flux.velocity_xyz[faces].copy()
        mass = np.full(count, self.particle_mass_kg, dtype=np.float64)
        return ParticleBatch(position, velocity, mass)
