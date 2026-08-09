"""Conservative height-field shallow-water solver for Phase 2."""

from .solver import ShallowWaterSolver, TerrainData, WaterfallFlux
from .particle_emitter import FluxParticleEmitter, ParticleBatch

__all__ = ["FluxParticleEmitter", "ParticleBatch", "ShallowWaterSolver", "TerrainData", "WaterfallFlux"]
