from __future__ import annotations

import unittest

import numpy as np

from phase2.shallow_water import FluxParticleEmitter, ShallowWaterSolver, TerrainData, WaterfallFlux


def terrain(*, cliff: bool = False, source_flow: float = 0.0) -> TerrainData:
    shape = (24, 16)
    height = np.zeros(shape, dtype=np.float64)
    height += np.linspace(0.08, 0.0, shape[0])[:, None]
    cliff_mask = np.zeros(shape, dtype=bool)
    if cliff:
        cliff_mask[13:15] = True
    channel = np.ones(shape, dtype=bool)
    source = np.zeros(shape, dtype=bool)
    source[:2, 6:10] = True
    return TerrainData(height, cliff_mask, channel, source, 0.25, 0.25, 4.0, 6.0, source_flow, (0.0, 1.0))


class ShallowWaterTests(unittest.TestCase):
    def test_still_water_remains_nonnegative(self) -> None:
        solver = ShallowWaterSolver(terrain(), initial_depth_m=0.05)
        solver.advance(0.2)
        self.assertTrue(np.isfinite(solver.h).all())
        self.assertGreaterEqual(float(solver.h.min()), 0.0)

    def test_source_volume_is_accounted_for(self) -> None:
        solver = ShallowWaterSolver(terrain(source_flow=0.2))
        initial = solver.volume
        solver.advance(0.1)
        error = abs(solver.mass_balance_error(initial))
        self.assertLess(error, 2e-3)

    def test_cliff_produces_waterfall_flux(self) -> None:
        solver = ShallowWaterSolver(terrain(cliff=True), initial_depth_m=0.08)
        solver.hv[:13] = solver.h[:13] * 1.5
        events = solver.advance(0.2)
        self.assertTrue(events)
        self.assertGreater(solver.waterfall_volume, 0.0)
        self.assertTrue(all(np.all(event.velocity_xyz[:, 2] >= 0.0) for event in events))

    def test_flux_particle_emitter_preserves_fractional_mass(self) -> None:
        flux = WaterfallFlux(
            np.array([0.0]), np.array([2.0]), np.array([1.0]), np.array([0.001]), np.array([[0.0, 0.0, 1.0]])
        )
        emitter = FluxParticleEmitter(particle_mass_kg=0.25, seed=1)
        batches = [emitter.emit(flux, 0.1) for _ in range(10)]
        represented = sum(batch.total_mass_kg for batch in batches) + emitter.residual_mass_kg
        self.assertAlmostEqual(represented, 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
