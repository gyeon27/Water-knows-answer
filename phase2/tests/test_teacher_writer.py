from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from phase2.teacher import DebugTeacherWriter, TrajectoryConfig
from test_shallow_water import terrain


class TeacherWriterTests(unittest.TestCase):
    def test_schema_and_shapes(self) -> None:
        config = TrajectoryConfig(frames=4, dt=0.02, max_particles=32, particle_mass_kg=0.1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "teacher.npz"
            DebugTeacherWriter(terrain(cliff=True, source_flow=0.1), "synthetic", config).run(output)
            with np.load(output) as data:
                self.assertEqual(data["positions"].shape, (4, 32, 3))
                self.assertEqual(data["velocities"].shape, (4, 32, 3))
                self.assertEqual(data["active_mask"].shape, (4, 32))
                self.assertEqual(data["splash_roi"].shape, (4, 32))
                self.assertEqual(str(data["terrain_id"]), "synthetic")


if __name__ == "__main__":
    unittest.main()
