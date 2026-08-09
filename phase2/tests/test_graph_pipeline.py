from __future__ import annotations

from pathlib import Path
import unittest

import torch

from phase2.gnn import ResidualGNS, TrajectoryGraphDataset


class GraphPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.trajectory = cls.root / "datasets" / "debug" / "trajectory_000_single_cliff.npz"

    def test_graph_shapes_and_forward(self) -> None:
        if not self.trajectory.exists():
            self.skipTest("debug trajectory has not been generated")
        dataset = TrajectoryGraphDataset([self.trajectory], self.root / "terrains")
        self.assertGreater(len(dataset), 0)
        sample = dataset[0]
        self.assertEqual(sample.node_features.shape[1], 33)
        self.assertEqual(sample.edge_features.shape[1], 8)
        batch = dataset.to_torch(sample)
        model = ResidualGNS(33, 8, hidden_size=16, blocks=2)
        output = model(batch["node_features"], batch["edge_features"], batch["edge_index"])
        self.assertEqual(output.shape, batch["target_delta_v"].shape)
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
