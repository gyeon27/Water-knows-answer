from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from phase3.config import Phase3Config
from phase3.data import UnifiedGraph, batch_graphs, deterministic_windows, graph_from_gns, radius_graph
from phase3.gns_data_adapter import iter_tfrecord, decode_sequence_example
from phase3.models import UnifiedGNS
from phase3.training import physics_informed_loss


class Phase3Test(unittest.TestCase):
    def test_deterministic_windows(self):
        self.assertEqual(deterministic_windows(100, 10, 7), deterministic_windows(100, 10, 7))
        self.assertEqual(len(deterministic_windows(100, 10, 7)), 10)

    def test_radius_graph(self):
        position = np.asarray(((0, 0, 0), (0.1, 0, 0), (2, 0, 0)), np.float32)
        edge_index, edge = radius_graph(position, 0.2, 48)
        self.assertEqual(edge_index.shape, (2, 2))
        self.assertEqual(edge.shape, (2, 4))

    def test_common_feature_shape_and_no_future_feature_leak(self):
        frames, particles = 12, 16
        rng = np.random.default_rng(4)
        position = np.cumsum(rng.normal(0, 0.01, (frames, particles, 3)), axis=0).astype(np.float32)
        record = {"position": position.copy(), "particle_type": np.zeros(particles, np.int64), "key": 1}
        metadata = {
            "bounds": [[-2, 2], [-2, 2], [-2, 2]], "default_connectivity_radius": 0.5,
            "vel_mean": [0, 0, 0], "vel_std": [1, 1, 1], "acc_mean": [0, -0.01, 0], "acc_std": [1, 1, 1],
        }
        cfg = Phase3Config(max_neighbors=8)
        first = graph_from_gns(record, metadata, 5, cfg)
        record["position"][6:] += 100  # only target/future data changes
        second = graph_from_gns(record, metadata, 5, cfg)
        np.testing.assert_allclose(first.node_features, second.node_features)
        np.testing.assert_array_equal(first.edge_index, second.edge_index)
        self.assertEqual(first.node_features.shape[1], 27)
        self.assertEqual(first.edge_features.shape[1], 4)

    def test_model_forward_backward(self):
        model = UnifiedGNS(hidden=16, blocks=2)
        node = torch.randn(8, 27, requires_grad=True)
        types = torch.zeros(8, dtype=torch.long)
        edges = torch.randn(10, 4)
        edge_index = torch.randint(0, 8, (2, 10))
        output = model(node, types, edges, edge_index)
        self.assertEqual(tuple(output.shape), (8, 3))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(node.grad).all())

    def test_disconnected_batch_matches_individual_outputs(self):
        rng = np.random.default_rng(17)
        def make_graph(name):
            node = rng.normal(size=(6, 27)).astype(np.float32)
            position = rng.normal(size=(6, 3)).astype(np.float32)
            edge_index, edge = radius_graph(position, 10.0, 5)
            return UnifiedGraph(node, np.zeros(6, np.int64), edge_index, edge,
                                np.zeros((6, 3), np.float32), np.ones(6, bool),
                                position, np.zeros((6, 3), np.float32), np.zeros(6, np.uint8), name)
        first, second = make_graph("first"), make_graph("second")
        combined = batch_graphs([first, second])
        torch.manual_seed(3)
        model = UnifiedGNS(hidden=16, blocks=2).eval()
        def infer(graph):
            return model(torch.from_numpy(graph.node_features), torch.from_numpy(graph.particle_type),
                         torch.from_numpy(graph.edge_features), torch.from_numpy(graph.edge_index)).detach()
        expected = torch.cat((infer(first), infer(second)))
        torch.testing.assert_close(infer(combined), expected, rtol=1e-5, atol=1e-6)

    def test_physics_losses_detect_violation(self):
        metadata = {
            "bounds": [[0, 1], [0, 1], [0, 1]],
            "default_connectivity_radius": 0.5,
            "acc_mean": [0, 0, 0], "acc_std": [1, 1, 1],
        }
        batch = {
            "target": torch.zeros(2, 3), "mask": torch.ones(2, dtype=torch.bool),
            "position": torch.tensor([[0.4, 0.4, 0.4], [0.6, 0.4, 0.4]]),
            "velocity": torch.zeros(2, 3), "routing": torch.ones(2, dtype=torch.long),
            "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        }
        normal_total, normal = physics_informed_loss(torch.zeros(2, 3), batch, metadata)
        self.assertAlmostEqual(float(normal_total), 0.0, places=7)
        violation_prediction = torch.tensor([[0, 2.0, 0], [0, 2.0, 0]], requires_grad=True)
        violation_total, violation = physics_informed_loss(violation_prediction, batch, metadata)
        self.assertGreater(float(violation["penetration"].detach()), 0.0)
        self.assertGreater(float(violation["energy"].detach()), 0.0)
        violation_total.backward()
        self.assertTrue(torch.isfinite(violation_prediction.grad).all())

    def test_real_waterdrop_sample_if_present(self):
        repository = Path(__file__).resolve().parents[2]
        root = repository / "phase3" / "datasets" / "gns" / "WaterDropSample" / "source"
        if not (root / "train.tfrecord").exists():
            self.skipTest("WaterDropSample is not present")
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        payload = next(iter_tfrecord(root / "train.tfrecord"))
        record = decode_sequence_example(payload, metadata)
        self.assertEqual(record["position"].shape[0], metadata["sequence_length"] + 1)
        self.assertEqual(record["position"].shape[2], metadata["dim"])


if __name__ == "__main__":
    unittest.main()
