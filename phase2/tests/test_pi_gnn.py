from __future__ import annotations

import unittest

import numpy as np
import torch

from phase2.gnn import GraphSample, ResidualGNS, pack_graphs, physics_losses


def sample(clearance=1.0):
    node = np.zeros((2, 33), np.float32)
    node[:, 18] = clearance
    node[:, 20] = 1.0
    node[:, 31] = -1.0
    edge_index = np.array([[0, 1], [1, 0]], np.int64)
    edge = np.zeros((2, 8), np.float32)
    position = np.array([[0, 1, 0], [.1, 1, 0]], np.float32)
    return GraphSample(node, edge_index, edge, np.zeros((2, 3), np.float32), position, np.ones(2, bool), np.arange(2, dtype=np.int32), np.ones(2, np.float32) * .1)


class PIGNNTests(unittest.TestCase):
    def test_physics_losses_zero_for_teacher_prediction(self):
        batch = pack_graphs([sample()])
        target = torch.zeros((2, 3))
        losses = physics_losses(target, target, batch)
        for value in losses.values():
            self.assertLess(float(value), 1e-7)

    def test_physics_violations_increase_losses(self):
        batch = pack_graphs([sample(clearance=.001)])
        prediction = torch.tensor([[0, -20, 0], [8, 12, 0]], dtype=torch.float32)
        losses = physics_losses(prediction, torch.zeros_like(prediction), batch)
        self.assertGreater(float(losses["penetration"]), 0)
        self.assertGreater(float(losses["momentum"]), 0)
        self.assertGreater(float(losses["density"]), 0)
        self.assertGreater(float(losses["energy"]), 0)

    def test_disconnected_batch_matches_individual_graphs(self):
        torch.manual_seed(3)
        model = ResidualGNS(33, 8, hidden_size=16, blocks=2).eval()
        one = sample()
        single = pack_graphs([one])
        combined = pack_graphs([one, one])
        with torch.no_grad():
            expected = model(single["node_features"], single["edge_features"], single["edge_index"])
            actual = model(combined["node_features"], combined["edge_features"], combined["edge_index"])
        self.assertTrue(torch.allclose(actual[:2], expected, atol=1e-6))
        self.assertTrue(torch.allclose(actual[2:], expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
