"""Disconnected graph batching without a torch-geometric dependency."""

from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import torch

from .dataset import GraphSample


def pack_graphs(samples: list[GraphSample], pin_memory: bool = False) -> dict[str, torch.Tensor]:
    offsets = np.cumsum([0] + [sample.node_features.shape[0] for sample in samples[:-1]])
    edge_index = [sample.edge_index + offset for sample, offset in zip(samples, offsets) if sample.edge_index.shape[1]]
    result = {
        "node_features": torch.from_numpy(np.concatenate([s.node_features for s in samples])),
        "edge_features": torch.from_numpy(np.concatenate([s.edge_features for s in samples])),
        "edge_index": torch.from_numpy(np.concatenate(edge_index, axis=1) if edge_index else np.empty((2, 0), np.int64)).long(),
        "target_delta_v": torch.from_numpy(np.concatenate([s.target_delta_v for s in samples])),
        "positions": torch.from_numpy(np.concatenate([s.positions for s in samples])),
        "splash_mask": torch.from_numpy(np.concatenate([s.splash_mask for s in samples])).bool(),
        "particle_mass": torch.from_numpy(np.concatenate([s.particle_mass for s in samples])),
        "particle_id": torch.from_numpy(np.concatenate([s.particle_id for s in samples])).long(),
        "graph_index": torch.from_numpy(np.concatenate([np.full(s.node_features.shape[0], i, np.int64) for i, s in enumerate(samples)])).long(),
    }
    if pin_memory and torch.cuda.is_available():
        result = {key: value.pin_memory() for key, value in result.items()}
    return result


def dynamic_batches(samples: list[GraphSample], max_nodes: int, max_edges: int, shuffle: bool = True) -> Iterable[list[GraphSample]]:
    order = list(range(len(samples)))
    if shuffle:
        random.shuffle(order)
    batch, nodes, edges = [], 0, 0
    for index in order:
        sample = samples[index]
        n, e = sample.node_features.shape[0], sample.edge_index.shape[1]
        if batch and (nodes + n > max_nodes or edges + e > max_edges):
            yield batch
            batch, nodes, edges = [], 0, 0
        batch.append(sample)
        nodes += n
        edges += e
    if batch:
        yield batch
