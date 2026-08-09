"""Runtime graph construction using predicted state only."""

from __future__ import annotations

import numpy as np

from .dataset import GraphSample, TrajectoryGraphDataset


def terrain_sample(terrain, position):
    col = np.clip(np.rint((position[:, 0] + terrain.width_m * 0.5) / terrain.dx).astype(int), 0, terrain.height.shape[1] - 1)
    row = np.clip(np.rint(position[:, 2] / terrain.dz).astype(int), 0, terrain.height.shape[0] - 1)
    bed = terrain.height[row, col]
    dz, dx = np.gradient(terrain.height, terrain.dz, terrain.dx)
    normal = np.column_stack((-dx[row, col], np.ones(row.size), -dz[row, col]))
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    return bed, normal, np.hypot(dx[row, col], dz[row, col]), terrain.cliff[row, col]


def runtime_graph(position, velocity_history, active, terrain, particle_ids, particle_mass, flow, radius=0.32):
    ids = np.flatnonzero(active)
    if not ids.size:
        return None, ids
    p, v = position[ids], velocity_history[-1, ids]
    helper = object.__new__(TrajectoryGraphDataset)
    helper.radius, helper.max_neighbors = radius, 48
    full_edges, _, _ = helper._radius_graph(p)
    bed, normal, slope, cliff = terrain_sample(terrain, p)
    clearance = p[:, 1] - bed
    speed = np.linalg.norm(v, axis=1)
    approach = np.sum(v * normal, axis=1)
    roi = ((clearance < 0.65) & ((approach < -0.15) | (speed > 1.0))) | (cliff & (clearance < 1.2))
    context = roi.copy()
    if full_edges.shape[1]:
        touch = roi[full_edges[0]] | roi[full_edges[1]]
        context[full_edges[:, touch].reshape(-1)] = True
    local = np.flatnonzero(context)
    selected_ids = ids[local]
    p, v = position[selected_ids], velocity_history[-1, selected_ids]
    edge_index, relative, distance = helper._radius_graph(p)
    bed, normal, slope, cliff = terrain_sample(terrain, p)
    clearance = p[:, 1] - bed
    collision = (clearance < 0.03).astype(np.float32)
    local_roi = roi[local]
    pool = (~local_roi) & (clearance < 0.1) & (np.linalg.norm(v, axis=1) < 0.35) & (slope < 0.18)
    state = np.column_stack((~(local_roi | pool), local_roi, pool)).astype(np.float32)
    history = []
    for frame in velocity_history:
        history.append(frame[selected_ids])
    neighbor_count = np.bincount(edge_index[1], minlength=p.shape[0]) if edge_index.shape[1] else np.zeros(p.shape[0])
    thickness = np.clip(neighbor_count * (particle_mass / 1000.0) / (np.pi * radius**2), 0, 1)
    node = np.column_stack((np.concatenate(history, axis=1), clearance, normal, slope, cliff, collision, thickness, np.full(p.shape[0], flow / terrain.width_m), state, np.tile((0, -1, 0), (p.shape[0], 1)))).astype(np.float32)
    if edge_index.shape[1]:
        sender, receiver = edge_index
        edge = np.column_stack((relative / radius, distance / radius, v[sender] - v[receiver], bed[sender] - bed[receiver])).astype(np.float32)
    else:
        edge = np.empty((0, 8), np.float32)
    sample = GraphSample(node, edge_index, edge, np.zeros((p.shape[0], 3), np.float32), p.astype(np.float32), local_roi, particle_ids[selected_ids].astype(np.int32), np.full(p.shape[0], particle_mass, np.float32))
    return sample, selected_ids
