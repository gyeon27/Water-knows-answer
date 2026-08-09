"""Convert fixed-shape trajectories into radius graphs for residual learning."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np

try:
    from phase2.shallow_water import TerrainData
except ModuleNotFoundError:
    from shallow_water import TerrainData


@dataclass(frozen=True)
class GraphSample:
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    target_delta_v: np.ndarray
    positions: np.ndarray
    splash_mask: np.ndarray
    particle_id: np.ndarray
    particle_mass: np.ndarray


class TrajectoryGraphDataset:
    """Lazy trajectory loader. Graphs are rebuilt for every requested frame."""

    def __init__(
        self,
        files: list[Path | str],
        terrain_root: Path | str,
        history: int = 6,
        radius_m: float = 0.32,
        max_neighbors: int = 48,
        cache_trajectories: bool = False,
    ):
        self.files = [Path(path) for path in files]
        self.terrain_root = Path(terrain_root)
        self.history = int(history)
        self.radius = float(radius_m)
        self.max_neighbors = int(max_neighbors)
        self.index: list[tuple[int, int]] = []
        self._terrain: dict[str, TerrainData] = {}
        self._cache: list[dict[str, np.ndarray] | None] = []
        for file_index, path in enumerate(self.files):
            with np.load(path, allow_pickle=False) as data:
                frames = data["positions"].shape[0]
                active = data["active_mask"]
                roi = data["splash_roi"]
                for frame in range(self.history - 1, frames - 1):
                    if np.any(active[frame] & (roi[frame] | roi[frame + 1])):
                        self.index.append((file_index, frame))
                self._cache.append({key: data[key] for key in data.files} if cache_trajectories else None)

    def _open(self, file_index: int):
        cached = self._cache[file_index]
        return nullcontext(cached) if cached is not None else np.load(self.files[file_index], allow_pickle=False)

    def __len__(self) -> int:
        return len(self.index)

    def _load_terrain(self, terrain_id: str) -> TerrainData:
        if terrain_id not in self._terrain:
            self._terrain[terrain_id] = TerrainData.load(self.terrain_root / terrain_id)
        return self._terrain[terrain_id]

    @staticmethod
    def _terrain_features(terrain: TerrainData, position: np.ndarray) -> tuple[np.ndarray, ...]:
        col = np.clip(np.rint((position[:, 0] + terrain.width_m * 0.5) / terrain.dx).astype(int), 0, terrain.height.shape[1] - 1)
        row = np.clip(np.rint(position[:, 2] / terrain.dz).astype(int), 0, terrain.height.shape[0] - 1)
        bed = terrain.height[row, col]
        dz, dx = np.gradient(terrain.height, terrain.dz, terrain.dx)
        normal = np.column_stack((-dx[row, col], np.ones(row.size), -dz[row, col]))
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
        slope = np.hypot(dx[row, col], dz[row, col])
        cliff = terrain.cliff[row, col].astype(np.float64)
        return bed, normal, slope, cliff

    def _radius_graph(self, position: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if position.shape[0] == 0:
            return np.empty((2, 0), dtype=np.int64), np.empty((0, 8), dtype=np.float32), np.empty(0)
        cell = self.radius
        bins: dict[tuple[int, int, int], list[int]] = {}
        coordinates = np.floor(position / cell).astype(np.int64)
        for index, key in enumerate(map(tuple, coordinates)):
            bins.setdefault(key, []).append(index)
        senders: list[int] = []
        receivers: list[int] = []
        distances: list[float] = []
        relative: list[np.ndarray] = []
        for receiver, base in enumerate(coordinates):
            candidates: list[tuple[float, int, np.ndarray]] = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for sender in bins.get(tuple(base + (dx, dy, dz)), ()):
                            if sender == receiver:
                                continue
                            delta = position[sender] - position[receiver]
                            distance = float(np.linalg.norm(delta))
                            if 1e-8 < distance <= self.radius:
                                candidates.append((distance, sender, delta))
            candidates.sort(key=lambda item: item[0])
            for distance, sender, delta in candidates[: self.max_neighbors]:
                senders.append(sender)
                receivers.append(receiver)
                distances.append(distance)
                relative.append(delta)
        edge_index = np.asarray((senders, receivers), dtype=np.int64)
        return edge_index, np.asarray(relative, dtype=np.float64), np.asarray(distances, dtype=np.float64)

    def __getitem__(self, item: int) -> GraphSample:
        file_index, frame = self.index[item]
        with self._open(file_index) as data:
            active = data["active_mask"][frame]
            all_position = data["positions"][frame].astype(np.float64)
            all_velocity = data["velocities"].astype(np.float64)
            roi_all = active & (data["splash_roi"][frame] | data["splash_roi"][frame + 1])
            terrain_id = str(data["terrain_id"])
            flow = float(data["flow_rate"][frame, 0])
            dt = float(data["dt"])
            metadata = json.loads(str(data["metadata_json"].item()))
            particle_mass = float(metadata.get("config", {}).get("particle_mass_kg", 2.0))
            particle_ids = data["particle_id"]

            active_ids = np.flatnonzero(active)
            active_position = all_position[active_ids]
            full_edges, _, _ = self._radius_graph(active_position)
            local_roi = roi_all[active_ids]
            context = local_roi.copy()
            if full_edges.shape[1]:
                touches = local_roi[full_edges[0]] | local_roi[full_edges[1]]
                context[full_edges[:, touches].reshape(-1)] = True
            selected_ids = active_ids[context]
            position = all_position[selected_ids]
            velocity = all_velocity[frame, selected_ids]
            edge_index, relative, distance = self._radius_graph(position)

            terrain = self._load_terrain(terrain_id)
            bed, normal, slope, cliff = self._terrain_features(terrain, position)
            clearance = position[:, 1] - bed
            collision = (clearance < 0.03).astype(np.float64)
            speed = np.linalg.norm(velocity, axis=1)
            splash = roi_all[selected_ids]
            pool = (~splash) & (clearance < 0.1) & (speed < 0.35) & (slope < 0.18)
            stream = ~(splash | pool)
            state = np.column_stack((stream, splash, pool)).astype(np.float64)

            history = []
            first = frame - self.history + 1
            for history_frame in range(first, frame + 1):
                present = data["active_mask"][history_frame, selected_ids]
                value = all_velocity[history_frame, selected_ids].copy()
                value[~present] = velocity[~present]
                history.append(value)
            velocity_history = np.concatenate(history, axis=1)
            neighbor_count = np.bincount(edge_index[1], minlength=position.shape[0]) if edge_index.shape[1] else np.zeros(position.shape[0])
            particle_volume = particle_mass / 1000.0
            sheet_thickness = np.clip(neighbor_count * particle_volume / (np.pi * self.radius**2), 0.0, 1.0)
            local_flow = np.full(position.shape[0], flow / max(terrain.width_m, 1e-6))
            gravity = np.tile((0.0, -1.0, 0.0), (position.shape[0], 1))
            node = np.column_stack(
                (velocity_history, clearance, normal, slope, cliff, collision, sheet_thickness, local_flow, state, gravity)
            ).astype(np.float32)

            if edge_index.shape[1]:
                sender, receiver = edge_index
                relative_velocity = velocity[sender] - velocity[receiver]
                bed_delta = bed[sender] - bed[receiver]
                edge = np.column_stack((relative / self.radius, distance / self.radius, relative_velocity, bed_delta)).astype(np.float32)
            else:
                edge = np.empty((0, 8), dtype=np.float32)

            base_next = (velocity + np.array((0.0, -9.81 * dt, 0.0))) * np.exp(-0.08 * dt)
            target = (all_velocity[frame + 1, selected_ids] - base_next).astype(np.float32)
            return GraphSample(
                node,
                edge_index,
                edge,
                target,
                position.astype(np.float32),
                roi_all[selected_ids],
                particle_ids[selected_ids].astype(np.int32),
                np.full(position.shape[0], particle_mass, dtype=np.float32),
            )

    @staticmethod
    def to_torch(sample: GraphSample, device: str = "cpu") -> dict[str, "torch.Tensor"]:
        import torch

        return {
            "node_features": torch.as_tensor(sample.node_features, device=device),
            "edge_index": torch.as_tensor(sample.edge_index, dtype=torch.long, device=device),
            "edge_features": torch.as_tensor(sample.edge_features, device=device),
            "target_delta_v": torch.as_tensor(sample.target_delta_v, device=device),
            "positions": torch.as_tensor(sample.positions, device=device),
            "splash_mask": torch.as_tensor(sample.splash_mask, dtype=torch.bool, device=device),
            "particle_mass": torch.as_tensor(sample.particle_mass, device=device),
        }
