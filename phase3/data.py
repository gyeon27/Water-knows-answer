"""Indexed Water-3D access and a common WCSPH/GNS graph representation."""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import random
import struct
import time
from typing import Iterator
from urllib.request import Request, urlopen

import numpy as np
from scipy.spatial import cKDTree

from .config import Phase3Config
from .gns_data_adapter import BASE_URL, _masked_crc32c, decode_sequence_example


KINEMATIC_ID = 3
NUMERIC_NODE_SIZE = 27
EDGE_SIZE = 4


@dataclass
class UnifiedGraph:
    node_features: np.ndarray
    particle_type: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    target: np.ndarray
    target_mask: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    routing_state: np.ndarray
    graph_id: str


def batch_graphs(graphs: list[UnifiedGraph]) -> UnifiedGraph:
    """Combine graphs without cross-trajectory edges using node offsets."""
    if not graphs:
        raise ValueError("at least one graph is required")
    offsets = np.cumsum([0, *[graph.node_features.shape[0] for graph in graphs[:-1]]], dtype=np.int64)
    edge_indices = [graph.edge_index + offset for graph, offset in zip(graphs, offsets)]
    return UnifiedGraph(
        node_features=np.concatenate([graph.node_features for graph in graphs]),
        particle_type=np.concatenate([graph.particle_type for graph in graphs]),
        edge_index=np.concatenate(edge_indices, axis=1),
        edge_features=np.concatenate([graph.edge_features for graph in graphs]),
        target=np.concatenate([graph.target for graph in graphs]),
        target_mask=np.concatenate([graph.target_mask for graph in graphs]),
        positions=np.concatenate([graph.positions for graph in graphs]),
        velocities=np.concatenate([graph.velocities for graph in graphs]),
        routing_state=np.concatenate([graph.routing_state for graph in graphs]),
        graph_id="+".join(graph.graph_id for graph in graphs),
    )


def download_resumable(url: str, destination: Path) -> dict[str, object]:
    """Download a large file with HTTP Range resume and final SHA-256."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    digest_cache = destination.with_suffix(destination.suffix + ".sha256.json")
    if destination.is_file() and destination.stat().st_size > 0:
        if digest_cache.is_file():
            cached = json.loads(digest_cache.read_text(encoding="utf-8"))
            if int(cached.get("bytes", -1)) == destination.stat().st_size:
                if partial.exists():
                    partial.unlink()
                return {**cached, "status": "kept"}
        # The download may have completed and been atomically renamed just
        # before interruption, before its SHA sidecar was written. Hash the
        # completed destination instead of starting again from byte zero.
        digest = hashlib.sha256()
        with destination.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        result = {"file": destination.name, "bytes": destination.stat().st_size, "sha256": digest.hexdigest()}
        digest_cache.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if partial.exists():
            partial.unlink()
        return {**result, "status": "recovered-complete"}
    start = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "Water-knows-answer/Phase3"}
    if start:
        headers["Range"] = f"bytes={start}-"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=120) as response:
        # A server that ignores Range returns 200; restart instead of appending.
        append = start > 0 and getattr(response, "status", 200) == 206
        mode = "ab" if append else "wb"
        if not append:
            start = 0
        remaining = int(response.headers.get("Content-Length", 0))
        total = start + remaining
        downloaded = start
        next_report = downloaded + 1024**3
        report_started = time.monotonic()
        print(f"download {destination.name}: {downloaded / 2**30:.2f}/{total / 2**30:.2f} GiB", flush=True)
        with partial.open(mode) as stream:
            while True:
                block = response.read(4 * 1024 * 1024)
                if not block:
                    break
                stream.write(block)
                downloaded += len(block)
                if downloaded >= next_report:
                    elapsed = max(time.monotonic() - report_started, 1e-6)
                    rate = (downloaded - start) / elapsed / 2**20
                    print(f"download {destination.name}: {downloaded / 2**30:.2f}/{total / 2**30:.2f} GiB ({rate:.1f} MiB/s)", flush=True)
                    next_report = downloaded + 1024**3
    partial.replace(destination)
    digest = hashlib.sha256()
    with destination.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    result = {"file": destination.name, "bytes": destination.stat().st_size, "sha256": digest.hexdigest()}
    digest_cache.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return {**result, "status": "downloaded"}


def build_tfrecord_index(path: Path, output: Path, verify_crc: bool = True) -> np.ndarray:
    """Write [payload_offset, payload_length] for every TFRecord entry."""
    verification = output.with_suffix(output.suffix + ".verified.json")
    if output.is_file() and verification.is_file():
        cached = json.loads(verification.read_text(encoding="utf-8"))
        stat = path.stat()
        if int(cached.get("record_bytes", -1)) == stat.st_size and bool(cached.get("payload_crc", False)) >= verify_crc:
            return np.load(output, mmap_mode="r")
    entries = []
    with path.open("rb") as stream:
        while True:
            length_offset = stream.tell()
            length_bytes = stream.read(8)
            if not length_bytes:
                break
            if len(length_bytes) != 8:
                raise ValueError(f"truncated length at {length_offset}")
            length_crc = stream.read(4)
            length = struct.unpack("<Q", length_bytes)[0]
            payload_offset = stream.tell()
            payload = stream.read(length)
            payload_crc = stream.read(4)
            if len(length_crc) != 4 or len(payload) != length or len(payload_crc) != 4:
                raise ValueError(f"truncated record at {length_offset}")
            if verify_crc:
                if struct.unpack("<I", length_crc)[0] != _masked_crc32c(length_bytes):
                    raise ValueError(f"length CRC mismatch at record {len(entries)}")
                if struct.unpack("<I", payload_crc)[0] != _masked_crc32c(payload):
                    raise ValueError(f"payload CRC mismatch at record {len(entries)}")
            entries.append((payload_offset, length))
    index = np.asarray(entries, dtype=np.int64)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, index)
    verification.write_text(json.dumps({
        "record": path.name,
        "record_bytes": path.stat().st_size,
        "records": len(index),
        "length_crc": bool(verify_crc),
        "payload_crc": bool(verify_crc),
    }, indent=2), encoding="utf-8")
    return index


class IndexedTFRecord:
    def __init__(self, record: Path, index: Path, metadata: Path):
        self.record = record
        self.index_path = index
        self.metadata = json.loads(metadata.read_text(encoding="utf-8"))
        self.index = np.load(index, mmap_mode="r")

    def __len__(self) -> int:
        return int(self.index.shape[0])

    def read(self, item: int) -> dict:
        offset, length = map(int, self.index[item])
        with self.record.open("rb") as stream:
            stream.seek(offset)
            payload = stream.read(length)
            crc_bytes = stream.read(4)
        if len(payload) != length or len(crc_bytes) != 4:
            raise ValueError(f"truncated indexed record {item}")
        if struct.unpack("<I", crc_bytes)[0] != _masked_crc32c(payload):
            raise ValueError(f"payload CRC mismatch at record {item}")
        return decode_sequence_example(payload, self.metadata)


class UnifiedParticleGraphDataset:
    """Trajectory-aware Water-3D graph loader with a bounded LRU cache."""

    def __init__(self, dataset: IndexedTFRecord, cfg: Phase3Config, objective: str, cache_trajectories: int = 1):
        self.dataset = dataset
        self.cfg = cfg
        self.objective = objective
        self.cache_trajectories = max(1, int(cache_trajectories))
        self._cache: OrderedDict[int, dict] = OrderedDict()

    def __len__(self) -> int:
        return len(self.dataset)

    def trajectory(self, index: int) -> dict:
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        value = self.dataset.read(index)
        self._cache[index] = value
        while len(self._cache) > self.cache_trajectories:
            self._cache.popitem(last=False)
        return value

    def graph(self, trajectory: int, frame: int) -> UnifiedGraph:
        return graph_from_gns(self.trajectory(trajectory), self.dataset.metadata, frame, self.cfg, self.objective)


def deterministic_windows(frames: int, count: int, seed: int, history: int = 6) -> list[int]:
    valid = list(range(history - 1, frames - 1))
    if not valid:
        return []
    rng = random.Random(seed)
    if count >= len(valid):
        return valid
    return sorted(rng.sample(valid, count))


def radius_graph(position: np.ndarray, radius: float, max_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    """Directed nearest-neighbor radius graph using SciPy's C implementation."""
    position = np.ascontiguousarray(position, dtype=np.float32)
    count = position.shape[0]
    if count < 2:
        return np.empty((2, 0), np.int64), np.empty((0, EDGE_SIZE), np.float32)
    k = min(max_neighbors + 1, count)  # include the zero-distance self result
    # Spawning a full worker pool costs more than the query for the small ROI
    # graphs used by selective PI-GNN. Parallelize only sufficiently large
    # graphs; this preserves throughput for full GNS while removing ~milliseconds
    # of per-frame scheduling overhead for a few hundred SPLASH particles.
    workers = 1 if count < 2_000 else -1
    distances, indices = cKDTree(position).query(
        position, k=k, distance_upper_bound=radius, workers=workers
    )
    if k == 1:
        distances = distances[:, None]; indices = indices[:, None]
    receivers_grid = np.broadcast_to(np.arange(count, dtype=np.int64)[:, None], indices.shape)
    valid = np.isfinite(distances) & (indices < count) & (indices != receivers_grid) & (distances > 1e-9)
    senders = indices[valid].astype(np.int64, copy=False)
    receivers = receivers_grid[valid]
    selected_distances = distances[valid].astype(np.float32, copy=False)
    relative = (position[senders] - position[receivers]) / float(radius)
    features = np.concatenate((relative, (selected_distances / float(radius))[:, None]), axis=1).astype(np.float32, copy=False)
    return np.stack((senders, receivers)), features


def _routing_state(velocity: np.ndarray, acceleration: np.ndarray, position: np.ndarray, bounds: np.ndarray, particle_type: np.ndarray, vel_std: np.ndarray, acc_std: np.ndarray) -> np.ndarray:
    speed_n = np.linalg.norm(velocity / np.maximum(vel_std, 1e-8), axis=1)
    accel_n = np.linalg.norm(acceleration / np.maximum(acc_std, 1e-8), axis=1)
    lower = position - bounds[:, 0]
    upper = bounds[:, 1] - position
    clearance = np.min(np.concatenate((lower, upper), axis=1), axis=1)
    radius_scale = float(np.mean(bounds[:, 1] - bounds[:, 0]))
    near_boundary = clearance < 0.04 * radius_scale
    splash = (speed_n > 1.0) & (accel_n > 1.0) & near_boundary
    pool = (~splash) & (speed_n < 0.25) & (position[:, 1] < bounds[1, 0] + 0.15 * (bounds[1, 1] - bounds[1, 0]))
    state = np.zeros(position.shape[0], np.uint8)
    state[splash] = 1
    state[pool] = 2
    state[particle_type == KINEMATIC_ID] = 3
    return state


def graph_from_gns(record: dict, metadata: dict, frame: int, cfg: Phase3Config, objective: str = "ours") -> UnifiedGraph:
    position = np.asarray(record["position"], np.float32)
    types = np.asarray(record["particle_type"], np.int64)
    bounds = np.asarray(metadata["bounds"], np.float32)
    radius = float(metadata["default_connectivity_radius"])
    vel_mean = np.asarray(metadata["vel_mean"], np.float32)
    vel_std = np.asarray(metadata["vel_std"], np.float32)
    acc_mean = np.asarray(metadata["acc_mean"], np.float32)
    acc_std = np.asarray(metadata["acc_std"], np.float32)
    velocities = np.diff(position[frame - 5 : frame + 2], axis=0)
    history = velocities[:5]
    current_v, next_v = velocities[4], velocities[5]
    acceleration = next_v - current_v
    previous_acc = velocities[4] - velocities[3]
    current = position[frame]
    state = _routing_state(current_v, previous_acc, current, bounds, types, vel_std, acc_std)
    velocity_features = ((history - vel_mean) / np.maximum(vel_std, 1e-8)).transpose(1, 0, 2).reshape(current.shape[0], 15)
    distances = np.concatenate((current - bounds[:, 0], bounds[:, 1] - current), axis=1)
    distances = np.clip(distances / radius, -1.0, 1.0)
    state_onehot = np.zeros((current.shape[0], 3), np.float32)
    fluid = types != KINEMATIC_ID
    for value in range(3):
        state_onehot[:, value] = state == value
    gravity = np.tile((0.0, -1.0, 0.0), (current.shape[0], 1)).astype(np.float32)
    node = np.concatenate((velocity_features, distances, state_onehot, gravity), axis=1).astype(np.float32)
    edges, edge_features = radius_graph(current, radius, cfg.max_neighbors)
    normalized_acc = (acceleration - acc_mean) / np.maximum(acc_std, 1e-8)
    if objective == "gns":
        target = normalized_acc
        mask = fluid
    else:
        # The analytic baseline is the dataset mean acceleration (mostly
        # gravity); the residual is normalized by the official acceleration std.
        target = (acceleration - acc_mean) / np.maximum(acc_std, 1e-8)
        if objective == "ours":
            mask = fluid & (state == 1)
        elif objective == "reversed":
            mask = fluid & (state != 1)
        elif objective == "all_residual":
            mask = fluid
        else:
            raise ValueError(f"unknown objective {objective}")
    if not np.any(mask):
        mask = fluid
    return UnifiedGraph(node, types, edges, edge_features, target.astype(np.float32), mask, current, current_v, state, str(record.get("key", 0)))


def graph_from_wcsph(path: Path, terrain_root: Path, frame: int, cfg: Phase3Config) -> UnifiedGraph:
    """Map a Phase-2 WCSPH frame to the same 27/4 public-data feature space."""
    from phase2.shallow_water import TerrainData

    with np.load(path, allow_pickle=False) as data:
        positions_all = data["positions"].astype(np.float32)
        velocities_all = data["velocities"].astype(np.float32)
        active_all = data["active_mask"].astype(bool)
        splash_all = data["splash_roi"].astype(bool)
        terrain_id = str(data["terrain_id"].item())
        dt = float(data["dt"])
    ids = np.flatnonzero(active_all[frame])
    if not ids.size:
        raise ValueError(f"no active WCSPH particles at frame {frame}")
    terrain = TerrainData.load(terrain_root / terrain_id)
    current = positions_all[frame, ids]
    current_v = velocities_all[frame, ids] * dt  # position change per model step
    history = velocities_all[frame - 4 : frame + 1, ids] * dt
    all_active_v = velocities_all[active_all] * dt
    vel_mean = all_active_v.mean(0)
    vel_std = np.maximum(all_active_v.std(0), 1e-5)
    acceleration_values = np.diff(velocities_all * dt, axis=0)
    acc_mask = active_all[1:] & active_all[:-1]
    active_acc = acceleration_values[acc_mask]
    acc_mean = active_acc.mean(0) if active_acc.size else np.array((0, -9.81 * dt * dt, 0), np.float32)
    acc_std = np.maximum(active_acc.std(0), 1e-5) if active_acc.size else np.ones(3, np.float32)
    velocity_features = ((history - vel_mean) / vel_std).transpose(1, 0, 2).reshape(ids.size, 15)
    bounds = np.asarray(((-terrain.width_m / 2, terrain.width_m / 2), (float(terrain.height.min()), float(terrain.height.max() + 8)), (0, terrain.length_m)), np.float32)
    radius = 0.32
    distances = np.clip(np.concatenate((current - bounds[:, 0], bounds[:, 1] - current), axis=1) / radius, -1, 1)
    splash = splash_all[frame, ids]
    speed = np.linalg.norm(velocities_all[frame, ids], axis=1)
    pool = (~splash) & (speed < 0.35) & (current[:, 1] < terrain.height.max() + 0.5)
    state = np.zeros(ids.size, np.uint8); state[splash] = 1; state[pool] = 2
    state_onehot = np.eye(3, dtype=np.float32)[state]
    gravity = np.tile((0, -1, 0), (ids.size, 1)).astype(np.float32)
    node = np.concatenate((velocity_features, distances, state_onehot, gravity), axis=1).astype(np.float32)

    # Sample the height field at approximately one boundary node per radius.
    row_stride = max(1, int(round(radius / terrain.dz)))
    col_stride = max(1, int(round(radius / terrain.dx)))
    rr = np.arange(0, terrain.height.shape[0], row_stride)
    cc = np.arange(0, terrain.height.shape[1], col_stride)
    zz, xx = np.meshgrid(rr * terrain.dz, cc * terrain.dx - terrain.width_m / 2, indexing="ij")
    boundary = np.column_stack((xx.ravel(), terrain.height[np.ix_(rr, cc)].ravel(), zz.ravel())).astype(np.float32)
    boundary_count = boundary.shape[0]
    boundary_node = np.zeros((boundary_count, NUMERIC_NODE_SIZE), np.float32)
    boundary_distance = np.clip(np.concatenate((boundary - bounds[:, 0], bounds[:, 1] - boundary), axis=1) / radius, -1, 1)
    boundary_node[:, 15:21] = boundary_distance
    boundary_node[:, 24:27] = (0, -1, 0)
    combined_position = np.concatenate((current, boundary))
    combined_node = np.concatenate((node, boundary_node))
    types = np.concatenate((np.zeros(ids.size, np.int64), np.full(boundary_count, KINEMATIC_ID, np.int64)))
    edges, edge_features = radius_graph(combined_position, radius, cfg.max_neighbors)
    next_v = velocities_all[frame + 1, ids] * dt
    target_fluid = ((next_v - current_v) - acc_mean) / acc_std
    target = np.concatenate((target_fluid, np.zeros((boundary_count, 3), np.float32)))
    mask = np.concatenate((splash, np.zeros(boundary_count, bool)))
    if not np.any(mask):
        mask[:ids.size] = True
    combined_state = np.concatenate((state, np.full(boundary_count, 3, np.uint8)))
    combined_velocity = np.concatenate((current_v, np.zeros((boundary_count, 3), np.float32)))
    return UnifiedGraph(combined_node, types, edges, edge_features, target.astype(np.float32), mask, combined_position, combined_velocity, combined_state, f"{path.stem}:{frame}")


def prepare_water3d(root: Path, cfg: Phase3Config, splits: tuple[str, ...] | None = None, write_manifest: bool = True) -> dict[str, object]:
    raw = root / "raw" / cfg.dataset
    indices = root / "indices" / cfg.dataset
    raw.mkdir(parents=True, exist_ok=True)
    metadata_path = raw / "metadata.json"
    selected = splits or ("train", "valid", "test")
    manifest = {"dataset": cfg.dataset, "splits": list(selected), "files": []}
    manifest["files"].append(download_resumable(f"{BASE_URL}/{cfg.dataset}/metadata.json", metadata_path))
    expected = {"train": cfg.train_trajectories, "valid": cfg.valid_trajectories, "test": cfg.test_trajectories}
    for split in selected:
        count = expected[split]
        record = raw / f"{split}.tfrecord"
        manifest["files"].append(download_resumable(f"{BASE_URL}/{cfg.dataset}/{split}.tfrecord", record))
        index_path = indices / f"{split}.npy"
        index = build_tfrecord_index(record, index_path)
        if len(index) != count:
            raise ValueError(f"{split}: expected {count} trajectories, found {len(index)}")
    if write_manifest:
        (root / "indices" / "water3d_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
