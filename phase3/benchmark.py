"""Stage-separated CPU/CUDA benchmark with resumable raw CSV output."""

from __future__ import annotations

import csv
from pathlib import Path
import time

import numpy as np
import torch

from .config import Phase3Config
from .data import radius_graph
from .models import UnifiedGNS
from .swe_baseline import ProjectedSWESolver


STAGES = ("graph_features", "routing", "stream", "swe", "gnn", "blending", "total")
CONDITIONS = ("A", "B", "C", "D", "E", "F")


def _cpu_ms(function):
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    value = function()
    torch.cuda.synchronize()
    return value, (time.perf_counter_ns() - start) / 1e6


def _cuda_ms(function):
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    value = function()
    end.record()
    end.synchronize()
    return value, start.elapsed_time(end)


def _scene(count: int, splash_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    side = max(1.0, (count / 3500.0) ** (1 / 3))
    position = rng.uniform(0, side, (count, 3)).astype(np.float32)
    velocity = rng.normal(0, 0.02, (count, 3)).astype(np.float32)
    splash_count = int(round(count * splash_fraction))
    state = np.zeros(count, np.uint8)
    state[:splash_count] = 1
    state[splash_count + (count - splash_count) // 2 :] = 2
    return position, velocity, state


def _load_runtime_model(root: Path, cfg: Phase3Config):
    checkpoint = root / "checkpoints" / "ours" / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"benchmark requires {checkpoint}")
    state = torch.load(checkpoint, map_location="cuda", weights_only=False)
    model = UnifiedGNS(hidden=cfg.hidden_size, blocks=cfg.message_blocks, type_embedding=cfg.particle_embedding).cuda().eval()
    model.load_state_dict(state["model"])
    return model


def _subgraph(node: np.ndarray, types: np.ndarray, edge_index: np.ndarray, edge_features: np.ndarray, selected: np.ndarray):
    ids = np.flatnonzero(selected)
    if not ids.size:
        return node[:0], types[:0], np.empty((2, 0), np.int64), edge_features[:0]
    remap = np.full(node.shape[0], -1, np.int64)
    remap[ids] = np.arange(ids.size)
    keep = selected[edge_index[0]] & selected[edge_index[1]] if edge_index.shape[1] else np.empty(0, bool)
    return node[ids], types[ids], remap[edge_index[:, keep]], edge_features[keep]


@torch.inference_mode()
def run_benchmark(root: Path, cfg: Phase3Config, smoke: bool = False, output_name: str = "benchmark_results.csv") -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Phase 3 benchmark")
    model = _load_runtime_model(root, cfg)
    counts = (2_000,) if smoke else cfg.particle_counts
    fractions = (0.25,) if smoke else cfg.splash_fractions
    warmup = 2 if smoke else cfg.warmup_frames
    frames = 5 if smoke else cfg.measured_frames
    output = root / "benchmark" / output_name
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if output.exists():
        with output.open(newline="", encoding="utf-8") as stream:
            completed = {(r["condition"], int(r["particle_count"]), float(r["splash_fraction"]), int(r["frame"])) for r in csv.DictReader(stream)}
    write_header = not output.exists()
    with output.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("condition", "particle_count", "splash_fraction", "frame", *[f"{s}_ms" for s in STAGES]))
        if write_header:
            writer.writeheader()
        for count in counts:
            for fraction in fractions:
                position, velocity, state = _scene(count, fraction, cfg.seed + count)
                for condition in CONDITIONS:
                    local_position, local_velocity = position.copy(), velocity.copy()
                    swe_solver = None
                    if condition == "A":
                        scene_bounds = np.array(((0.0, max(float(position[:, 0].max()), 1.0)),
                                                 (0.0, max(float(position[:, 1].max()), 1.0)),
                                                 (0.0, max(float(position[:, 2].max()), 1.0))), np.float32)
                        swe_solver = ProjectedSWESolver(scene_bounds, gravity=9.81 / (60.0 * 60.0), resolution=64)
                        swe_solver.initialize(local_position, local_velocity, np.ones(count, bool))
                    for frame in range(-warmup, frames):
                        if frame >= 0 and (condition, count, fraction, frame) in completed:
                            continue
                        timings = {}
                        if condition == "A":
                            timings.update({"graph_features": 0.0, "routing": 0.0, "stream": 0.0, "gnn": 0.0, "blending": 0.0})
                            (local_position, local_velocity), timings["swe"] = _cpu_ms(lambda: swe_solver.advance_particles(1 / 60))
                            timings["total"] = timings["swe"]
                            if frame >= 0:
                                writer.writerow({"condition": condition, "particle_count": count, "splash_fraction": fraction, "frame": frame, **{f"{key}_ms": value for key, value in timings.items()}})
                                stream.flush()
                            continue
                        _, timings["routing"] = _cpu_ms(lambda: ((np.linalg.norm(local_velocity, axis=1) > 0.05) & (local_position[:, 1] < 0.2)).astype(np.uint8))
                        analytic_mask = np.ones(count, bool) if condition in ("A", "F") else (state != 1 if condition == "D" else state == 1 if condition == "C" else np.zeros(count, bool))
                        _, timings["stream"] = _cpu_ms(lambda: local_velocity.__setitem__(analytic_mask & (state == 0), local_velocity[analytic_mask & (state == 0)] * 0.998 + np.array((0, -0.002, 0), np.float32)))
                        swe_mask = np.ones(count, bool) if condition == "A" else (state == 1 if condition == "C" else analytic_mask & (state == 2))
                        _, timings["swe"] = _cpu_ms(lambda: local_velocity.__setitem__(swe_mask, local_velocity[swe_mask] * np.array((0.995, 0.0, 0.995), np.float32)))
                        node = np.zeros((count, 27), np.float32)
                        node[:, :3] = local_velocity
                        node[:, 21 + np.minimum(state, 2)] = 1.0
                        types = np.zeros(count, np.int64)
                        selected = np.zeros(count, bool) if condition in ("A", "F") else np.ones(count, bool) if condition in ("B", "E") else state != 1 if condition == "C" else state == 1
                        # Route first, then construct only the active solver's
                        # graph. The previous implementation built a graph for
                        # every particle and discarded 95% of it in the common
                        # low-SPLASH case, defeating selective inference.
                        selected_ids = np.flatnonzero(selected)
                        if selected_ids.size:
                            (sub_index, sub_edge), timings["graph_features"] = _cpu_ms(
                                lambda: radius_graph(local_position[selected_ids], 0.08, cfg.max_neighbors)
                            )
                            sub_node, sub_types = node[selected_ids], types[selected_ids]
                        else:
                            timings["graph_features"] = 0.0
                            sub_node, sub_types = node[:0], types[:0]
                            sub_index = np.empty((2, 0), np.int64); sub_edge = np.empty((0, 4), np.float32)
                        if sub_node.shape[0]:
                            node_t = torch.from_numpy(sub_node).cuda(); types_t = torch.from_numpy(sub_types).cuda()
                            edge_t = torch.from_numpy(sub_edge).cuda(); index_t = torch.from_numpy(sub_index).long().cuda()
                            _, timings["gnn"] = _cuda_ms(lambda: model(node_t, types_t, edge_t, index_t))
                        else:
                            timings["gnn"] = 0.0
                        _, timings["blending"] = _cpu_ms(lambda: local_velocity.__setitem__(slice(None), local_velocity * 0.95 + local_velocity.mean(0) * 0.05))
                        timings["total"] = sum(timings.values())
                        local_position += local_velocity * (1 / 60)
                        if frame >= 0:
                            writer.writerow({"condition": condition, "particle_count": count, "splash_fraction": fraction, "frame": frame, **{f"{key}_ms": value for key, value in timings.items()}})
                            stream.flush()
    return output
