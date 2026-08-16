"""Six-condition, teacher-aligned Water-3D evaluation and autonomous rollout."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .config import Phase3Config
from .data import IndexedTFRecord, deterministic_windows, graph_from_gns, radius_graph, KINEMATIC_ID
from .models import UnifiedGNS
from .swe_baseline import ProjectedSWESolver


LEARNED = {"B": "gnn_only", "C": "reversed", "D": "ours", "E": "baseline_gns", "G": "ours"}
CONDITION_NAMES = {"A": "SWE-only", "B": "GNN-only", "C": "Reversed", "D": "Ours", "E": "Baseline-GNS", "F": "Simple-3D", "G": "Optimized-Ours"}


def _load_model(root: Path, name: str, cfg: Phase3Config, device: torch.device) -> UnifiedGNS:
    path = root / "checkpoints" / name / "best.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing trained checkpoint {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    model = UnifiedGNS(hidden=cfg.hidden_size, blocks=cfg.message_blocks, type_embedding=cfg.particle_embedding).to(device)
    model.load_state_dict(state["model"])
    return model.eval()


@torch.no_grad()
def _model_acceleration(model: UnifiedGNS, graph, metadata: dict, device: torch.device) -> np.ndarray:
    node = torch.from_numpy(graph.node_features).to(device)
    types = torch.from_numpy(graph.particle_type).long().to(device)
    edge = torch.from_numpy(graph.edge_features).to(device)
    edge_index = torch.from_numpy(graph.edge_index).long().to(device)
    with torch.autocast("cuda", dtype=torch.float16):
        normalized = model(node, types, edge, edge_index)
    return normalized.float().cpu().numpy() * np.asarray(metadata["acc_std"], np.float32) + np.asarray(metadata["acc_mean"], np.float32)


@torch.inference_mode()
def _roi_model_acceleration(model: UnifiedGNS, graph, metadata: dict, device: torch.device) -> np.ndarray:
    """Infer only the routed SPLASH induced graph and scatter to global IDs."""
    mean = np.broadcast_to(np.asarray(metadata["acc_mean"], np.float32), graph.positions.shape).copy()
    selected = (graph.particle_type != KINEMATIC_ID) & (graph.routing_state == 1)
    ids = np.flatnonzero(selected)
    if not ids.size:
        return mean
    edge_index, edge_features = radius_graph(
        graph.positions[ids], float(metadata["default_connectivity_radius"]), 48
    )
    node = torch.from_numpy(graph.node_features[ids]).to(device)
    types = torch.from_numpy(graph.particle_type[ids]).long().to(device)
    edge = torch.from_numpy(edge_features).to(device)
    index = torch.from_numpy(edge_index).long().to(device)
    with torch.autocast("cuda", dtype=torch.float16):
        normalized = model(node, types, edge, index)
    physical = normalized.float().cpu().numpy() * np.asarray(metadata["acc_std"], np.float32) + np.asarray(metadata["acc_mean"], np.float32)
    mean[ids] = physical
    return mean


def _condition_acceleration(condition: str, graph, metadata: dict, model: UnifiedGNS | None, device: torch.device) -> np.ndarray:
    mean = np.broadcast_to(np.asarray(metadata["acc_mean"], np.float32), graph.positions.shape).copy()
    splash = graph.routing_state == 1
    fluid = graph.particle_type != KINEMATIC_ID
    current_v = graph.velocities
    if condition == "A":
        acceleration = np.zeros_like(mean)
        acceleration[:, 1] = -current_v[:, 1]  # collapse vertical motion into a height field
    elif condition in LEARNED:
        learned = _roi_model_acceleration(model, graph, metadata, device) if condition == "G" else _model_acceleration(model, graph, metadata, device)
        if condition in ("B", "E"):
            acceleration = learned
        elif condition == "C":
            acceleration = learned
            acceleration[splash, 1] = -current_v[splash, 1]
            acceleration[splash, 0] = 0.0
            acceleration[splash, 2] = 0.0
        else:  # D/G: analytic mean outside the learned splash region
            acceleration = mean
            acceleration[splash] = learned[splash]
    elif condition == "F":
        acceleration = mean
        bounds = np.asarray(metadata["bounds"], np.float32)
        at_floor = graph.positions[:, 1] < bounds[1, 0] + float(metadata["default_connectivity_radius"])
        impact = fluid & splash & at_floor & (current_v[:, 1] < 0)
        acceleration[impact, 1] += -1.25 * current_v[impact, 1]
        acceleration[impact, 0] += 0.08 * np.sign(graph.positions[impact, 0] - np.mean(bounds[0]))
        acceleration[impact, 2] += 0.08 * np.sign(graph.positions[impact, 2] - np.mean(bounds[2]))
    else:
        raise ValueError(condition)
    acceleration[~fluid] = 0.0
    return acceleration


def _density_proxy(position: np.ndarray, radius: float) -> np.ndarray:
    edges, features = radius_graph(position, radius, 48)
    density = np.ones(position.shape[0], np.float32)
    if edges.shape[1]:
        weight = np.maximum(0.0, 1.0 - features[:, 3]) ** 3
        np.add.at(density, edges[1], weight)
    return density


def _metrics(predicted_p: np.ndarray, predicted_v: np.ndarray, teacher_p: np.ndarray, teacher_v: np.ndarray, fluid: np.ndarray, metadata: dict) -> dict[str, float]:
    pp, pv, tp, tv = predicted_p[fluid], predicted_v[fluid], teacher_p[fluid], teacher_v[fluid]
    bounds = np.asarray(metadata["bounds"], np.float32)
    penetration = np.any((pp < bounds[:, 0]) | (pp > bounds[:, 1]), axis=1)
    radius = float(metadata["default_connectivity_radius"])
    pd, td = _density_proxy(pp, radius), _density_proxy(tp, radius)
    pred_energy = np.mean(0.5 * np.sum(pv * pv, axis=1) + 9.81 * pp[:, 1])
    true_energy = np.mean(0.5 * np.sum(tv * tv, axis=1) + 9.81 * tp[:, 1])
    # A projected SWE baseline cannot preserve particle identity vertically.
    # Report a coarse free-surface error alongside particle-wise RMSE so the
    # 2D baseline is evaluated in its native representation as well.
    bounds = np.asarray(metadata["bounds"], np.float32)
    resolution = 32
    def surface(position: np.ndarray) -> np.ndarray:
        ix = np.clip(((position[:, 0] - bounds[0, 0]) / max(bounds[0, 1] - bounds[0, 0], 1e-9) * resolution).astype(int), 0, resolution - 1)
        iz = np.clip(((position[:, 2] - bounds[2, 0]) / max(bounds[2, 1] - bounds[2, 0], 1e-9) * resolution).astype(int), 0, resolution - 1)
        result = np.full((resolution, resolution), np.nan, np.float32)
        flat = np.full(resolution * resolution, -np.inf, np.float32)
        np.maximum.at(flat, iz * resolution + ix, position[:, 1])
        result[:] = flat.reshape(resolution, resolution)
        result[~np.isfinite(result)] = np.nan
        return result
    ps, ts = surface(pp), surface(tp)
    common = np.isfinite(ps) & np.isfinite(ts)
    return {
        "position_rmse": float(np.sqrt(np.mean((pp - tp) ** 2))),
        "velocity_rmse": float(np.sqrt(np.mean((pv - tv) ** 2))),
        "penetration_rate": float(np.mean(penetration)),
        "density_relative_error": float(np.mean(np.abs(pd - td) / np.maximum(td, 1e-3))),
        "momentum_error_per_particle": float(np.linalg.norm(np.mean(pv - tv, axis=0))),
        "horizontal_momentum_error_per_particle": float(np.linalg.norm(np.mean(pv[:, (0, 2)] - tv[:, (0, 2)], axis=0))),
        "surface_height_rmse": float(np.sqrt(np.mean((ps[common] - ts[common]) ** 2))) if np.any(common) else float("nan"),
        "energy_excess_per_particle": float(max(0.0, pred_energy - true_energy)),
        "active_count_error": 0.0,
    }


def _rollout(condition: str, source: dict, metadata: dict, start: int, maximum: int, cfg: Phase3Config, model, device, capture: bool = False):
    teacher = np.asarray(source["position"], np.float32)
    types = np.asarray(source["particle_type"], np.int64)
    working = teacher.copy()
    # Only the six seed frames are visible. Future working frames are replaced
    # autoregressively; kinematic particles continue to follow teacher motion.
    output = {}
    trace = {"horizon": [], "teacher_position": [], "predicted_position": [], "teacher_velocity": [], "predicted_velocity": [], "error": []}
    fluid = types != KINEMATIC_ID
    objective = {"B": "all_residual", "C": "reversed", "D": "ours", "E": "gns", "G": "ours"}.get(condition, "ours")
    previous_v = working[start] - working[start - 1]
    swe = None
    if condition == "A":
        bounds = np.asarray(metadata["bounds"], np.float32)
        gravity = abs(float(np.asarray(metadata["acc_mean"])[1]))
        swe = ProjectedSWESolver(bounds, gravity, resolution=64)
        swe.initialize(working[start], previous_v, fluid, teacher[start, ~fluid])
    horizons = set(cfg.rollout_horizons)
    for step in range(1, maximum + 1):
        frame = start + step - 1
        if condition == "A":
            next_p, next_v = swe.advance_particles(1.0)
        else:
            local = {**source, "position": working}
            graph = graph_from_gns(local, metadata, frame, cfg, objective)
            acceleration = _condition_acceleration(condition, graph, metadata, model, device)
            next_v = previous_v + acceleration
            next_p = working[frame] + next_v
        next_p[~fluid] = teacher[frame + 1, ~fluid]
        next_v[~fluid] = teacher[frame + 1, ~fluid] - teacher[frame, ~fluid]
        working[frame + 1] = next_p
        previous_v = next_v
        if step in horizons:
            teacher_v = teacher[frame + 1] - teacher[frame]
            output[step] = _metrics(next_p, next_v, teacher[frame + 1], teacher_v, fluid, metadata)
            if capture:
                trace["horizon"].append(step)
                trace["teacher_position"].append(teacher[frame + 1].copy())
                trace["predicted_position"].append(next_p.copy())
                trace["teacher_velocity"].append(teacher_v.copy())
                trace["predicted_velocity"].append(next_v.copy())
                trace["error"].append(np.linalg.norm(next_p - teacher[frame + 1], axis=1))
    return output, trace


def evaluate_all(root: Path, cfg: Phase3Config) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for learned-condition evaluation")
    device = torch.device("cuda")
    raw = root / "raw" / cfg.dataset
    dataset = IndexedTFRecord(raw / "test.tfrecord", root / "indices" / cfg.dataset / "test.npy", raw / "metadata.json")
    models = {condition: _load_model(root, name, cfg, device) for condition, name in LEARNED.items()}
    rows = []
    splash_ratios = []
    # Classify every test trajectory before choosing examples. The selected
    # example is the within-tertile median, never a visually chosen success.
    for trajectory in range(len(dataset)):
        source = dataset.read(trajectory)
        frames = int(source["position"].shape[0])
        start = deterministic_windows(frames - max(cfg.rollout_horizons), 1, cfg.seed + 900_000 + trajectory)[0]
        probe = graph_from_gns(source, dataset.metadata, start, cfg, "ours")
        splash_ratios.append(float(np.mean(probe.routing_state == 1)))
    low, high = np.quantile(splash_ratios, (1 / 3, 2 / 3))
    ranked = sorted(range(len(splash_ratios)), key=lambda i: (splash_ratios[i], i))
    cuts = np.array_split(ranked, 3)
    groups = {name: list(map(int, members)) for name, members in zip(("quiet", "complex", "violent"), cuts)}
    scene_group_by_trajectory = {trajectory: group for group, members in groups.items() for trajectory in members}
    representatives = {}
    for group, members in groups.items():
        group_median = float(np.median([splash_ratios[i] for i in members]))
        representatives[group] = min(members, key=lambda i: (abs(splash_ratios[i] - group_median), i))
    representative_lookup = {trajectory: group for group, trajectory in representatives.items()}
    output_dir = root / "rollouts" / "representatives"
    output_dir.mkdir(parents=True, exist_ok=True)
    for trajectory in range(len(dataset)):
        source = dataset.read(trajectory)
        frames = int(source["position"].shape[0])
        start = deterministic_windows(frames - max(cfg.rollout_horizons), 1, cfg.seed + 900_000 + trajectory)[0]
        for condition in CONDITION_NAMES:
            capture = trajectory in representative_lookup
            metrics, trace = _rollout(condition, source, dataset.metadata, start, max(cfg.rollout_horizons), cfg, models.get(condition), device, capture)
            for horizon, values in metrics.items():
                rows.append({"trajectory": trajectory, "condition": condition, "condition_name": CONDITION_NAMES[condition], "horizon": horizon, "splash_ratio": splash_ratios[trajectory], **values})
            if capture:
                group = representative_lookup[trajectory]
                np.savez_compressed(
                    output_dir / f"{group}_{condition}.npz",
                    trajectory=np.int64(trajectory), start_frame=np.int64(start),
                    particle_id=np.arange(source["position"].shape[1], dtype=np.int64),
                    particle_type=np.asarray(source["particle_type"], np.int64),
                    horizon=np.asarray(trace["horizon"], np.int64),
                    teacher_position=np.asarray(trace["teacher_position"], np.float32),
                    predicted_position=np.asarray(trace["predicted_position"], np.float32),
                    teacher_velocity=np.asarray(trace["teacher_velocity"], np.float32),
                    predicted_velocity=np.asarray(trace["predicted_velocity"], np.float32),
                    position_error=np.asarray(trace["error"], np.float32),
                )
    for row in rows:
        row["scene_group"] = scene_group_by_trajectory[row["trajectory"]]
    output = root / "rollouts" / "ablation_results.json"
    output.write_text(json.dumps({"rows": rows, "splash_tertiles": [float(low), float(high)], "representatives": representatives}, indent=2), encoding="utf-8")
    csv_output = root / "rollouts" / "ablation_results.csv"
    if rows:
        with csv_output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    return output


def evaluate_zero_shot(root: Path, cfg: Phase3Config) -> Path:
    """Evaluate the WCSPH-trained common-feature checkpoint on all Water-3D test trajectories."""
    device = torch.device("cuda")
    model = _load_model(root, "wcsph_zero_shot", cfg, device)
    raw = root / "raw" / cfg.dataset
    dataset = IndexedTFRecord(raw / "test.tfrecord", root / "indices" / cfg.dataset / "test.npy", raw / "metadata.json")
    squared = 0.0; count = 0; rows = []
    for trajectory in range(len(dataset)):
        source = dataset.read(trajectory)
        frames = int(source["position"].shape[0])
        for frame in deterministic_windows(frames, cfg.validation_windows, cfg.seed + 800_000 + trajectory):
            graph = graph_from_gns(source, dataset.metadata, frame, cfg, "ours")
            node = torch.from_numpy(graph.node_features).to(device)
            types = torch.from_numpy(graph.particle_type).long().to(device)
            edge = torch.from_numpy(graph.edge_features).to(device)
            edge_index = torch.from_numpy(graph.edge_index).long().to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                prediction = model(node, types, edge, edge_index).float()
            target = torch.from_numpy(graph.target).to(device)
            mask = torch.from_numpy(graph.target_mask).bool().to(device)
            error = prediction[mask] - target[mask]
            value = float(torch.sqrt(torch.mean(error * error)))
            rows.append({"trajectory": trajectory, "frame": frame, "normalized_acceleration_rmse": value})
            squared += float(torch.sum(error * error)); count += error.numel()
    output = root / "rollouts" / "zero_shot_results.json"
    output.write_text(json.dumps({"aggregate_normalized_acceleration_rmse": float(np.sqrt(squared / max(count, 1))), "rows": rows}, indent=2), encoding="utf-8")
    return output


def evaluate_optimized_ours(root: Path, cfg: Phase3Config) -> Path:
    """Append a genuine ROI-graph autonomous rollout to the fixed A-F results."""
    output = root / "rollouts" / "ablation_results.json"
    if not output.exists():
        raise FileNotFoundError("A-F ablation_results.json must exist before optimized evaluation")
    stored = json.loads(output.read_text(encoding="utf-8"))
    old_rows = [row for row in stored["rows"] if row["condition"] != "G"]
    splash_by_trajectory = {}
    group_by_trajectory = {}
    for row in old_rows:
        splash_by_trajectory[int(row["trajectory"])] = float(row["splash_ratio"])
        group_by_trajectory[int(row["trajectory"])] = row["scene_group"]
    representatives = {key: int(value) for key, value in stored["representatives"].items()}
    representative_lookup = {value: key for key, value in representatives.items()}
    device = torch.device("cuda")
    model = _load_model(root, "ours", cfg, device)
    raw = root / "raw" / cfg.dataset
    dataset = IndexedTFRecord(raw / "test.tfrecord", root / "indices" / cfg.dataset / "test.npy", raw / "metadata.json")
    rows = []
    output_dir = root / "rollouts" / "representatives"; output_dir.mkdir(parents=True, exist_ok=True)
    for trajectory in range(len(dataset)):
        source = dataset.read(trajectory)
        frames = int(source["position"].shape[0])
        start = deterministic_windows(frames - max(cfg.rollout_horizons), 1, cfg.seed + 900_000 + trajectory)[0]
        capture = trajectory in representative_lookup
        metrics, trace = _rollout("G", source, dataset.metadata, start, max(cfg.rollout_horizons), cfg, model, device, capture)
        for horizon, values in metrics.items():
            rows.append({"trajectory": trajectory, "condition": "G", "condition_name": CONDITION_NAMES["G"],
                         "horizon": horizon, "splash_ratio": splash_by_trajectory[trajectory],
                         "scene_group": group_by_trajectory[trajectory], **values})
        if capture:
            group = representative_lookup[trajectory]
            np.savez_compressed(output_dir / f"{group}_G.npz", trajectory=np.int64(trajectory), start_frame=np.int64(start),
                                particle_id=np.arange(source["position"].shape[1], dtype=np.int64),
                                particle_type=np.asarray(source["particle_type"], np.int64), horizon=np.asarray(trace["horizon"], np.int64),
                                teacher_position=np.asarray(trace["teacher_position"], np.float32), predicted_position=np.asarray(trace["predicted_position"], np.float32),
                                teacher_velocity=np.asarray(trace["teacher_velocity"], np.float32), predicted_velocity=np.asarray(trace["predicted_velocity"], np.float32),
                                position_error=np.asarray(trace["error"], np.float32))
    combined = old_rows + rows
    output.write_text(json.dumps({**stored, "rows": combined}, indent=2), encoding="utf-8")
    csv_output = root / "rollouts" / "ablation_results.csv"
    with csv_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(combined[0])); writer.writeheader(); writer.writerows(combined)
    return output
