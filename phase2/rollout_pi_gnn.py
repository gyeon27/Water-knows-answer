"""Create teacher-forced and autonomous PI-GNN comparison trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from .gnn import ResidualGNS, runtime_graph, terrain_sample
    from .shallow_water import TerrainData
except ImportError:  # direct script execution
    from gnn import ResidualGNS, runtime_graph, terrain_sample
    from shallow_water import TerrainData


def load_model(path: Path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ResidualGNS(checkpoint["node_size"], checkpoint["edge_size"], checkpoint["hidden_size"], checkpoint["blocks"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    norm = {key: torch.as_tensor(value, device=device) for key, value in checkpoint["normalization"].items()}
    return model, norm


@torch.no_grad()
def infer(model, norm, sample, device):
    if sample is None or sample.node_features.shape[0] == 0:
        return np.empty((0, 3), np.float32)
    node = torch.as_tensor(sample.node_features, device=device)
    edge = torch.as_tensor(sample.edge_features, device=device)
    edge_index = torch.as_tensor(sample.edge_index, dtype=torch.long, device=device)
    node = (node - norm["node_mean"]) / norm["node_std"]
    if edge.numel():
        edge = (edge - norm["edge_mean"]) / norm["edge_std"]
    with torch.autocast("cuda", dtype=torch.float16):
        output = model(node, edge, edge_index)
    return (output.float() * norm["target_std"] + norm["target_mean"]).cpu().numpy()


def project(position, velocity, active, terrain):
    ids = np.flatnonzero(active)
    if not ids.size:
        return
    bed, normal, _, _ = terrain_sample(terrain, position[ids])
    below = position[ids, 1] < bed + 0.015
    hit = ids[below]
    if hit.size:
        n = normal[below]
        position[hit, 1] = bed[below] + 0.015
        vn = np.sum(velocity[hit] * n, axis=1)
        inward = vn < 0
        velocity[hit[inward]] -= vn[inward, None] * n[inward]


def limit_mechanical_energy(previous_p, previous_v, next_p, next_v, ids, tolerance=0.02):
    """Prevent a learned residual from injecting net mechanical energy."""
    if not ids.size:
        return
    previous = np.sum(0.5 * np.sum(previous_v[ids] ** 2, axis=1) + 9.81 * previous_p[ids, 1])
    potential = np.sum(9.81 * next_p[ids, 1])
    kinetic = np.sum(0.5 * np.sum(next_v[ids] ** 2, axis=1))
    allowed_kinetic = max(0.0, previous * (1.0 + tolerance) - potential)
    if kinetic > allowed_kinetic and kinetic > 1e-8:
        next_v[ids] *= np.sqrt(allowed_kinetic / kinetic)


def density_proxy(position, radius=0.32):
    if not position.shape[0]:
        return np.empty(0, np.float32)
    cell = np.floor(position / radius).astype(np.int32)
    bins = {}
    for i, key in enumerate(map(tuple, cell)):
        bins.setdefault(key, []).append(i)
    density = np.zeros(position.shape[0], np.float32)
    for i, base in enumerate(cell):
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                for oz in (-1, 0, 1):
                    for j in bins.get(tuple(base + (ox, oy, oz)), ()):
                        distance = np.linalg.norm(position[i] - position[j])
                        if distance < radius:
                            density[i] += max(0.0, 1.0 - distance / radius) ** 3
    return density


def metrics(teacher_p, predicted_p, teacher_v, predicted_v, active, terrain, mass):
    ids = np.flatnonzero(active)
    if not ids.size:
        return 0, 0, 0, 0, 0
    error = np.linalg.norm(predicted_p[ids] - teacher_p[ids], axis=1)
    bed, _, _, _ = terrain_sample(terrain, predicted_p[ids])
    penetration = np.mean(predicted_p[ids, 1] < bed - 1e-4)
    momentum = np.linalg.norm(np.sum((predicted_v[ids] - teacher_v[ids]) * mass, axis=0)) / max(ids.size * mass, 1e-6)
    teacher_density = density_proxy(teacher_p[ids])
    predicted_density = density_proxy(predicted_p[ids])
    density_error = np.mean(np.abs(predicted_density - teacher_density) / np.maximum(teacher_density, 1e-3))
    pred_e = np.mean(0.5 * np.sum(predicted_v[ids] ** 2, axis=1) + 9.81 * predicted_p[ids, 1])
    true_e = np.mean(0.5 * np.sum(teacher_v[ids] ** 2, axis=1) + 9.81 * teacher_p[ids, 1])
    return float(np.sqrt(np.mean(error**2))), float(penetration), float(density_error), float(momentum), float(max(0, pred_e - true_e))


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=root / "checkpoints" / "pi_gnn_best.pt")
    parser.add_argument("--trajectory", type=Path, default=root / "datasets" / "wcsph" / "trajectory_001_natural_waterfall.npz")
    parser.add_argument("--output", type=Path, default=root / "outputs" / "pi_gnn_comparison.npz")
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; use .venv-gpu\\Scripts\\python.exe")
    device = torch.device("cuda")
    model, norm = load_model(args.checkpoint, device)
    with np.load(args.trajectory, allow_pickle=False) as data:
        teacher_p, teacher_v = data["positions"].copy(), data["velocities"].copy()
        teacher_active = data["active_mask"].copy()
        terrain_id, dt = str(data["terrain_id"].item()), float(data["dt"])
        metadata = json.loads(str(data["metadata_json"].item()))
        particle_ids = data["particle_id"].copy()
        flow = float(data["flow_rate"][0, 0])
    terrain = TerrainData.load(root / "terrains" / terrain_id)
    mass = float(metadata["config"]["particle_mass_kg"])
    frames, particles, _ = teacher_p.shape
    limit = min(frames, max(7, args.steps + 6))
    predicted_p, predicted_v = np.zeros_like(teacher_p[:limit]), np.zeros_like(teacher_v[:limit])
    one_p, one_v = teacher_p[:limit].copy(), teacher_v[:limit].copy()
    predicted_active = np.zeros_like(teacher_active[:limit])
    predicted_p[:6], predicted_v[:6], predicted_active[:6] = teacher_p[:6], teacher_v[:6], teacher_active[:6]
    first_active = np.full(particles, frames + 1)
    for i in range(particles):
        found = np.flatnonzero(teacher_active[:, i])
        if found.size:
            first_active[i] = found[0]
    metric_values = np.zeros((limit, 5), np.float32)
    for t in range(5, limit - 1):
        for autonomous, out_p, out_v, out_active in ((True, predicted_p, predicted_v, predicted_active), (False, one_p, one_v, teacher_active[:limit])):
            current_p = out_p[t] if autonomous else teacher_p[t]
            current_v = out_v[t] if autonomous else teacher_v[t]
            active = out_active[t].copy() if autonomous else teacher_active[t].copy()
            history = out_v[t - 5:t + 1] if autonomous else teacher_v[t - 5:t + 1]
            sample, selected = runtime_graph(current_p, history, active, terrain, particle_ids, mass, flow)
            delta = infer(model, norm, sample, device)
            next_v = current_v.copy()
            ids = np.flatnonzero(active)
            next_v[ids] = (current_v[ids] + np.array((0, -9.81 * dt, 0), np.float32)) * np.exp(-0.08 * dt)
            next_v[selected] += delta
            next_p = current_p.copy()
            next_p[ids] += next_v[ids] * dt
            next_active = active.copy()
            births = np.flatnonzero(first_active == t + 1)
            next_p[births], next_v[births], next_active[births] = teacher_p[t + 1, births], teacher_v[t + 1, births], True
            project(next_p, next_v, next_active, terrain)
            limit_mechanical_energy(current_p, current_v, next_p, next_v, ids)
            dead = (np.abs(next_p[:, 0]) > terrain.width_m * 0.5) | (next_p[:, 2] < 0) | (next_p[:, 2] > terrain.length_m)
            next_active[dead] = False
            out_p[t + 1], out_v[t + 1] = next_p, next_v
            if autonomous:
                out_active[t + 1] = next_active
        common = teacher_active[t + 1] & predicted_active[t + 1]
        metric_values[t + 1] = metrics(teacher_p[t + 1], predicted_p[t + 1], teacher_v[t + 1], predicted_v[t + 1], common, terrain, mass)
    error = np.linalg.norm(predicted_p - teacher_p[:limit], axis=2).astype(np.float32)
    one_step_error = np.linalg.norm(one_p - teacher_p[:limit], axis=2).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, teacher_positions=teacher_p[:limit], teacher_velocities=teacher_v[:limit], teacher_active=teacher_active[:limit], predicted_positions=predicted_p, predicted_velocities=predicted_v, predicted_active=predicted_active, one_step_positions=one_p, one_step_velocities=one_v, position_error=error, one_step_error=one_step_error, physics_metrics=metric_values, metric_names=np.asarray(["rmse_m", "penetration_rate", "density_error", "momentum_error_mps", "energy_excess"]), particle_id=particle_ids, terrain_id=np.asarray(terrain_id), dt=np.asarray(dt, np.float32))
    print(json.dumps({"output": str(args.output), "frames": limit, "final_metrics": dict(zip(["rmse_m", "penetration_rate", "density_error", "momentum_error_mps", "energy_excess"], metric_values[-1].tolist()))}, indent=2))


if __name__ == "__main__":
    main()
