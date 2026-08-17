"""Run the Phase-3 optimized ROI UnifiedGNS on a Phase-2 height-map waterfall."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from phase2.gnn.runtime import terrain_sample
from phase2.rollout_pi_gnn import limit_mechanical_energy, metrics, project
from phase2.shallow_water import TerrainData

from phase3.config import Phase3Config, resolve_data_root
from phase3.data import graph_from_wcsph
from phase3.evaluation import _condition_acceleration, _load_model


def _normalization(velocity: np.ndarray, active: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    displacement_v = velocity * dt
    acceleration = np.diff(displacement_v, axis=0)
    mask = active[1:] & active[:-1]
    values = acceleration[mask]
    if not values.size:
        return np.array((0.0, -9.81 * dt * dt, 0.0), np.float32), np.ones(3, np.float32)
    return values.mean(0).astype(np.float32), np.maximum(values.std(0), 1e-5).astype(np.float32)


def _route(position: np.ndarray, velocity: np.ndarray, active: np.ndarray, terrain: TerrainData) -> np.ndarray:
    state = np.zeros(active.shape, bool)
    ids = np.flatnonzero(active)
    if not ids.size:
        return state
    bed, normal, _slope, cliff = terrain_sample(terrain, position[ids])
    clearance = position[ids, 1] - bed
    speed = np.linalg.norm(velocity[ids], axis=1)
    approach = np.sum(velocity[ids] * normal, axis=1)
    # Same present-state-only terrain router as the Phase-2 runtime.  No
    # teacher SPLASH mask or future frame is consulted.
    roi = ((clearance < 0.65) & ((approach < -0.15) | (speed > 1.0))) | (cliff & (clearance < 1.2))
    state[ids[roi]] = True
    return state


def run(args: argparse.Namespace) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    phase2_root = Path(__file__).resolve().parents[3] / "phase2"
    terrain_root = phase2_root / "terrains"
    with np.load(args.trajectory, allow_pickle=False) as data:
        teacher_p = data["positions"].astype(np.float32)
        teacher_v = data["velocities"].astype(np.float32)
        teacher_a = data["active_mask"].astype(bool)
        terrain_id = str(data["terrain_id"].item())
        dt = float(data["dt"])
        metadata_json = json.loads(str(data["metadata_json"].item()))
        particle_ids = data["particle_id"].astype(np.int32)
    terrain = TerrainData.load(terrain_root / terrain_id)
    mass = float(metadata_json["config"]["particle_mass_kg"])
    root = resolve_data_root(args.data_root)
    cfg = Phase3Config()
    device = torch.device("cuda")
    # Same optimized Phase-3 architecture and SPLASH-only graph execution,
    # but the checkpoint trained on the WCSPH domain rather than Water-3D.
    model = _load_model(root, "wcsph_zero_shot", cfg, device)
    acc_mean, acc_std = _normalization(teacher_v, teacher_a, dt)
    metadata = {
        "acc_mean": acc_mean,
        "acc_std": acc_std,
        "default_connectivity_radius": 0.32,
    }

    start = max(5, int(args.start))
    end = min(teacher_p.shape[0] - 1, start + int(args.steps))
    count = end - start + 1
    predicted_p = teacher_p.copy()
    predicted_v = teacher_v.copy()
    predicted_a = teacher_a.copy()
    routing = np.zeros_like(teacher_a)
    first_active = np.full(teacher_a.shape[1], teacher_a.shape[0] + 1, np.int32)
    for particle in range(teacher_a.shape[1]):
        born = np.flatnonzero(teacher_a[:, particle])
        if born.size:
            first_active[particle] = int(born[0])
    metric_values = np.zeros((count, 5), np.float32)
    roi_count = np.zeros(count, np.int32)

    for frame in range(start, end):
        active = predicted_a[frame].copy()
        splash = _route(predicted_p[frame], predicted_v[frame], active, terrain)
        routing[frame] = splash
        graph = graph_from_wcsph(
            args.trajectory, terrain_root, frame, cfg,
            positions_override=predicted_p,
            velocities_override=predicted_v,
            active_override=predicted_a,
            splash_override=routing,
            build_edges=False,
        )
        active_ids = np.flatnonzero(active)
        roi_count[frame - start] = int(np.sum(splash))
        acceleration = _condition_acceleration("G", graph, metadata, model, device)
        # Cheap analytic base for ordinary flow; the learned acceleration is
        # substituted only on routed 3-D SPLASH particles.  Using the global
        # WCSPH mean outside the ROI underestimates gravity by almost 10x on
        # this trajectory and makes the hybrid drift immediately.
        base_v = (predicted_v[frame, active_ids] + np.array((0.0, -9.81 * dt, 0.0), np.float32)) * np.exp(-0.08 * dt)
        displacement_v = base_v * dt
        local_splash = splash[active_ids]
        current_displacement = predicted_v[frame, active_ids] * dt
        displacement_v[local_splash] = (
            current_displacement[local_splash] + acceleration[: active_ids.size][local_splash]
        )
        next_v = predicted_v[frame].copy()
        next_p = predicted_p[frame].copy()
        next_v[active_ids] = displacement_v / dt
        next_p[active_ids] += displacement_v
        next_active = active.copy()
        births = np.flatnonzero(first_active == frame + 1)
        next_p[births] = teacher_p[frame + 1, births]
        next_v[births] = teacher_v[frame + 1, births]
        next_active[births] = True
        project(next_p, next_v, next_active, terrain)
        limit_mechanical_energy(predicted_p[frame], predicted_v[frame], next_p, next_v, active_ids)
        dead = ((np.abs(next_p[:, 0]) > terrain.width_m * 0.5) |
                (next_p[:, 2] < 0) | (next_p[:, 2] > terrain.length_m))
        next_active[dead] = False
        predicted_p[frame + 1] = next_p
        predicted_v[frame + 1] = next_v
        predicted_a[frame + 1] = next_active
        common = teacher_a[frame + 1] & next_active
        metric_values[frame + 1 - start] = metrics(
            teacher_p[frame + 1], next_p, teacher_v[frame + 1], next_v,
            common, terrain, mass,
        )

    routing[end] = _route(predicted_p[end], predicted_v[end], predicted_a[end], terrain)
    roi_count[-1] = int(np.sum(routing[end]))
    sl = slice(start, end + 1)
    error = np.linalg.norm(predicted_p[sl] - teacher_p[sl], axis=2).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        teacher_positions=teacher_p[sl], teacher_velocities=teacher_v[sl], teacher_active=teacher_a[sl],
        predicted_positions=predicted_p[sl], predicted_velocities=predicted_v[sl], predicted_active=predicted_a[sl],
        one_step_positions=teacher_p[sl], one_step_velocities=teacher_v[sl],
        position_error=error, one_step_error=np.zeros_like(error), physics_metrics=metric_values,
        metric_names=np.asarray(["rmse_m", "penetration_rate", "density_error", "momentum_error_mps", "energy_excess"]),
        particle_id=particle_ids, terrain_id=np.asarray(terrain_id), dt=np.asarray(dt, np.float32),
        routing_state=routing[sl], roi_count=roi_count,
        model_name=np.asarray("Phase3 UnifiedGNS / WCSPH-domain Optimized-Ours"),
    )
    print(json.dumps({
        "output": str(args.output), "terrain": terrain_id, "frames": count,
        "max_roi": int(roi_count.max()),
        "final_metrics": dict(zip(["rmse_m", "penetration_rate", "density_error", "momentum_error_mps", "energy_excess"], metric_values[-1].tolist())),
    }, ensure_ascii=False, indent=2))
    return args.output


def main() -> None:
    project = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="auto")
    parser.add_argument("--trajectory", type=Path, default=project / "phase2" / "datasets" / "wcsph" / "trajectory_001_natural_waterfall.npz")
    parser.add_argument("--output", type=Path, default=project / "phase2" / "outputs" / "phase3_optimized_natural_waterfall.npz")
    parser.add_argument("--start", type=int, default=5, help="first frame after the six-frame history seed")
    parser.add_argument("--steps", type=int, default=114, help="default covers the remaining 120-frame Phase-2 timeline")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
