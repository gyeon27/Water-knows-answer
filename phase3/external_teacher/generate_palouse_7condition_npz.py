"""Generate A–G one-step prediction NPZ files on the same Palouse DEM teacher."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from phase3.config import Phase3Config
from phase3.data import UnifiedGraph, radius_graph
from phase3.evaluation import CONDITION_NAMES, LEARNED, _condition_acceleration, _load_model


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"E:\WaterKnowsAnswer_Phase3"))
    parser.add_argument("--teacher", type=Path, default=here / "datasets/external_dfSPH_palouse_falls_001.npz")
    parser.add_argument("--terrain", type=Path, default=here / "palouse_generated/terrain_height.npz")
    parser.add_argument("--output", type=Path, default=here / "palouse_7condition_npz")
    parser.add_argument("--frames", type=int, default=300)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for B/C/D/E/G generation")

    with np.load(args.teacher, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    with np.load(args.terrain, allow_pickle=False) as source:
        terrain = {key: source[key] for key in source.files}
    metadata = json.loads((args.data_root / "raw/Water-3D/metadata.json").read_text(encoding="utf-8"))
    cfg = Phase3Config()
    device = torch.device("cuda")
    models = {condition: _load_model(args.data_root, name, cfg, device) for condition, name in LEARNED.items()}

    positions = np.asarray(data["positions"], np.float32)
    velocities_si = np.asarray(data["velocities"], np.float32)
    active = np.asarray(data["active_mask"], bool)
    routing = np.asarray(data["routing_state"], np.uint8)
    dt = float(data["dt"])
    live = positions[active]
    source_min, source_max = live.min(0), live.max(0)
    scale = 0.70 / float(np.maximum(source_max - source_min, 1e-6).max())
    offset = np.full(3, 0.5, np.float32) - scale * (source_min + source_max) * 0.5
    normalized = positions * scale + offset
    bounds = np.asarray(metadata["bounds"], np.float32)
    radius = float(metadata["default_connectivity_radius"])
    vel_mean = np.asarray(metadata["vel_mean"], np.float32)
    vel_std = np.maximum(np.asarray(metadata["vel_std"], np.float32), 1e-8)
    acc_mean = np.asarray(metadata["acc_mean"], np.float32)

    end = positions.shape[0] - 2
    frame_ids = np.arange(max(5, end - args.frames + 1), end + 1, dtype=np.int32)
    slots = positions.shape[1]
    predictions = {condition: [] for condition in CONDITION_NAMES}
    predicted_velocities = {condition: [] for condition in CONDITION_NAMES}
    teacher_frames, teacher_velocities, active_frames, state_frames, valid_frames = [], [], [], [], []
    rejected = 0

    for frame in frame_ids:
        ids = np.flatnonzero(active[frame - 5:frame + 2].all(axis=0))
        teacher_next = positions[frame + 1].copy()
        teacher_velocity = velocities_si[frame + 1].copy()
        condition_position = {condition: teacher_next.copy() for condition in CONDITION_NAMES}
        condition_velocity = {condition: teacher_velocity.copy() for condition in CONDITION_NAMES}
        valid_slots = np.zeros(slots, bool)
        if ids.size:
            current = normalized[frame, ids]
            history = np.diff(normalized[frame - 5:frame + 1, ids], axis=0)
            current_v = history[-1]
            velocity_features = ((history - vel_mean) / vel_std).transpose(1, 0, 2).reshape(ids.size, 15)
            distances = np.clip(np.concatenate((current - bounds[:, 0], bounds[:, 1] - current), axis=1) / radius, -1, 1)
            state = routing[frame, ids]
            node = np.concatenate((velocity_features, distances, np.eye(3, dtype=np.float32)[np.minimum(state, 2)],
                                   np.tile((0.0, -1.0, 0.0), (ids.size, 1))), axis=1).astype(np.float32)
            edges, edge_features = radius_graph(current, radius, cfg.max_neighbors)
            graph = UnifiedGraph(node, np.zeros(ids.size, np.int64), edges, edge_features,
                                 np.zeros((ids.size, 3), np.float32), np.ones(ids.size, bool),
                                 current, current_v, state, f"palouse:{frame}")
            continuity_error = np.linalg.norm((positions[frame + 1, ids] - positions[frame, ids]) - teacher_velocity[ids] * dt, axis=1)
            # An emitter may recycle a particle slot without an inactive frame.
            # Reject if any transition in the six-frame model history jumps.
            history_steps = np.diff(positions[frame - 5:frame + 2, ids], axis=0)
            history_continuous = np.max(np.linalg.norm(history_steps, axis=2), axis=0) < 0.60
            continuous = (continuity_error < 0.44) & history_continuous
            rejected += int(np.count_nonzero(~continuous))
            valid_slots[ids[continuous]] = True
            for condition in CONDITION_NAMES:
                acceleration = _condition_acceleration(condition, graph, metadata, models.get(condition), device)
                displacement = current_v + acceleration
                predicted_normalized = current + displacement
                if condition == "G":
                    # Validation-calibrated confidence used by the optimized GUI.
                    analytic = current + current_v + acc_mean
                    splash = state == 1
                    predicted_normalized[splash] = analytic[splash] + 0.4010018813825451 * (predicted_normalized[splash] - analytic[splash])
                    displacement[splash] = predicted_normalized[splash] - current[splash]
                predicted_si = (predicted_normalized - offset) / scale
                usable = continuous
                condition_position[condition][ids[usable]] = predicted_si[usable]
                condition_velocity[condition][ids[usable]] = (displacement[usable] / (scale * dt)).astype(np.float32)
        teacher_frames.append(teacher_next)
        teacher_velocities.append(teacher_velocity)
        active_frames.append(active[frame + 1])
        state_frames.append(routing[frame])
        valid_frames.append(valid_slots)
        for condition in CONDITION_NAMES:
            predictions[condition].append(condition_position[condition])
            predicted_velocities[condition].append(condition_velocity[condition])

    args.output.mkdir(parents=True, exist_ok=True)
    common = {
        "frames": frame_ids + 1,
        "particle_id": np.arange(slots, dtype=np.int64),
        "teacher_position": np.asarray(teacher_frames, np.float32),
        "teacher_velocity": np.asarray(teacher_velocities, np.float32),
        "active_mask": np.asarray(active_frames, bool),
        "routing_state": np.asarray(state_frames, np.uint8),
        "prediction_valid_mask": np.asarray(valid_frames, bool),
        "terrain_height": np.asarray(terrain["height"], np.float32),
        "terrain_length_m": np.asarray(terrain["length_m"], np.float32),
        "terrain_width_m": np.asarray(terrain["width_m"], np.float32),
        "dt": np.asarray(dt, np.float32),
    }
    manifest = {"teacher": str(args.teacher.resolve()), "terrain": str(args.terrain.resolve()),
                "frames": int(len(frame_ids)), "slots": int(slots), "rejected_reuse_transitions": int(rejected),
                "coordinate_adapter": {"scale": scale, "offset": offset.tolist()}, "conditions": {}}
    for condition, name in CONDITION_NAMES.items():
        output = args.output / f"palouse_DEM_{condition}_{name.lower().replace('-', '_')}.npz"
        predicted = np.asarray(predictions[condition], np.float32)
        predicted_v = np.asarray(predicted_velocities[condition], np.float32)
        mask = common["active_mask"] & common["prediction_valid_mask"]
        splash_mask = mask & (common["routing_state"] == 1)
        error = np.linalg.norm(predicted - common["teacher_position"], axis=2)
        np.savez_compressed(output, **common, predicted_position=predicted,
                            predicted_velocity=predicted_v, position_error_m=error.astype(np.float32),
                            condition=np.asarray(condition), condition_name=np.asarray(name),
                            protocol=np.asarray("Palouse DEM external DFSPH; teacher-forced 1-step; identical frames A-G"))
        manifest["conditions"][condition] = {"name": name, "file": output.name,
            "valid_position_rmse_m": float(np.sqrt(np.mean(error[mask] ** 2))),
            "splash_position_rmse_m": float(np.sqrt(np.mean(error[splash_mask] ** 2))),
            "valid_samples": int(mask.sum()), "splash_samples": int(splash_mask.sum()),
            "bytes": output.stat().st_size}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
