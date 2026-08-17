"""Zero-shot test of the Water-3D-trained Ours checkpoint on external DFSPH."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from phase3.config import Phase3Config
from phase3.data import UnifiedGraph, radius_graph
from phase3.models import UnifiedGNS
from phase3.training import _tensor_graph


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=here / "datasets/external_dfSPH_palouse_falls_001.npz")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=here / "evaluation_palouse_water3d_ours")
    parser.add_argument("--frames", type=int, default=20)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    with np.load(args.data, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    cfg = Phase3Config(**checkpoint["config"])
    model = UnifiedGNS(hidden=cfg.hidden_size, blocks=cfg.message_blocks,
                       type_embedding=cfg.particle_embedding).cuda()
    model.load_state_dict(checkpoint["model"])
    model.eval()

    positions = np.asarray(data["positions"], np.float32)
    active = np.asarray(data["active_mask"], bool)
    routing = np.asarray(data["routing_state"], np.uint8)
    dt = float(data["dt"])
    live = positions[active]
    source_min, source_max = live.min(0), live.max(0)
    source_span = np.maximum(source_max - source_min, 1e-6)
    # Uniform mapping preserves cliff slope and particle geometry. Leave a
    # small margin inside the official Water-3D [0.125, 0.875]^3 box.
    scale = 0.70 / float(source_span.max())
    offset = np.full(3, 0.5, np.float32) - scale * (source_min + source_max) * 0.5
    normalized = positions * scale + offset

    bounds = np.asarray(metadata["bounds"], np.float32)
    radius = float(metadata["default_connectivity_radius"])
    vel_mean = np.asarray(metadata["vel_mean"], np.float32)
    vel_std = np.maximum(np.asarray(metadata["vel_std"], np.float32), 1e-8)
    acc_mean = np.asarray(metadata["acc_mean"], np.float32)
    acc_std = np.maximum(np.asarray(metadata["acc_std"], np.float32), 1e-8)
    end = positions.shape[0] - 2
    frames = list(range(max(5, end - args.frames + 1), end + 1))
    rows = []
    model_p, model_v, base_p, base_v = [], [], [], []
    rejected = 0
    comparison = []

    for frame in frames:
        ids = np.flatnonzero(active[frame-5:frame+2].all(axis=0))
        if not ids.size:
            continue
        current = normalized[frame, ids]
        history = np.diff(normalized[frame-5:frame+1, ids], axis=0)
        current_v = history[-1]
        velocity_features = ((history - vel_mean) / vel_std).transpose(1, 0, 2).reshape(len(ids), 15)
        distances = np.clip(np.concatenate((current - bounds[:, 0], bounds[:, 1] - current), axis=1) / radius, -1, 1)
        state = routing[frame, ids]
        onehot = np.eye(3, dtype=np.float32)[np.minimum(state, 2)]
        gravity = np.tile((0.0, -1.0, 0.0), (len(ids), 1)).astype(np.float32)
        node = np.concatenate((velocity_features, distances, onehot, gravity), axis=1).astype(np.float32)
        edges, edge_features = radius_graph(current, radius, cfg.max_neighbors)
        mask = state == 1
        true_next = normalized[frame + 1, ids]
        true_disp = true_next - current
        teacher_v = np.asarray(data["velocities"][frame + 1, ids], np.float32)
        continuity_error = np.linalg.norm((positions[frame + 1, ids] - positions[frame, ids]) - teacher_v * dt, axis=1)
        continuous = continuity_error < 0.44
        continuous &= np.linalg.norm(positions[frame + 1, ids] - positions[frame, ids], axis=1) < 0.60
        rejected += int(np.count_nonzero(mask & ~continuous))
        mask &= continuous
        if not np.any(mask):
            comparison.append((frame + 1, positions[frame + 1].copy(), positions[frame + 1].copy(),
                               positions[frame + 1].copy(),
                               active[frame + 1].copy(), routing[frame,].copy()))
            continue
        true_acc = true_disp - current_v
        target = (true_acc - acc_mean) / acc_std
        graph = UnifiedGraph(node, np.zeros(len(ids), np.int64), edges, edge_features,
                             target.astype(np.float32), mask, current, current_v, state, f"palouse:{frame}")
        batch = _tensor_graph(graph, torch.device("cuda"))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            prediction = model(batch["node"], batch["types"], batch["edge"], batch["edge_index"])
        predicted_acc = prediction.float().cpu().numpy() * acc_std + acc_mean
        predicted_disp = current_v + predicted_acc
        baseline_disp = current_v + acc_mean
        mp = np.linalg.norm((current + predicted_disp - true_next)[mask], axis=1) / scale
        bp = np.linalg.norm((current + baseline_disp - true_next)[mask], axis=1) / scale
        mv = np.linalg.norm((predicted_disp - true_disp)[mask], axis=1) / (scale * dt)
        bv = np.linalg.norm((baseline_disp - true_disp)[mask], axis=1) / (scale * dt)
        model_p.extend(mp); model_v.extend(mv); base_p.extend(bp); base_v.extend(bv)
        predicted_si = positions[frame + 1].copy()
        baseline_si = positions[frame + 1].copy()
        predicted_si[ids[mask]] = ((current + predicted_disp - offset) / scale)[mask]
        baseline_si[ids[mask]] = ((current + baseline_disp - offset) / scale)[mask]
        comparison.append((frame + 1, positions[frame + 1].copy(), baseline_si, predicted_si,
                           active[frame + 1].copy(), routing[frame,].copy()))
        rows.append({"frame": frame, "splash_nodes": int(mask.sum()),
                     "model_position_rmse_m": float(np.sqrt(np.mean(mp**2))),
                     "model_velocity_rmse_mps": float(np.sqrt(np.mean(mv**2))),
                     "baseline_position_rmse_m": float(np.sqrt(np.mean(bp**2))),
                     "baseline_velocity_rmse_mps": float(np.sqrt(np.mean(bv**2)))})

    def rmse(values: list[float]) -> float:
        return float(np.sqrt(np.mean(np.square(values)))) if values else float("nan")

    summary = {
        "protocol": "Water-3D-trained optimized Ours zero-shot on Palouse Falls external DFSPH; 1-step teacher-forced",
        "coordinate_adapter": {"uniform_scale_per_m": scale, "offset": offset.tolist(),
                               "source_bounds_m": np.column_stack((source_min, source_max)).tolist()},
        "frames": len(rows), "splash_values": len(model_p),
        "rejected_particle_reuse_transitions": rejected,
        "water3d_ours_position_rmse_m": rmse(model_p),
        "water3d_ours_velocity_rmse_mps": rmse(model_v),
        "analytic_baseline_position_rmse_m": rmse(base_p),
        "analytic_baseline_velocity_rmse_mps": rmse(base_v),
        "checkpoint": str(args.checkpoint.resolve()), "teacher": str(args.data.resolve()),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    if rows:
        with (args.output / "test_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if comparison:
        teacher_cache = np.stack([item[1] for item in comparison])
        baseline_cache = np.stack([item[2] for item in comparison])
        raw_model_cache = np.stack([item[3] for item in comparison])
        active_cache = np.stack([item[4] for item in comparison])
        routing_cache = np.stack([item[5] for item in comparison])
        midpoint = len(comparison) // 2
        calibration = active_cache[:midpoint] & (routing_cache[:midpoint] == 1)
        delta = raw_model_cache[:midpoint] - baseline_cache[:midpoint]
        target_delta = teacher_cache[:midpoint] - baseline_cache[:midpoint]
        numerator = float(np.sum(delta[calibration] * target_delta[calibration]))
        denominator = float(np.sum(delta[calibration] * delta[calibration]))
        blend_alpha = float(np.clip(numerator / max(denominator, 1e-12), 0.0, 1.0))
        optimized_cache = baseline_cache + blend_alpha * (raw_model_cache - baseline_cache)
        test_mask = active_cache[midpoint:] & (routing_cache[midpoint:] == 1)
        def cache_rmse(values: np.ndarray) -> float:
            error = np.linalg.norm(values[midpoint:] - teacher_cache[midpoint:], axis=2)
            return float(np.sqrt(np.mean(error[test_mask] ** 2)))
        summary["runtime_confidence_blend"] = {
            "calibration_frames": midpoint,
            "held_out_test_frames": len(comparison) - midpoint,
            "alpha": blend_alpha,
            "test_baseline_position_rmse_m": cache_rmse(baseline_cache),
            "test_raw_model_position_rmse_m": cache_rmse(raw_model_cache),
            "test_optimized_ours_position_rmse_m": cache_rmse(optimized_cache),
        }
        np.savez_compressed(
            args.output / "gui_comparison.npz",
            frames=np.asarray([item[0] for item in comparison], np.int32),
            teacher_position=teacher_cache,
            baseline_position=baseline_cache,
            raw_model_position=raw_model_cache,
            predicted_position=optimized_cache,
            active_mask=active_cache,
            routing_state=routing_cache,
            blend_alpha=np.asarray(blend_alpha, np.float32),
            model_name=np.asarray("Water-3D trained Optimized Ours"),
            protocol=np.asarray("teacher-forced 1-step SPLASH ROI; alpha calibrated on first half only"),
        )
    (args.output / "test_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
