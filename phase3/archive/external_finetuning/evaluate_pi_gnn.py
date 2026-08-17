"""Held-out external DFSPH one-step evaluation for the proposed PI-GNN."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from phase3.config import Phase3Config
from phase3.models import UnifiedGNS
from phase3.training import _tensor_graph, physics_informed_loss
from phase3.archive.external_finetuning.train_pi_gnn import make_graph


def main() -> None:
    here = Path(__file__).resolve().parents[2] / "external_teacher"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=here / "datasets/external_dfSPH_natural_cliff_001.npz")
    parser.add_argument("--checkpoint", type=Path, default=here / "checkpoints/pi_gnn_ours/best.pt")
    parser.add_argument("--output", type=Path, default=here / "evaluation")
    parser.add_argument("--frames", type=int, default=20,
                        help="number of final valid frames for a new-terrain generalization test")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = Phase3Config(**{k: v for k, v in state["config"].items() if k in Phase3Config.__dataclass_fields__})
    model = UnifiedGNS(hidden=cfg.hidden_size, blocks=cfg.message_blocks, type_embedding=cfg.particle_embedding).to(device)
    model.load_state_dict(state["model"]); model.eval()
    stats = {key: np.asarray(value, np.float32) if key in {"vel_mean", "vel_std", "acc_mean", "acc_std", "bounds"} else value for key, value in state["metadata"].items()}
    with np.load(args.data, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    saved_frames = [int(x) for x in state["splits"]["test"]]
    last_valid = int(arrays["positions"].shape[0] - 2)
    if saved_frames and max(saved_frames) <= last_valid:
        test_frames = saved_frames
    else:
        first = max(5, last_valid - args.frames + 1)
        test_frames = list(range(first, last_valid + 1))
    dt = float(arrays["dt"])
    rows, cache = [], []
    squared_p = squared_v = 0.0
    values = rejected_reuse = 0
    for frame in test_frames:
        graph = make_graph(arrays, frame, stats, cfg)
        ids = np.flatnonzero(arrays["active_mask"][frame-5:frame+2].all(axis=0))
        batch = _tensor_graph(graph, device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            prediction = model(batch["node"], batch["types"], batch["edge"], batch["edge_index"])
            loss, components = physics_informed_loss(prediction, batch, state["metadata"])
        pred_acc = prediction.float().cpu().numpy() * stats["acc_std"] + stats["acc_mean"]
        # Ours: analytic gravity for all particles, learned residual only in
        # SPLASH ROI. Non-SPLASH nodes therefore receive exactly the baseline.
        mask = graph.target_mask.copy()
        pred_acc[~mask] = stats["acc_mean"]
        pred_step_v = graph.velocities + pred_acc
        pred_position = graph.positions + pred_step_v
        teacher_position = arrays["positions"][frame + 1, ids]
        teacher_velocity = arrays["velocities"][frame + 1, ids]
        # Emitter particle reuse can preserve an ID while teleporting that
        # slot from the outlet back to the source. Such a transition is not a
        # physical trajectory and must not be scored as model position error.
        teacher_step = teacher_position - graph.positions
        continuity_error = np.linalg.norm(teacher_step - teacher_velocity * dt, axis=1)
        continuous = continuity_error < max(0.25, 2.0 * float(stats["default_connectivity_radius"]))
        rejected_reuse += int(np.count_nonzero(mask & ~continuous))
        mask &= continuous
        if not np.any(mask):
            continue
        pred_velocity = pred_step_v / dt
        position_error = np.linalg.norm(pred_position[mask] - teacher_position[mask], axis=1)
        velocity_error = np.linalg.norm(pred_velocity[mask] - teacher_velocity[mask], axis=1)
        squared_p += float(np.sum(position_error ** 2)); squared_v += float(np.sum(velocity_error ** 2)); values += int(mask.sum())
        rows.append({
            "frame": frame, "splash_nodes": int(mask.sum()),
            "position_rmse_m": float(np.sqrt(np.mean(position_error ** 2))),
            "velocity_rmse_mps": float(np.sqrt(np.mean(velocity_error ** 2))),
            "loss_total": float(loss),
            **{f"loss_{key}": float(value) for key, value in components.items()},
        })
        cache.append((frame, ids, graph.positions, teacher_position, pred_position, graph.routing_state, mask, position_error))
    args.output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (args.output / "test_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    max_nodes = max(len(item[1]) for item in cache)
    shape = (len(cache), max_nodes, 3)
    current = np.zeros(shape, np.float32); teacher = np.zeros(shape, np.float32); predicted = np.zeros(shape, np.float32)
    routing = np.full((len(cache), max_nodes), 3, np.uint8); active = np.zeros((len(cache), max_nodes), bool); error = np.zeros((len(cache), max_nodes), np.float32)
    particle_id = np.full((len(cache), max_nodes), -1, np.int32)
    for i, (_, ids, cur, true, pred, state_value, eval_mask, splash_error) in enumerate(cache):
        n = len(ids); active[i, :n] = True; particle_id[i, :n] = ids
        current[i, :n] = cur; teacher[i, :n] = true; predicted[i, :n] = pred; routing[i, :n] = state_value
        error[i, np.flatnonzero(eval_mask)] = splash_error
    np.savez_compressed(args.output / "test_comparison.npz", frames=np.asarray(test_frames), particle_id=particle_id,
                        active_mask=active, current_position=current, teacher_position=teacher,
                        predicted_position=predicted, routing_state=routing, position_error_m=error)
    summary = {
        "protocol": "held-out chronological 1-step teacher-forced; GNN residual applied to SPLASH only",
        "frames": len(test_frames), "splash_values": values,
        "rejected_particle_reuse_transitions": rejected_reuse,
        "position_rmse_m": float(np.sqrt(squared_p / max(values, 1))),
        "velocity_rmse_mps": float(np.sqrt(squared_v / max(values, 1))),
        "mean_physics_losses": {key: float(np.mean([row[key] for row in rows])) for key in fields if key.startswith("loss_")},
        "checkpoint": str(args.checkpoint.resolve()), "teacher": str(args.data.resolve()),
    }
    (args.output / "test_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
