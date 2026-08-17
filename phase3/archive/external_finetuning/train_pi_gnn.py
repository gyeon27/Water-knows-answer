"""Train the proposed SPLASH-ROI residual PI-GNN on external DFSPH teacher data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from phase3.config import Phase3Config
from phase3.data import UnifiedGraph, radius_graph
from phase3.models import UnifiedGNS
from phase3.training import _tensor_graph, physics_informed_loss


def statistics(position: np.ndarray, velocity: np.ndarray, active: np.ndarray, frames: list[int], dt: float) -> dict:
    samples_v, samples_a = [], []
    for frame in frames:
        ids = np.flatnonzero(active[frame-5:frame+2].all(axis=0))
        if not ids.size:
            continue
        samples_v.append((velocity[frame-4:frame+1, ids] * dt).reshape(-1, 3))
        true_a = (velocity[frame+1, ids] - velocity[frame, ids]) * dt
        samples_a.append(true_a)
    all_v = np.concatenate(samples_v)
    all_a = np.concatenate(samples_a)
    base = np.array((0.0, -9.81 * dt * dt, 0.0), np.float32)
    residual = all_a - base
    live_position = position[active]
    margin = np.array((0.5, 0.5, 0.5), np.float32)
    return {
        "vel_mean": all_v.mean(0).astype(np.float32),
        "vel_std": np.maximum(all_v.std(0), 1e-4).astype(np.float32),
        "acc_mean": base,
        "acc_std": np.maximum(residual.std(0), 1e-4).astype(np.float32),
        "bounds": np.column_stack((live_position.min(0) - margin, live_position.max(0) + margin)).astype(np.float32),
        "default_connectivity_radius": 0.22,
    }


def make_graph(data: dict[str, np.ndarray], frame: int, stats: dict, cfg: Phase3Config) -> UnifiedGraph:
    position, velocity, active = data["positions"], data["velocities"], data["active_mask"]
    # Every selected node must have a complete five-velocity history and a
    # next-frame target. Newly emitted particles enter training after that.
    ids = np.flatnonzero(active[frame-5:frame+2].all(axis=0))
    if not ids.size:
        raise ValueError(f"no complete-history particles at frame {frame}")
    dt = float(data["dt"])
    current = position[frame, ids]
    current_v = velocity[frame, ids] * dt
    history = velocity[frame-4:frame+1, ids] * dt
    velocity_features = ((history - stats["vel_mean"]) / stats["vel_std"]).transpose(1, 0, 2).reshape(len(ids), 15)
    bounds = stats["bounds"]
    radius = float(stats["default_connectivity_radius"])
    distances = np.clip(np.concatenate((current - bounds[:, 0], bounds[:, 1] - current), axis=1) / radius, -1, 1)
    state = data["routing_state"][frame, ids].astype(np.uint8)
    state_onehot = np.eye(3, dtype=np.float32)[np.minimum(state, 2)]
    gravity = np.tile((0, -1, 0), (len(ids), 1)).astype(np.float32)
    node = np.concatenate((velocity_features, distances, state_onehot, gravity), axis=1).astype(np.float32)
    edge_index, edge_features = radius_graph(current, radius, cfg.max_neighbors)
    next_step_v = velocity[frame + 1, ids] * dt
    true_acc = next_step_v - current_v
    target = ((true_acc - stats["acc_mean"]) / stats["acc_std"]).astype(np.float32)
    mask = state == 1
    if not np.any(mask):
        raise ValueError(f"frame {frame} has no SPLASH target")
    return UnifiedGraph(
        node, np.zeros(len(ids), np.int64), edge_index, edge_features, target,
        mask, current.astype(np.float32), current_v.astype(np.float32), state,
        f"external_dfSPH:{frame}",
    )


@torch.no_grad()
def validate(model, arrays, frames, stats, cfg, device) -> float:
    model.eval()
    squared = 0.0
    count = 0
    selected = frames if len(frames) <= 24 else frames[::max(1, len(frames)//24)][:24]
    for frame in selected:
        batch = _tensor_graph(make_graph(arrays, frame, stats, cfg), device)
        with torch.autocast("cuda", dtype=torch.float16):
            prediction = model(batch["node"], batch["types"], batch["edge"], batch["edge_index"])
        error = prediction.float()[batch["mask"]] - batch["target"][batch["mask"]]
        squared += float((error * error).sum())
        count += error.numel()
    model.train()
    return float(np.sqrt(squared / max(count, 1)))


def train(path: Path, output: Path, steps: int, resume: bool) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for external PI-GNN training")
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    frames_total = arrays["positions"].shape[0]
    eligible = [f for f in range(5, frames_total-1)
                if np.any(arrays["active_mask"][f-5:f+2].all(axis=0) & arrays["splash_roi"][f])]
    cut_train = int(frames_total * 0.70)
    cut_valid = int(frames_total * 0.90)
    train_frames = [f for f in eligible if f < cut_train]
    valid_frames = [f for f in eligible if cut_train <= f < cut_valid]
    test_frames = [f for f in eligible if f >= cut_valid]
    if not train_frames or not valid_frames or not test_frames:
        raise ValueError(f"insufficient chronological split: {len(train_frames)}/{len(valid_frames)}/{len(test_frames)}")
    cfg = Phase3Config(training_steps=steps, validate_every=100, checkpoint_every=100)
    stats = statistics(arrays["positions"], arrays["velocities"], arrays["active_mask"], train_frames, float(arrays["dt"]))
    metadata = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in stats.items()}
    device = torch.device("cuda")
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    model = UnifiedGNS(hidden=cfg.hidden_size, blocks=cfg.message_blocks, type_embedding=cfg.particle_embedding).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=cfg.learning_rate * 0.05)
    scaler = torch.amp.GradScaler("cuda")
    output.mkdir(parents=True, exist_ok=True)
    latest, best = output / "latest.pt", output / "best.pt"
    step, best_rmse = 0, float("inf")
    if resume and latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"]); scaler.load_state_dict(state["scaler"])
        step, best_rmse = int(state["step"]), float(state["best_rmse"])
    log_path = output / "training_loss.csv"
    write_header = not log_path.exists() or not resume
    log = log_path.open("a" if resume else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(log, fieldnames=["step", "loss_total", "loss_supervised", "loss_penetration", "loss_momentum", "loss_density", "loss_energy", "validation_rmse", "steps_per_s", "gpu_memory_mb"])
    if write_header: writer.writeheader()
    started = time.perf_counter()
    model.train()
    while step < steps:
        frame = train_frames[random.randrange(len(train_frames))]
        batch = _tensor_graph(make_graph(arrays, frame, stats, cfg), device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            prediction = model(batch["node"], batch["types"], batch["edge"], batch["edge_index"])
            loss, components = physics_informed_loss(prediction, batch, metadata)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step+1}")
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
        scaler.step(optimizer); scaler.update(); scheduler.step(); step += 1
        validation_rmse = ""
        if step == 1 or step % 100 == 0 or step == steps:
            validation_rmse = validate(model, arrays, valid_frames, stats, cfg, device)
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "step": step, "best_rmse": min(best_rmse, validation_rmse), "metadata": metadata, "source": str(path), "splits": {"train": train_frames, "validation": valid_frames, "test": test_frames}, "config": cfg.__dict__, "model_name": "external_dfSPH_ours", "objective": "SPLASH residual PI-GNN"}
            torch.save(state, latest)
            if validation_rmse < best_rmse:
                best_rmse = validation_rmse; torch.save(state, best)
            print(json.dumps({"step": step, "loss": float(loss.detach()), "validation_rmse": validation_rmse, "best_rmse": best_rmse, "gpu_mb": torch.cuda.max_memory_allocated()/2**20}), flush=True)
        elapsed = max(time.perf_counter() - started, 1e-6)
        writer.writerow({"step": step, "loss_total": float(loss.detach()), **{f"loss_{k}": float(v.detach()) for k, v in components.items()}, "validation_rmse": validation_rmse, "steps_per_s": step/elapsed, "gpu_memory_mb": torch.cuda.max_memory_allocated()/2**20})
        if step % 25 == 0: log.flush()
    log.close()
    test_rmse = validate(model, arrays, test_frames, stats, cfg, device)
    summary = {"steps": step, "best_validation_rmse": best_rmse, "test_rmse": test_rmse, "frames": {"train": len(train_frames), "validation": len(valid_frames), "test": len(test_frames)}, "source": str(path.resolve()), "device": torch.cuda.get_device_name(0), "elapsed_s": time.perf_counter()-started}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return best


def main() -> None:
    here = Path(__file__).resolve().parents[2] / "external_teacher"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=here / "datasets/external_dfSPH_natural_cliff_001.npz")
    parser.add_argument("--output", type=Path, default=here / "checkpoints/pi_gnn_ours")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    train(args.data, args.output, args.steps, not args.no_resume)


if __name__ == "__main__":
    main()
