"""CUDA/AMP trainer for the physics-informed residual GNS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from gnn import ResidualGNS, TrajectoryGraphDataset, dynamic_batches, pack_graphs, physics_losses


WEIGHTS = {"supervised": 1.0, "penetration": 0.10, "momentum": 0.05, "density": 0.05, "energy": 0.05}


def require_cuda(device: str) -> torch.device:
    if device != "cuda":
        raise ValueError("PI-GNN training requires --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run with .venv-gpu\\Scripts\\python.exe and verify the NVIDIA driver.")
    tensor = torch.ones(32, device="cuda", requires_grad=True)
    tensor.square().sum().backward()
    return torch.device("cuda")


def split_files(files: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    if len(files) >= 12:
        return files[:8], files[8:10], files[10:12]
    if len(files) >= 3:
        return files[:-2], files[-2:-1], files[-1:]
    if len(files) == 2:
        return files[:1], files[1:], files[1:]
    raise ValueError("at least two trajectories are required")


def load_samples(files: list[Path], terrain_root: Path):
    dataset = TrajectoryGraphDataset(files, terrain_root, cache_trajectories=True)
    return [dataset[i] for i in range(len(dataset))]


def curriculum_horizon(step: int, total: int) -> int:
    fraction = step / max(total, 1)
    return 1 if fraction < .2 else 4 if fraction < .4 else 8 if fraction < .6 else 16 if fraction < .8 else 32


def truncated_rollout_loss(model, sequence, normalization, device):
    """Velocity-feedback rollout; topology is rebuilt from each teacher frame."""
    previous_ids, previous_velocity = None, None
    losses = []
    for sample in sequence:
        batch = move(pack_graphs([sample], pin_memory=True), device)
        raw_node = batch["node_features"].clone()
        if previous_ids is not None:
            common, previous_index, current_index = np.intersect1d(previous_ids, sample.particle_id, return_indices=True)
            if common.size:
                raw_node[torch.as_tensor(current_index, device=device), 15:18] = previous_velocity[torch.as_tensor(previous_index, device=device)]
        node = (raw_node - normalization["node_mean"]) / normalization["node_std"]
        edge = batch["edge_features"]
        if edge.numel():
            edge = (edge - normalization["edge_mean"]) / normalization["edge_std"]
        with torch.autocast("cuda", dtype=torch.float16):
            prediction_n = model(node, edge, batch["edge_index"])
        delta = prediction_n.float() * normalization["target_std"] + normalization["target_mean"]
        mask = batch["splash_mask"]
        losses.append(torch.mean((delta[mask] - batch["target_delta_v"][mask]) ** 2))
        current_v = raw_node[:, 15:18]
        base_v = (current_v + raw_node[:, 30:33] * 9.81 / 30.0) * np.exp(-0.08 / 30.0)
        previous_velocity = base_v + delta
        previous_ids = sample.particle_id
    return torch.stack(losses).mean()


def stats(samples, name: str):
    arrays = [getattr(sample, name) for sample in samples if getattr(sample, name).size]
    value = np.concatenate(arrays, axis=0)
    return value.mean(0).astype(np.float32), np.maximum(value.std(0), 1e-5).astype(np.float32)


def move(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def normalized_inputs(batch, normalization):
    node = (batch["node_features"] - normalization["node_mean"]) / normalization["node_std"]
    edge = batch["edge_features"]
    if edge.numel():
        edge = (edge - normalization["edge_mean"]) / normalization["edge_std"]
    return node, edge


@torch.no_grad()
def validate(model, samples, normalization, device, max_nodes, max_edges):
    model.eval()
    squared, count = 0.0, 0
    component_sums = {name: 0.0 for name in WEIGHTS}
    batches = 0
    for group in dynamic_batches(samples, max_nodes, max_edges, shuffle=False):
        batch = move(pack_graphs(group, pin_memory=True), device)
        node, edge = normalized_inputs(batch, normalization)
        with torch.autocast("cuda", dtype=torch.float16):
            prediction_n = model(node, edge, batch["edge_index"])
        prediction = prediction_n.float() * normalization["target_std"] + normalization["target_mean"]
        losses = physics_losses(prediction, batch["target_delta_v"], batch)
        error = prediction[batch["splash_mask"]] - batch["target_delta_v"][batch["splash_mask"]]
        squared += float(torch.sum(error * error))
        count += error.numel()
        for name, value in losses.items():
            component_sums[name] += float(value)
        batches += 1
    model.train()
    return {"rmse_mps": (squared / max(count, 1)) ** 0.5, **{f"val_{name}": value / max(batches, 1) for name, value in component_sums.items()}}


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "datasets" / "wcsph_pi")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-nodes", type=int, default=8_000)
    parser.add_argument("--max-edges", type=int, default=120_000)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--rollout-every", type=int, default=50)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, default=root / "checkpoints" / "pi_gnn_best.pt")
    args = parser.parse_args()
    device = require_cuda(args.device)
    torch.manual_seed(20260809)
    random.seed(20260809)
    np.random.seed(20260809)
    files = sorted(args.data.glob("*.npz"))
    if len(files) < 2 and args.data.name == "wcsph_pi":
        files = sorted((root / "datasets" / "wcsph").glob("*.npz"))
    train_files, validation_files, test_files = split_files(files)
    print(f"GPU={torch.cuda.get_device_name(0)} CUDA={torch.version.cuda}")
    print(f"loading train={len(train_files)} validation={len(validation_files)} test={len(test_files)} trajectories")
    train_groups = [load_samples([path], root / "terrains") for path in train_files]
    train_samples = [sample for group in train_groups for sample in group]
    validation_samples = load_samples(validation_files, root / "terrains")
    node_mean, node_std = stats(train_samples, "node_features")
    edge_mean, edge_std = stats(train_samples, "edge_features")
    target_mean, target_std = stats(train_samples, "target_delta_v")
    normalization = {name: torch.as_tensor(value, device=device) for name, value in {
        "node_mean": node_mean, "node_std": node_std, "edge_mean": edge_mean,
        "edge_std": edge_std, "target_mean": target_mean, "target_std": target_std,
    }.items()}
    first = train_samples[0]
    model = ResidualGNS(first.node_features.shape[1], first.edge_features.shape[1], args.hidden, args.blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.05)
    scaler = torch.amp.GradScaler("cuda")
    start_step, best_rmse, history = 0, float("inf"), []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint["step"])
        best_rmse = float(checkpoint.get("best_validation_rmse_mps", best_rmse))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    batch_iterator = iter(dynamic_batches(train_samples, args.max_nodes, args.max_edges))
    started = time.perf_counter()
    model.train()
    for step in range(start_step + 1, args.steps + 1):
        try:
            group = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(dynamic_batches(train_samples, args.max_nodes, args.max_edges))
            group = next(batch_iterator)
        batch = move(pack_graphs(group, pin_memory=True), device)
        node, edge = normalized_inputs(batch, normalization)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            prediction_n = model(node, edge, batch["edge_index"])
        prediction = prediction_n.float() * normalization["target_std"] + normalization["target_mean"]
        losses = physics_losses(prediction, batch["target_delta_v"], batch)
        loss = sum(WEIGHTS[name] * losses[name] for name in WEIGHTS)
        rollout_horizon = curriculum_horizon(step, args.steps)
        rollout_value = prediction.sum() * 0.0
        if rollout_horizon > 1 and step % args.rollout_every == 0:
            candidates = [group for group in train_groups if len(group) >= rollout_horizon]
            group = random.choice(candidates)
            start = random.randrange(len(group) - rollout_horizon + 1)
            sequence = group[start : start + rollout_horizon]
            rollout_value = truncated_rollout_loss(model, sequence, normalization, device)
            loss = loss + 0.25 * rollout_value
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= previous_scale:
            scheduler.step()
        if step == 1 or step % args.validate_every == 0 or step == args.steps:
            metrics = validate(model, validation_samples, normalization, device, args.max_nodes, args.max_edges)
            record = {"step": step, "loss": float(loss.detach()), "rollout_horizon": rollout_horizon, "rollout_loss": float(rollout_value.detach()), **{name: float(value.detach()) for name, value in losses.items()}, **metrics, "gpu_memory_mb": torch.cuda.max_memory_allocated() / 1048576}
            history.append(record)
            print(json.dumps(record, ensure_ascii=False))
            state = {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "step": step, "best_validation_rmse_mps": min(best_rmse, metrics["rmse_mps"]), "node_size": first.node_features.shape[1],
                "edge_size": first.edge_features.shape[1], "hidden_size": args.hidden, "blocks": args.blocks,
                "normalization": {name: value.detach().cpu().numpy() for name, value in normalization.items()},
                "train_files": [str(path) for path in train_files], "validation_files": [str(path) for path in validation_files], "test_files": [str(path) for path in test_files],
                "device": torch.cuda.get_device_name(0), "weights": WEIGHTS,
            }
            torch.save(state, args.output.with_name("pi_gnn_latest.pt"))
            if metrics["rmse_mps"] < best_rmse:
                best_rmse = metrics["rmse_mps"]
                torch.save(state, args.output)
    summary = {"best_validation_rmse_mps": best_rmse, "steps": args.steps, "elapsed_s": time.perf_counter() - started, "history": history}
    args.output.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
