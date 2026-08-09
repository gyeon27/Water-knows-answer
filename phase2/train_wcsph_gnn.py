"""Train the first residual GNS baseline on WCSPH trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from gnn import ResidualGNS, TrajectoryGraphDataset


def statistics(samples, attribute: str) -> tuple[np.ndarray, np.ndarray]:
    values = [getattr(sample, attribute) for sample in samples if getattr(sample, attribute).size]
    joined = np.concatenate(values, axis=0)
    mean = joined.mean(axis=0).astype(np.float32)
    std = joined.std(axis=0).astype(np.float32)
    return mean, np.maximum(std, 1e-5)


def temporal_split(dataset: TrajectoryGraphDataset, train_fraction: float = 0.78, gap: int = 6):
    frame_limits: dict[int, int] = {}
    for file_index, frame in dataset.index:
        frame_limits[file_index] = max(frame_limits.get(file_index, 0), frame)
    cuts = {key: int(value * train_fraction) for key, value in frame_limits.items()}
    train, validation = [], []
    for index, (file_index, frame) in enumerate(dataset.index):
        if frame <= cuts[file_index]:
            train.append(index)
        elif frame >= cuts[file_index] + gap:
            validation.append(index)
    return train, validation


def tensors(sample, norm, device: str):
    batch = TrajectoryGraphDataset.to_torch(sample, device)
    node_mean, node_std, edge_mean, edge_std, target_mean, target_std = norm
    batch["node_features"] = (batch["node_features"] - node_mean) / node_std
    if batch["edge_features"].numel():
        batch["edge_features"] = (batch["edge_features"] - edge_mean) / edge_std
    batch["target_normalized"] = (batch["target_delta_v"] - target_mean) / target_std
    return batch


@torch.no_grad()
def evaluate(model, samples, norm, device: str) -> dict[str, float]:
    model.eval()
    squared, count = 0.0, 0
    target_mean, target_std = norm[4], norm[5]
    for sample in samples:
        batch = tensors(sample, norm, device)
        prediction_n = model(batch["node_features"], batch["edge_features"], batch["edge_index"])
        mask = batch["splash_mask"]
        prediction = prediction_n * target_std + target_mean
        error = prediction[mask] - batch["target_delta_v"][mask]
        squared += float(torch.sum(error * error))
        count += int(error.numel())
    model.train()
    return {"rmse_mps": (squared / max(count, 1)) ** 0.5, "components": count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--data", type=Path, default=root / "datasets" / "wcsph")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--validation-every", type=int, default=50)
    parser.add_argument("--output", type=Path, default=root / "checkpoints" / "wcsph_gns_baseline.pt")
    args = parser.parse_args()

    torch.manual_seed(20260809)
    np.random.seed(20260809)
    random.seed(20260809)
    files = sorted(args.data.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no NPZ trajectories in {args.data}")
    dataset = TrajectoryGraphDataset(files, root / "terrains")
    train_indices, validation_indices = temporal_split(dataset)
    if not train_indices or not validation_indices:
        raise ValueError("not enough graph frames for temporal train/validation split")
    print(f"loading graphs: train={len(train_indices)} validation={len(validation_indices)}")
    train_samples = [dataset[index] for index in train_indices]
    validation_samples = [dataset[index] for index in validation_indices]

    node_mean, node_std = statistics(train_samples, "node_features")
    edge_mean, edge_std = statistics(train_samples, "edge_features")
    target_mean, target_std = statistics(train_samples, "target_delta_v")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    norm_np = (node_mean, node_std, edge_mean, edge_std, target_mean, target_std)
    norm = tuple(torch.as_tensor(value, device=device) for value in norm_np)
    first = train_samples[0]
    model = ResidualGNS(first.node_features.shape[1], first.edge_features.shape[1], args.hidden, args.blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.learning_rate * 0.1)
    best_rmse = float("inf")
    history = []
    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        sample = train_samples[random.randrange(len(train_samples))]
        batch = tensors(sample, norm, device)
        prediction = model(batch["node_features"], batch["edge_features"], batch["edge_index"])
        mask = batch["splash_mask"]
        delta_loss = F.mse_loss(prediction[mask], batch["target_normalized"][mask])
        physical_delta = prediction[mask] * norm[5] + norm[4]
        momentum_loss = torch.mean(torch.sum(physical_delta, dim=0) ** 2) / max(int(mask.sum()), 1)
        loss = delta_loss + 0.01 * momentum_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % args.validation_every == 0 or step == args.steps:
            metrics = evaluate(model, validation_samples, norm, device)
            entry = {"step": step, "train_loss": float(loss.detach()), **metrics}
            history.append(entry)
            print(json.dumps(entry, ensure_ascii=False))
            if metrics["rmse_mps"] < best_rmse:
                best_rmse = metrics["rmse_mps"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "node_size": first.node_features.shape[1],
                        "edge_size": first.edge_features.shape[1],
                        "hidden_size": args.hidden,
                        "blocks": args.blocks,
                        "normalization": {"node_mean": node_mean, "node_std": node_std, "edge_mean": edge_mean, "edge_std": edge_std, "target_mean": target_mean, "target_std": target_std},
                        "files": [str(path) for path in files],
                        "best_validation_rmse_mps": best_rmse,
                    },
                    args.output,
                )

    summary = {
        "device": device,
        "files": [path.name for path in files],
        "train_graphs": len(train_samples),
        "validation_graphs": len(validation_samples),
        "steps": args.steps,
        "best_validation_rmse_mps": best_rmse,
        "elapsed_s": time.perf_counter() - started,
        "history": history,
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
