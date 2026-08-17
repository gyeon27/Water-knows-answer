"""Overfit the small residual GCN on one debug trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from phase2.gnn import ResidualGNS, TrajectoryGraphDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--cycle-graphs", action="store_true", help="cycle through every graph instead of overfitting graph 0")
    phase2_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--trajectory", type=Path, default=phase2_root / "datasets" / "debug" / "trajectory_000_single_cliff.npz")
    parser.add_argument("--output", type=Path, default=phase2_root / "checkpoints" / "debug_gnn.pt")
    args = parser.parse_args()
    torch.manual_seed(20260809)
    random.seed(20260809)
    root = phase2_root
    dataset = TrajectoryGraphDataset([args.trajectory], root / "terrains")
    if len(dataset) == 0:
        raise ValueError("trajectory contains no trainable SPLASH graph")
    first = dataset[0]
    model = ResidualGNS(first.node_features.shape[1], first.edge_features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    initial_loss = None
    final_loss = None
    model.train()
    for step in range(args.steps):
        sample = dataset[step % len(dataset)] if args.cycle_graphs else first
        batch = dataset.to_torch(sample)
        prediction = model(batch["node_features"], batch["edge_features"], batch["edge_index"])
        mask = batch["splash_mask"]
        delta_loss = F.mse_loss(prediction[mask], batch["target_delta_v"][mask])
        momentum_loss = torch.mean(torch.sum(prediction[mask], dim=0) ** 2) / max(int(mask.sum()), 1)
        loss = delta_loss + 0.05 * momentum_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        value = float(loss.detach())
        initial_loss = value if initial_loss is None else initial_loss
        final_loss = value
        if step % 25 == 0 or step == args.steps - 1:
            print(f"step={step:04d} loss={value:.8f} nodes={mask.numel()} roi={int(mask.sum())} edges={batch['edge_index'].shape[1]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "node_size": first.node_features.shape[1], "edge_size": first.edge_features.shape[1]}, args.output)
    summary = {
        "steps": args.steps,
        "available_graphs": len(dataset),
        "training_mode": "cycle" if args.cycle_graphs else "fixed_graph_overfit",
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_ratio": final_loss / max(initial_loss, 1e-12),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
