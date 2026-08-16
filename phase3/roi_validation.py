"""Measure the accuracy cost of constructing only the routed SPLASH graph."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .config import Phase3Config
from .data import IndexedTFRecord, deterministic_windows, graph_from_gns, radius_graph, KINEMATIC_ID
from .evaluation import _load_model


@torch.inference_mode()
def validate_roi(root: Path, cfg: Phase3Config, trajectories: int = 100) -> Path:
    device = torch.device("cuda")
    raw = root / "raw" / cfg.dataset
    data = IndexedTFRecord(raw / "test.tfrecord", root / "indices" / cfg.dataset / "test.npy", raw / "metadata.json")
    model = _load_model(root, "ours", cfg, device)
    full_sq = roi_sq = difference_sq = 0.0
    values = 0
    rows = []
    for trajectory in range(min(trajectories, len(data))):
        source = data.read(trajectory)
        frame = deterministic_windows(int(source["position"].shape[0]), 1, cfg.seed + 1_100_000 + trajectory)[0]
        graph = graph_from_gns(source, data.metadata, frame, cfg, "ours")
        # Do not use target_mask here: training deliberately falls back to all
        # fluid nodes when a frame has no SPLASH so that a batch is non-empty.
        # Runtime ROI must instead skip GNN work on such a frame.
        selected = (graph.particle_type != KINEMATIC_ID) & (graph.routing_state == 1)
        ids = np.flatnonzero(selected)
        if not ids.size:
            continue
        node = torch.from_numpy(graph.node_features).to(device)
        types = torch.from_numpy(graph.particle_type).long().to(device)
        edge = torch.from_numpy(graph.edge_features).to(device)
        edge_index = torch.from_numpy(graph.edge_index).long().to(device)
        ids_device = torch.from_numpy(ids).long().to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            full = model(node, types, edge, edge_index)[ids_device].float().cpu().numpy()
        roi_index, roi_edge = radius_graph(graph.positions[ids], float(data.metadata["default_connectivity_radius"]), cfg.max_neighbors)
        with torch.autocast("cuda", dtype=torch.float16):
            roi = model(
                torch.from_numpy(graph.node_features[ids]).to(device),
                torch.from_numpy(graph.particle_type[ids]).long().to(device),
                torch.from_numpy(roi_edge).to(device),
                torch.from_numpy(roi_index).long().to(device),
            ).float().cpu().numpy()
        target = graph.target[ids]
        full_error = full - target; roi_error = roi - target
        full_sq += float(np.sum(full_error * full_error))
        roi_sq += float(np.sum(roi_error * roi_error))
        difference_sq += float(np.sum((roi - full) ** 2))
        values += int(target.size)
        rows.append({"trajectory": trajectory, "frame": frame, "all_nodes": int(len(selected)),
                     "roi_nodes": int(len(ids)), "splash_fraction": float(np.mean(selected)),
                     "full_rmse": float(np.sqrt(np.mean(full_error ** 2))),
                     "roi_rmse": float(np.sqrt(np.mean(roi_error ** 2)))})
    result = {
        "trajectories": len(rows), "values": values,
        "full_graph_normalized_acceleration_rmse": float(np.sqrt(full_sq / max(values, 1))),
        "roi_graph_normalized_acceleration_rmse": float(np.sqrt(roi_sq / max(values, 1))),
        "roi_vs_full_prediction_rmse": float(np.sqrt(difference_sq / max(values, 1))),
        "mean_splash_fraction": float(np.mean([r["splash_fraction"] for r in rows])) if rows else 0.0,
        "rows": rows,
    }
    output = root / "reports" / "roi_accuracy_validation.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    import sys
    print(validate_roi(Path(sys.argv[1]), Phase3Config()))
