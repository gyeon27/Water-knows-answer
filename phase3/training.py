"""CUDA/AMP training with trajectory-aware resume for Phase 3 models."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .config import Phase3Config
from .data import IndexedTFRecord, UnifiedParticleGraphDataset, deterministic_windows, graph_from_gns, graph_from_wcsph
from .models import UnifiedGNS


MODEL_OBJECTIVES = {
    "gnn_only": "all_residual",
    "reversed": "reversed",
    "ours": "ours",
    "baseline_gns": "gns",
}


def _append_csv(path: Path, row: dict[str, object], fieldnames: list[str]) -> None:
    """Durably append one graph-ready metric row, including across resumes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


TRAINING_FIELDS = [
    "model", "step", "loss_total", "loss_supervised", "loss_penetration",
    "loss_momentum", "loss_density", "loss_energy", "learning_rate",
    "steps_per_s", "gpu_memory_mb",
]
VALIDATION_FIELDS = [
    "model", "step", "normalized_acceleration_rmse", "trajectories", "values",
    "best_rmse", "gpu_memory_mb",
]


def _restore_training_state(
    state: dict,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    cfg: Phase3Config,
) -> None:
    """Restore a checkpoint while allowing a pre-registered longer horizon.

    A checkpoint made under the old 5k protocol has a cosine scheduler whose
    minimum learning rate was reached at step 5k.  Keeping that T_max would
    invalidate the 10k continuation, so only the scheduler horizon is updated;
    model, optimizer, scaler and the already consumed step remain unchanged.
    """
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    old_steps = int(state.get("config", {}).get("training_steps", cfg.training_steps))
    if old_steps != cfg.training_steps:
        scheduler.T_max = cfg.training_steps


def _early_stopping_state(history: list[dict], cfg: Phase3Config) -> tuple[float, int]:
    """Reconstruct the meaningful-improvement reference and patience counter."""
    values = []
    for row in history:
        value = row.get("normalized_acceleration_rmse", row.get("validation_rmse"))
        if value is not None:
            values.append(float(value))
    reference = float("inf")
    stale = 0
    for value in values:
        if value < reference - cfg.early_stopping_min_delta:
            reference, stale = value, 0
        else:
            stale += 1
    return reference, stale


def _tensor_graph(graph, device: torch.device) -> dict[str, torch.Tensor]:
    def transfer(array: np.ndarray, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        tensor = torch.from_numpy(array)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        if device.type == "cuda":
            tensor = tensor.pin_memory()
        return tensor.to(device, non_blocking=True)
    return {
        "node": transfer(graph.node_features),
        "types": transfer(graph.particle_type, dtype=torch.long),
        "edge": transfer(graph.edge_features),
        "edge_index": transfer(graph.edge_index, dtype=torch.long),
        "target": transfer(graph.target),
        "mask": transfer(graph.target_mask, dtype=torch.bool),
        "position": transfer(graph.positions),
        "velocity": transfer(graph.velocities),
        "routing": transfer(graph.routing_state, dtype=torch.long),
    }


def physics_informed_loss(
    prediction: torch.Tensor,
    batch: dict[str, torch.Tensor],
    metadata: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervised acceleration plus the four pre-registered PI penalties.

    Aggregates are deliberately evaluated in float32 even under AMP. One model
    step in GNS units is used consistently for position and velocity updates.
    """
    prediction = prediction.float()
    target = batch["target"].float()
    mask = batch["mask"]
    std = torch.as_tensor(metadata["acc_std"], device=prediction.device, dtype=torch.float32)
    mean = torch.as_tensor(metadata["acc_mean"], device=prediction.device, dtype=torch.float32)
    pred_acc = prediction * std + mean
    true_acc = target * std + mean
    predicted_v = batch["velocity"].float() + pred_acc
    teacher_v = batch["velocity"].float() + true_acc
    predicted_p = batch["position"].float() + predicted_v
    teacher_p = batch["position"].float() + teacher_v

    supervised = torch.mean((prediction[mask] - target[mask]).square())
    bounds = torch.as_tensor(metadata["bounds"], device=prediction.device, dtype=torch.float32)
    below = torch.relu(bounds[:, 0] - predicted_p)
    above = torch.relu(predicted_p - bounds[:, 1])
    penetration = torch.mean((below[mask] + above[mask]).square())
    momentum = torch.mean((pred_acc[mask].mean(0) - true_acc[mask].mean(0)).square())

    senders, receivers = batch["edge_index"]
    edge_mask = mask[senders] & mask[receivers]
    if torch.any(edge_mask):
        senders, receivers = senders[edge_mask], receivers[edge_mask]
        radius = float(metadata["default_connectivity_radius"])
        pred_distance = torch.linalg.vector_norm(predicted_p[senders] - predicted_p[receivers], dim=1) / radius
        true_distance = torch.linalg.vector_norm(teacher_p[senders] - teacher_p[receivers], dim=1) / radius
        density = torch.mean((torch.relu(1.0 - pred_distance) - torch.relu(1.0 - true_distance)).square())
    else:
        density = prediction.sum() * 0.0

    collision = mask & (batch["routing"] == 1)
    if not torch.any(collision):
        collision = mask
    gravity = 9.81
    pred_energy = 0.5 * predicted_v[collision].square().sum(1) + gravity * predicted_p[collision, 1]
    true_energy = 0.5 * teacher_v[collision].square().sum(1) + gravity * teacher_p[collision, 1]
    energy = torch.mean(torch.relu(pred_energy - true_energy).square())
    components = {
        "supervised": supervised,
        "penetration": penetration,
        "momentum": momentum,
        "density": density,
        "energy": energy,
    }
    total = supervised + 0.10 * penetration + 0.05 * momentum + 0.05 * density + 0.05 * energy
    return total, components


@torch.no_grad()
def validate(model: UnifiedGNS, dataset: IndexedTFRecord, cfg: Phase3Config, objective: str, device: torch.device) -> dict[str, float]:
    model.eval()
    squared = 0.0
    count = 0
    trajectories = min(len(dataset), cfg.valid_trajectories)
    for trajectory in range(trajectories):
        record = dataset.read(trajectory)
        frames = int(record["position"].shape[0])
        windows = deterministic_windows(frames, cfg.validation_windows, cfg.seed + 100_000 + trajectory)
        for frame in windows:
            batch = _tensor_graph(graph_from_gns(record, dataset.metadata, frame, cfg, objective), device)
            with torch.autocast("cuda", dtype=torch.float16):
                prediction = model(batch["node"], batch["types"], batch["edge"], batch["edge_index"])
            error = prediction.float()[batch["mask"]] - batch["target"][batch["mask"]]
            squared += float(torch.sum(error * error))
            count += error.numel()
    model.train()
    return {"normalized_acceleration_rmse": float(np.sqrt(squared / max(count, 1))), "trajectories": trajectories, "values": count}


def train_water3d(model_name: str, root: Path, cfg: Phase3Config, resume: bool = True) -> Path:
    if model_name not in MODEL_OBJECTIVES:
        raise ValueError(f"unknown model {model_name}; choose {sorted(MODEL_OBJECTIVES)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Phase 3 training")
    objective = MODEL_OBJECTIVES[model_name]
    raw = root / "raw" / cfg.dataset
    indices = root / "indices" / cfg.dataset
    train_data = IndexedTFRecord(raw / "train.tfrecord", indices / "train.npy", raw / "metadata.json")
    valid_data = IndexedTFRecord(raw / "valid.tfrecord", indices / "valid.npy", raw / "metadata.json")
    train_graphs = UnifiedParticleGraphDataset(train_data, cfg, objective, cache_trajectories=1)
    if len(train_data) != cfg.train_trajectories or len(valid_data) != cfg.valid_trajectories:
        raise ValueError("Water-3D split count does not match the fixed experiment config")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda")
    model = UnifiedGNS(hidden=cfg.hidden_size, blocks=cfg.message_blocks, type_embedding=cfg.particle_embedding).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training_steps, eta_min=cfg.learning_rate * 0.05)
    scaler = torch.amp.GradScaler("cuda")
    checkpoint_dir = root / "checkpoints" / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest = checkpoint_dir / "latest.pt"
    best = checkpoint_dir / "best.pt"
    step = 0
    best_rmse = float("inf")
    history: list[dict] = []
    order = list(range(cfg.train_trajectories))
    random.Random(cfg.seed).shuffle(order)
    if resume and latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        _restore_training_state(state, model, optimizer, scheduler, scaler, cfg)
        step = int(state["step"])
        best_rmse = float(state["best_rmse"])
        history = list(state.get("history", []))
    early_reference, stale_validations = _early_stopping_state(history, cfg)
    stopped_early = False
    resumed_step = step
    started = time.perf_counter()
    model.train()
    # Exactly 20 deterministic windows from every trajectory. Resume skips
    # completed global steps but preserves the fixed traversal order.
    for trajectory in order:
        record = train_graphs.trajectory(trajectory)
        windows = deterministic_windows(int(record["position"].shape[0]), cfg.windows_per_train_trajectory, cfg.seed + trajectory)
        for frame in windows:
            global_index = order.index(trajectory) * cfg.windows_per_train_trajectory + windows.index(frame) + 1
            if global_index <= step:
                continue
            graph = graph_from_gns(record, train_data.metadata, frame, cfg, objective)
            batch = _tensor_graph(graph, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                prediction = model(batch["node"], batch["types"], batch["edge"], batch["edge_index"])
                if model_name == "baseline_gns":
                    supervised = torch.mean((prediction[batch["mask"]] - batch["target"][batch["mask"]]) ** 2)
                    loss = supervised
                    zero = supervised.detach() * 0.0
                    loss_components = {"supervised": supervised, "penetration": zero, "momentum": zero, "density": zero, "energy": zero}
                else:
                    loss, loss_components = physics_informed_loss(prediction, batch, train_data.metadata)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {global_index}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step = global_index
            elapsed = max(time.perf_counter() - started, 1e-6)
            rate = (step - resumed_step) / elapsed
            _append_csv(checkpoint_dir / "training_loss.csv", {
                "model": model_name, "step": step, "loss_total": float(loss.detach()),
                **{f"loss_{key}": float(value.detach()) for key, value in loss_components.items()},
                "learning_rate": optimizer.param_groups[0]["lr"], "steps_per_s": rate,
                "gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
            }, TRAINING_FIELDS)
            do_validation = step == 1 or step % cfg.validate_every == 0 or step == cfg.training_steps
            metrics = None
            if do_validation:
                metrics = validate(model, valid_data, cfg, objective, device)
                record_log = {
                    "step": step, "loss": float(loss.detach()),
                    **{f"loss_{key}": float(value.detach()) for key, value in loss_components.items()},
                    **metrics, "gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
                }
                history.append(record_log)
                print(json.dumps(record_log), flush=True)
                validation_rmse = metrics["normalized_acceleration_rmse"]
                if validation_rmse < early_reference - cfg.early_stopping_min_delta:
                    early_reference, stale_validations = validation_rmse, 0
                else:
                    stale_validations += 1
                _append_csv(checkpoint_dir / "validation_metrics.csv", {
                    "model": model_name, "step": step, **metrics,
                    "best_rmse": min(best_rmse, validation_rmse),
                    "gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
                }, VALIDATION_FIELDS)
            elif step % 25 == 0:
                print(json.dumps({
                    "model": model_name, "step": step, "loss": float(loss.detach()),
                    "steps_per_s": rate, "eta_hours": (cfg.training_steps - step) / max(rate, 1e-9) / 3600,
                    "gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
                }), flush=True)
            if step % cfg.checkpoint_every == 0 or do_validation:
                improved = bool(metrics and metrics["normalized_acceleration_rmse"] < best_rmse)
                if improved:
                    best_rmse = metrics["normalized_acceleration_rmse"]
                state = {
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "step": step, "best_rmse": best_rmse, "model_name": model_name, "objective": objective,
                    "trajectory": trajectory, "frame": frame,
                    "config": cfg.__dict__, "device": torch.cuda.get_device_name(0), "history": history,
                    "early_reference": early_reference, "stale_validations": stale_validations,
                }
                torch.save(state, latest)
                if improved:
                    torch.save(state, best)
            if do_validation and stale_validations >= cfg.early_stopping_patience_validations:
                stopped_early = True
                print(json.dumps({"model": model_name, "early_stopped_at": step,
                                  "best_rmse": best_rmse,
                                  "patience_validations": stale_validations}), flush=True)
                break
            if step >= cfg.training_steps:
                break
        if step >= cfg.training_steps or stopped_early:
            break
    summary = {"model": model_name, "steps": step, "best_rmse": best_rmse,
               "stopped_early": stopped_early, "elapsed_s": time.perf_counter() - started,
               "history": history}
    (checkpoint_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return best


def train_wcsph_zero_shot(root: Path, repository: Path, cfg: Phase3Config, resume: bool = True) -> Path:
    """Train the common-feature PI-GNN on WCSPH for honest Water-3D zero-shot."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    files = sorted((repository / "phase2" / "datasets" / "wcsph_pi").glob("*.npz"))
    if len(files) < 12:
        raise FileNotFoundError("the 12 Phase-2 WCSPH trajectories are required")
    train_files, valid_files = files[:8], files[8:10]
    device = torch.device("cuda")
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
    model = UnifiedGNS(hidden=cfg.hidden_size, blocks=cfg.message_blocks, type_embedding=cfg.particle_embedding).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training_steps, eta_min=cfg.learning_rate * 0.05)
    scaler = torch.amp.GradScaler("cuda")
    checkpoint_dir = root / "checkpoints" / "wcsph_zero_shot"; checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest, best = checkpoint_dir / "latest.pt", checkpoint_dir / "best.pt"
    step, best_rmse, history = 0, float("inf"), []
    if resume and latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        _restore_training_state(state, model, optimizer, scheduler, scaler, cfg)
        step, best_rmse, history = int(state["step"]), float(state["best_rmse"]), list(state.get("history", []))
    early_reference, stale_validations = _early_stopping_state(history, cfg)
    stopped_early = False
    resumed_step = step
    started = time.perf_counter()
    terrain_root = repository / "phase2" / "terrains"
    model.train()
    while step < cfg.training_steps:
        path = train_files[step % len(train_files)]
        with np.load(path, allow_pickle=False) as data:
            frames = int(data["positions"].shape[0]); active = data["active_mask"]
            eligible = [f for f in range(5, frames - 1) if np.any(active[f] & active[f + 1])]
        choices = eligible if len(eligible) <= 100 else sorted(random.Random(cfg.seed + step // len(train_files)).sample(eligible, 100))
        frame = choices[(step // len(train_files)) % len(choices)]
        graph = graph_from_wcsph(path, terrain_root, frame, cfg)
        batch = _tensor_graph(graph, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            prediction = model(batch["node"], batch["types"], batch["edge"], batch["edge_index"])
            loss = torch.mean((prediction[batch["mask"]] - batch["target"][batch["mask"]]) ** 2)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
        scaler.step(optimizer); scaler.update(); scheduler.step(); step += 1
        elapsed = max(time.perf_counter() - started, 1e-6)
        rate = (step - resumed_step) / elapsed
        _append_csv(checkpoint_dir / "training_loss.csv", {
            "model": "wcsph_zero_shot", "step": step, "loss_total": float(loss.detach()),
            "loss_supervised": float(loss.detach()), "loss_penetration": 0.0,
            "loss_momentum": 0.0, "loss_density": 0.0, "loss_energy": 0.0,
            "learning_rate": optimizer.param_groups[0]["lr"], "steps_per_s": rate,
            "gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
        }, TRAINING_FIELDS)
        if step % 25 == 0:
            print(json.dumps({
                "model": "wcsph_zero_shot", "step": step, "loss": float(loss.detach()),
                "steps_per_s": rate, "eta_hours": (cfg.training_steps - step) / max(rate, 1e-9) / 3600,
                "gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
            }), flush=True)
        if step == 1 or step % cfg.validate_every == 0 or step == cfg.training_steps:
            model.eval(); squared = 0.0; count = 0
            with torch.no_grad():
                for file_index, validation_path in enumerate(valid_files):
                    with np.load(validation_path, allow_pickle=False) as data:
                        frames = int(data["positions"].shape[0]); active = data["active_mask"]
                        eligible = [f for f in range(5, frames - 1) if np.any(active[f] & active[f + 1])]
                    selected = eligible if len(eligible) <= cfg.validation_windows else random.Random(cfg.seed + 50_000 + file_index).sample(eligible, cfg.validation_windows)
                    for validation_frame in sorted(selected):
                        value = _tensor_graph(graph_from_wcsph(validation_path, terrain_root, validation_frame, cfg), device)
                        with torch.autocast("cuda", dtype=torch.float16):
                            pred = model(value["node"], value["types"], value["edge"], value["edge_index"])
                        error = pred.float()[value["mask"]] - value["target"][value["mask"]]
                        squared += float(torch.sum(error * error)); count += error.numel()
            model.train(); rmse = float(np.sqrt(squared / max(count, 1)))
            history.append({"step": step, "loss": float(loss.detach()), "validation_rmse": rmse})
            print(json.dumps(history[-1]), flush=True)
            _append_csv(checkpoint_dir / "validation_metrics.csv", {
                "model": "wcsph_zero_shot", "step": step,
                "normalized_acceleration_rmse": rmse, "trajectories": len(valid_files),
                "values": count, "best_rmse": min(best_rmse, rmse),
                "gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
            }, VALIDATION_FIELDS)
            if rmse < early_reference - cfg.early_stopping_min_delta:
                early_reference, stale_validations = rmse, 0
            else:
                stale_validations += 1
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "step": step, "best_rmse": min(best_rmse, rmse), "model_name": "wcsph_zero_shot", "config": cfg.__dict__, "history": history, "early_reference": early_reference, "stale_validations": stale_validations}
            torch.save(state, latest)
            if rmse < best_rmse:
                best_rmse = rmse; torch.save(state, best)
            if stale_validations >= cfg.early_stopping_patience_validations:
                stopped_early = True
                print(json.dumps({"model": "wcsph_zero_shot", "early_stopped_at": step,
                                  "best_rmse": best_rmse,
                                  "patience_validations": stale_validations}), flush=True)
                break
    (checkpoint_dir / "summary.json").write_text(json.dumps({"steps": step, "best_rmse": best_rmse, "stopped_early": stopped_early, "history": history}, indent=2), encoding="utf-8")
    return best
