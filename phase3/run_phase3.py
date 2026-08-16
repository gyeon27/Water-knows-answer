"""Single entry point for the complete Phase 3 experiment."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from .benchmark import run_benchmark
from .config import Phase3Config, ensure_layout, resolve_data_root
from .data import prepare_water3d, radius_graph
from .evaluation import evaluate_all, evaluate_zero_shot
from .models import UnifiedGNS
from .reporting import generate_report
from .training import MODEL_OBJECTIVES, train_water3d, train_wcsph_zero_shot


def _git_commit(repository: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def write_run_manifest(root: Path, repository: Path, cfg: Phase3Config) -> None:
    value = {
        "created_unix_s": time.time(), "git_commit": _git_commit(repository), "python": sys.version,
        "numpy": np.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "config": cfg.__dict__,
    }
    (root / "run_manifest.json").write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    cfg.write(root)


def doctor(root: Path, cfg: Phase3Config) -> dict[str, object]:
    free = shutil.disk_usage(root).free / 2**30
    if free < 300:
        raise RuntimeError(f"data root has only {free:.1f} GB free; at least 300 GB is required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    # Forward/backward smoke test with a real radius graph.
    rng = np.random.default_rng(cfg.seed)
    position = rng.uniform(0, 1, (256, 3)).astype(np.float32)
    edge_index, edge = radius_graph(position, 0.20, cfg.max_neighbors)
    model = UnifiedGNS(hidden=32, blocks=2).cuda()
    node = torch.zeros((256, 27), device="cuda", requires_grad=True)
    types = torch.zeros(256, dtype=torch.long, device="cuda")
    edge_t = torch.from_numpy(edge).cuda(); index_t = torch.from_numpy(edge_index).long().cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(node, types, edge_t, index_t).square().mean()
    loss.backward()
    result = {"data_root": str(root), "free_gb": free, "gpu": torch.cuda.get_device_name(0), "edges": int(edge_index.shape[1]), "smoke_loss": float(loss.detach())}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def train_all(root: Path, repository: Path, cfg: Phase3Config, selected: list[str] | None = None) -> None:
    names = selected or ["wcsph_zero_shot", *MODEL_OBJECTIVES]
    for name in names:
        print(f"=== TRAIN {name} ===", flush=True)
        if name == "wcsph_zero_shot":
            train_wcsph_zero_shot(root, repository, cfg, resume=True)
        else:
            train_water3d(name, root, cfg, resume=True)


def publish_reports(root: Path, repository: Path) -> None:
    destination = repository / "phase3" / "results_summary"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("phase3_results.md", "phase3_discussion_gnn_justification.md", "benchmark_summary.json", "benchmark_summary.csv", "ablation_summary.json", "ablation_summary.csv", "validation_loss_curves.csv", "validation_loss_curves.png", "particle_count_vs_frame_time.png", "splash_ratio_vs_frame_time.png", "rollout_error.png"):
        source = root / "reports" / name
        if source.exists():
            shutil.copy2(source, destination / name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("doctor", "prepare", "train", "evaluate", "benchmark", "report", "all"))
    parser.add_argument("--data-root", default="auto")
    parser.add_argument("--models", nargs="+", choices=("wcsph_zero_shot", *MODEL_OBJECTIVES))
    parser.add_argument("--smoke", action="store_true", help="short benchmark only; does not change fixed full-training protocol")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = Path(__file__).resolve().parents[1]
    root = resolve_data_root(args.data_root)
    ensure_layout(root)
    cfg = Phase3Config()
    write_run_manifest(root, repository, cfg)
    if args.command == "all":
        doctor(root, cfg)
        # Preserve the experiment boundary: no GPU training starts until all
        # three splits have completed download, SHA-256, CRC32C, and indexing.
        prepare_water3d(root, cfg)
        train_all(root, repository, cfg, args.models)
        evaluate_zero_shot(root, cfg)
        evaluate_all(root, cfg)
        run_benchmark(root, cfg, smoke=args.smoke)
        generate_report(root, cfg)
        publish_reports(root, repository)
        return
    if args.command in ("doctor", "all"):
        doctor(root, cfg)
    if args.command in ("prepare", "all"):
        prepare_water3d(root, cfg)
    if args.command in ("train", "all"):
        train_all(root, repository, cfg, args.models)
    if args.command in ("evaluate", "all"):
        evaluate_zero_shot(root, cfg)
        evaluate_all(root, cfg)
    if args.command in ("benchmark", "all"):
        run_benchmark(root, cfg, smoke=args.smoke)
    if args.command in ("report", "all"):
        generate_report(root, cfg)
        publish_reports(root, repository)


if __name__ == "__main__":
    main()
