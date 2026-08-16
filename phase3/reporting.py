"""Generate honest Phase 3 tables, plots, and discussion from raw artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .config import Phase3Config


def _benchmark_summary(path: Path, cfg: Phase3Config) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    groups = {}
    for row in rows:
        key = (row["condition"], int(row["particle_count"]), float(row["splash_fraction"]))
        groups.setdefault(key, []).append(float(row["total_ms"]))
    result = []
    for (condition, count, fraction), values in sorted(groups.items()):
        data = np.asarray(values)
        mean = float(data.mean())
        result.append({
            "condition": condition, "particle_count": count, "splash_fraction": fraction, "mean_ms": mean,
            "std_ms": float(data.std()), "p95_ms": float(np.quantile(data, 0.95)), "max_ms": float(data.max()),
            "fps": 1000.0 / mean, "targets": {str(fps): mean <= 1000.0 / fps for fps in cfg.target_fps},
        })
    return result


def _ablation_summary(path: Path, seed: int = 20260809) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    groups = {}
    metrics = ("position_rmse", "velocity_rmse", "penetration_rate", "density_relative_error", "momentum_error_per_particle", "horizontal_momentum_error_per_particle", "surface_height_rmse", "energy_excess_per_particle")
    for row in rows:
        key = (row["condition"], row["condition_name"], row["scene_group"], int(row["horizon"]))
        groups.setdefault(key, []).append(row)
    output = []
    rng = np.random.default_rng(seed)
    for key, values in sorted(groups.items()):
        row = {"condition": key[0], "condition_name": key[1], "scene_group": key[2], "horizon": key[3], "trajectory_count": len(values)}
        for metric in metrics:
            data = np.asarray([v.get(metric, np.nan) for v in values], np.float64)
            data = data[np.isfinite(data)]
            row[metric] = float(np.mean(data)) if data.size else float("nan")
            if data.size:
                boot = np.mean(rng.choice(data, (2000, data.size), replace=True), axis=1)
                row[f"{metric}_ci95_low"], row[f"{metric}_ci95_high"] = map(float, np.quantile(boot, (0.025, 0.975)))
            else:
                row[f"{metric}_ci95_low"] = row[f"{metric}_ci95_high"] = float("nan")
        output.append(row)
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _loss_artifacts(root: Path, report_dir: Path) -> None:
    rows = []
    for name in ("wcsph_zero_shot", "gnn_only", "reversed", "ours", "baseline_gns"):
        csv_path = root / "checkpoints" / name / "validation_metrics.csv"
        if csv_path.exists():
            with csv_path.open(newline="", encoding="utf-8") as stream:
                rows.extend(dict(row) for row in csv.DictReader(stream))
            continue
        summary_path = root / "checkpoints" / name / "summary.json"
        if summary_path.exists():
            history = json.loads(summary_path.read_text(encoding="utf-8")).get("history", [])
            for item in history:
                value = item.get("normalized_acceleration_rmse", item.get("validation_rmse"))
                if value is not None:
                    rows.append({"model": name, "step": item["step"], "normalized_acceleration_rmse": value})
    if not rows:
        return
    fields = ["model", "step", "normalized_acceleration_rmse"]
    with (report_dir / "validation_loss_curves.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for model in sorted({r["model"] for r in rows}):
        values = sorted((r for r in rows if r["model"] == model), key=lambda r: int(r["step"]))
        plt.plot([int(r["step"]) for r in values], [float(r["normalized_acceleration_rmse"]) for r in values], marker="o", label=model)
    plt.xlabel("Training step"); plt.ylabel("Validation normalized acceleration RMSE")
    plt.grid(alpha=0.25); plt.legend(); plt.tight_layout()
    plt.savefig(report_dir / "validation_loss_curves.png", dpi=180); plt.close()


def _plots(benchmark: list[dict], ablation: list[dict], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    output.mkdir(parents=True, exist_ok=True)
    for condition in sorted({r["condition"] for r in benchmark}):
        values = [r for r in benchmark if r["condition"] == condition and r["splash_fraction"] == 0.50]
        plt.plot([r["particle_count"] for r in values], [r["mean_ms"] for r in values], marker="o", label=condition)
    plt.xlabel("Particles"); plt.ylabel("Frame time (ms)"); plt.legend(); plt.tight_layout()
    plt.savefig(output / "particle_count_vs_frame_time.png", dpi=180); plt.close()
    for condition in sorted({r["condition"] for r in benchmark}):
        values = [r for r in benchmark if r["condition"] == condition and r["particle_count"] == 20_000]
        plt.plot([100 * r["splash_fraction"] for r in values], [r["mean_ms"] for r in values], marker="o", label=condition)
    plt.xlabel("SPLASH ratio (%)"); plt.ylabel("Frame time (ms)"); plt.legend(); plt.tight_layout()
    plt.savefig(output / "splash_ratio_vs_frame_time.png", dpi=180); plt.close()
    final = [r for r in ablation if r["horizon"] in (1, 8, 16, 32, 100) and r["scene_group"] == "complex"]
    for condition in sorted({r["condition"] for r in final}):
        values = [r for r in final if r["condition"] == condition]
        plt.plot([r["horizon"] for r in values], [r["position_rmse"] for r in values], marker="o", label=condition)
    plt.xlabel("Rollout horizon"); plt.ylabel("Position RMSE (dataset coordinate)"); plt.legend(); plt.tight_layout()
    plt.savefig(output / "rollout_error.png", dpi=180); plt.close()


def generate_report(root: Path, cfg: Phase3Config) -> Path:
    benchmark_path = root / "benchmark" / "benchmark_results.csv"
    ablation_path = root / "rollouts" / "ablation_results.json"
    if not benchmark_path.exists() or not ablation_path.exists():
        raise FileNotFoundError("benchmark and ablation raw results are required")
    benchmark = _benchmark_summary(benchmark_path, cfg)
    ablation = _ablation_summary(ablation_path)
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "benchmark_summary.json").write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    (report_dir / "ablation_summary.json").write_text(json.dumps(ablation, indent=2), encoding="utf-8")
    _write_csv(report_dir / "benchmark_summary.csv", benchmark)
    _write_csv(report_dir / "ablation_summary.csv", ablation)
    _loss_artifacts(root, report_dir)
    _plots(benchmark, ablation, report_dir)
    complex32 = {r["condition"]: r for r in ablation if r["scene_group"] == "complex" and r["horizon"] == 32}
    novelty_speed = False
    novelty_accuracy = False
    # Runtime is reported per benchmark setting, while accuracy is read from
    # the fixed complex-scene rollout. Never invent a positive conclusion.
    if "D" in complex32 and "B" in complex32:
        novelty_accuracy = complex32["D"]["position_rmse"] <= complex32["B"]["position_rmse"] * 1.10
    comparable = [(r["particle_count"], r["splash_fraction"]) for r in benchmark if r["condition"] == "D"]
    speed_checks = []
    for count, fraction in comparable:
        d = next((r for r in benchmark if r["condition"] == "D" and r["particle_count"] == count and r["splash_fraction"] == fraction), None)
        b = next((r for r in benchmark if r["condition"] == "B" and r["particle_count"] == count and r["splash_fraction"] == fraction), None)
        if d and b:
            speed_checks.append(d["mean_ms"] < b["mean_ms"])
    novelty_speed = bool(speed_checks) and float(np.mean(speed_checks)) >= 0.75
    conclusion = "부분적으로 입증됨" if novelty_speed and novelty_accuracy else "현재 실험에서는 입증되지 않음"
    lines = [
        "# Phase 3 결과", "", f"- Water-3D split: {cfg.train_trajectories}/{cfg.valid_trajectories}/{cfg.test_trajectories}",
        f"- Novelty 결론: **{conclusion}**", "", "## 32-step 복합 장면 Ablation", "",
        "|조건|위치 RMSE|속도 RMSE|침투율|밀도 상대오차|", "|---|---:|---:|---:|---:|",
    ]
    for condition in "ABCDEFG":
        if condition in complex32:
            r = complex32[condition]
            lines.append(f"|{condition} {r['condition_name']}|{r['position_rmse']:.6g}|{r['velocity_rmse']:.6g}|{r['penetration_rate']:.3%}|{r['density_relative_error']:.3%}|")
    lines += ["", "## 해석", "", "결론은 사전에 고정한 조건과 전체 test split의 집계값에서 생성되며, 우위가 나오도록 사후 임계값을 조정하지 않았다."]
    report = report_dir / "phase3_results.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    discussion = report_dir / "phase3_discussion_gnn_justification.md"
    discussion.write_text(
        "# 왜 GNN인가\n\nGNN의 목적은 속도만이 아니라 국소 충돌·분열 상호작용을 학습 가능한 이웃 그래프로 근사하는 것이다. "
        f"본 실험의 사전 정의된 novelty 판정은 **{conclusion}**이다. 자세한 근거는 phase3_results.md의 A–F 비교를 따른다.\n",
        encoding="utf-8",
    )
    return report
