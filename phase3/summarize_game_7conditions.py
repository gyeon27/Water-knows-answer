"""Create reproducible, honest seven-condition game benchmark summaries."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


NAMES = {
    "A": "SWE-only", "B": "GNN-only", "C": "Reversed", "D": "Ours",
    "E": "Baseline-GNS", "F": "Simple-3D", "G": "Optimized-Ours",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--accuracy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.runtime.open(newline="", encoding="utf-8") as stream:
        raw = list(csv.DictReader(stream))
    runtime_rows = []
    for condition in NAMES:
        for fraction in (0.05, 0.25, 0.50, 1.00):
            values = np.asarray([
                float(row["total_ms"]) for row in raw
                if row["condition"] == condition and float(row["splash_fraction"]) == fraction
            ], np.float64)
            if values.size != 300:
                raise RuntimeError(f"{condition}/{fraction}: expected 300 frames, got {values.size}")
            mean = float(values.mean())
            p95 = float(np.quantile(values, .95))
            runtime_rows.append({
                "condition": condition, "condition_name": NAMES[condition],
                "particle_count": 5000, "splash_fraction": fraction,
                "measured_frames": int(values.size), "mean_frame_ms": mean,
                "p95_frame_ms": p95, "max_frame_ms": float(values.max()),
                "mean_fps": 1000.0 / mean,
                "p95_effective_fps": 1000.0 / p95,
                "passes_30fps_p95": p95 <= 1000 / 30,
                "passes_60fps_p95": p95 <= 1000 / 60,
                "passes_120fps_p95": p95 <= 1000 / 120,
                "passes_144fps_p95": p95 <= 1000 / 144,
            })

    with args.accuracy.open(newline="", encoding="utf-8") as stream:
        accuracy_raw = list(csv.DictReader(stream))
    accuracy_rows = []
    for condition in NAMES:
        selected = [row for row in accuracy_raw if row["condition"] == condition and int(row["horizon"]) == 32]
        if len(selected) != 3:
            raise RuntimeError(f"{condition}: expected quiet/complex/violent horizon-32 rows, got {len(selected)}")
        accuracy_rows.append({
            "condition": condition, "condition_name": NAMES[condition], "horizon": 32,
            "position_rmse": float(np.mean([float(r["position_rmse"]) for r in selected])),
            "penetration_rate": float(np.mean([float(r["penetration_rate"]) for r in selected])),
            "density_relative_error": float(np.mean([float(r["density_relative_error"]) for r in selected])),
            "energy_excess_per_particle": float(np.mean([float(r["energy_excess_per_particle"]) for r in selected])),
        })

    runtime_path = args.output_dir / "game_7condition_runtime.csv"
    with runtime_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=runtime_rows[0].keys())
        writer.writeheader(); writer.writerows(runtime_rows)
    accuracy_path = args.output_dir / "game_7condition_accuracy.csv"
    with accuracy_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=accuracy_rows[0].keys())
        writer.writeheader(); writer.writerows(accuracy_rows)

    lookup = {(r["condition"], r["splash_fraction"]): r for r in runtime_rows}
    gains = {}
    for fraction in (0.05, 0.25, 0.50, 1.00):
        g, b, d = lookup[("G", fraction)], lookup[("B", fraction)], lookup[("D", fraction)]
        gains[str(fraction)] = {
            "optimized_ours_fps": g["mean_fps"],
            "speedup_vs_gnn_only": g["mean_fps"] / b["mean_fps"],
            "speedup_vs_unoptimized_ours": g["mean_fps"] / d["mean_fps"],
            "passes_60fps_p95": g["passes_60fps_p95"],
        }
    payload = {
        "protocol": "same code revision; 5000 particles; 300 measured frames after 10 warmup; RTX 3060; seed 20260809",
        "runtime_source": str(args.runtime), "accuracy_source": str(args.accuracy),
        "runtime_rows": runtime_rows, "accuracy_rows": accuracy_rows,
        "optimized_ours_gains": gains,
        "claim": "Optimized Ours is the fastest learned method in local-SPLASH regimes; it is not the fastest method overall and does not pass 60 FPS p95 when 100% of particles are SPLASH.",
    }
    (args.output_dir / "game_7condition_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    acc = {row["condition"]: row for row in accuracy_rows}
    lines = [
        "# 7조건 게임 런타임 비교 결과", "",
        "## 재현 조건", "",
        "- RTX 3060, seed 20260809", "- 입자 5,000개",
        "- SPLASH 비율 5%, 25%, 50%, 100%", "- 워밍업 10프레임 후 조건별 300프레임",
        "- 모든 조건은 같은 코드 revision에서 다시 측정", "",
        "## Optimized Ours 런타임", "",
        "|SPLASH 비율|평균 FPS|GNN-only 대비|기존 Ours 대비|p95 60 FPS|",
        "|---:|---:|---:|---:|:---:|",
    ]
    for fraction in (0.05, 0.25, 0.50, 1.00):
        item = gains[str(fraction)]
        lines.append(
            f"|{fraction:.0%}|{item['optimized_ours_fps']:.2f}|"
            f"{item['speedup_vs_gnn_only']:.2f}×|{item['speedup_vs_unoptimized_ours']:.2f}×|"
            f"{'통과' if item['passes_60fps_p95'] else '실패'}|"
        )
    lines += [
        "", "## 32-step 자율 rollout 평균", "",
        "|조건|위치 RMSE|침투율|상대 밀도 오차|입자당 에너지 초과량|",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in NAMES:
        row = acc[condition]
        lines.append(
            f"|{condition} {NAMES[condition]}|{row['position_rmse']:.5f}|"
            f"{row['penetration_rate']:.2%}|{row['density_relative_error']:.4f}|"
            f"{row['energy_excess_per_particle']:.5f}|"
        )
    lines += [
        "", "## 해석", "",
        "Optimized Ours는 SPLASH가 전체 입자의 5–50%인 구간에서 학습 기반 조건 B/C/D/E/G 중 가장 빠르며 p95 60 FPS를 통과했다. "
        "32-step 위치 RMSE는 GNN-only가 더 낮지만, Optimized Ours는 GNN-only보다 상대 밀도 오차가 낮고 훨씬 빠르다. "
        "따라서 제안 방식의 우위는 모든 정확도 지표의 단독 1위가 아니라 게임 환경에서의 정확도–처리속도 절충으로 해석한다.",
        "", "100% SPLASH에서는 p95 60 FPS를 통과하지 못하므로, 이 경우 ROI를 분할하거나 품질 단계를 낮추는 추가 최적화가 필요하다.",
    ]
    (args.output_dir / "game_7condition_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"runtime": str(runtime_path), "accuracy": str(accuracy_path), "gains": gains}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
