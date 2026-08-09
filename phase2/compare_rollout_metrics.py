"""Write a compact baseline-vs-PI-GNN rollout metric report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read(path: Path):
    with np.load(path, allow_pickle=False) as data:
        names = [str(x) for x in data["metric_names"]]
        values = data["physics_metrics"][-1]
    return dict(zip(names, map(float, values)))


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=root / "outputs" / "baseline_gns_comparison.npz")
    parser.add_argument("--pi", type=Path, default=root / "outputs" / "pi_gnn_comparison.npz")
    parser.add_argument("--output", type=Path, default=root / "outputs" / "rollout_comparison.json")
    args = parser.parse_args()
    baseline, pi = read(args.baseline), read(args.pi)
    report = {"baseline_gns": baseline, "pi_gnn": pi, "improvement": {key: baseline[key] - pi[key] for key in baseline}}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
