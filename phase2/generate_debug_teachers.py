"""Generate the eight Phase 2 pipeline-validation trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shallow_water import TerrainData
from teacher import DebugTeacherWriter, TrajectoryConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--max-particles", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "datasets" / "debug")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    terrains = ["single_cliff", "sloped_cliff", "rocky_cliff", "split_channel"]
    summaries = []
    for index in range(args.count):
        terrain_id = terrains[index % len(terrains)]
        terrain = TerrainData.load(root / "terrains" / terrain_id)
        config = TrajectoryConfig(frames=args.frames, max_particles=args.max_particles, seed=20260809 + index)
        writer = DebugTeacherWriter(terrain, terrain_id, config)
        destination = args.output / f"trajectory_{index:03d}_{terrain_id}.npz"
        summary = writer.run(destination)
        summaries.append({"file": destination.name, **summary})
        print(f"wrote {destination}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
