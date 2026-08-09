"""Generate one or more high-fidelity WCSPH teacher trajectories."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from shallow_water import TerrainData
from teacher.trajectory_writer import TrajectoryConfig
from teacher.wcsph_writer import WCSPHTeacherWriter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain", default="single_cliff")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--max-particles", type=int, default=4096)
    parser.add_argument("--particle-mass", type=float, default=0.25)
    parser.add_argument("--flow-rate", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output or root / "datasets" / "wcsph" / f"trajectory_000_{args.terrain}.npz"
    terrain = TerrainData.load(root / "terrains" / args.terrain)
    if args.flow_rate is not None:
        terrain = replace(terrain, source_flow_m3s=args.flow_rate)
    config = TrajectoryConfig(frames=args.frames, max_particles=args.max_particles, particle_mass_kg=args.particle_mass, seed=args.seed)
    summary = WCSPHTeacherWriter(terrain, args.terrain, config).run(output)
    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
