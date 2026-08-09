"""Run the Phase 2 shallow-water baseline and save fields plus cliff flux."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from shallow_water import ShallowWaterSolver, TerrainData


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain", default="single_cliff")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--frame-dt", type=float, default=1.0 / 30.0)
    parser.add_argument("--initial-depth", type=float, default=0.015)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    terrain_dir = root / "terrains" / args.terrain
    terrain = TerrainData.load(terrain_dir)
    solver = ShallowWaterSolver(terrain, initial_depth_m=args.initial_depth)
    initial_volume = solver.volume
    output = args.output / args.terrain
    output.mkdir(parents=True, exist_ok=True)

    frames = int(np.ceil(args.seconds / args.frame_dt))
    flux_rows: list[np.ndarray] = []
    for frame in range(frames):
        for event in solver.advance(min(args.frame_dt, args.seconds - solver.time)):
            if event.x.size:
                flux_rows.append(
                    np.column_stack(
                        (
                            np.full(event.x.size, event.time_s),
                            np.full(event.x.size, event.duration_s),
                            event.x,
                            event.y,
                            event.z,
                            event.discharge_m3s,
                            event.velocity_xyz,
                        )
                    )
                )
        if frame % 10 == 0 or frame == frames - 1:
            np.savez_compressed(output / f"frame_{frame:05d}.npz", depth=solver.h, momentum_x=solver.hu, momentum_z=solver.hv)

    flux = np.concatenate(flux_rows, axis=0) if flux_rows else np.empty((0, 9))
    np.save(output / "waterfall_flux.npy", flux)
    summary = {
        "terrain": args.terrain,
        "seconds": solver.time,
        "initial_volume_m3": initial_volume,
        "final_volume_m3": solver.volume,
        "injected_volume_m3": solver.injected_volume,
        "waterfall_volume_m3": solver.waterfall_volume,
        "boundary_volume_m3": solver.boundary_volume,
        "mass_balance_error_m3": solver.mass_balance_error(initial_volume),
        "flux_schema": ["time_s", "duration_s", "x_m", "y_m", "z_m", "discharge_m3s", "vx_mps", "vy_mps", "vz_mps"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
