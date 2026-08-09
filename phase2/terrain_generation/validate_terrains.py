"""Validate generated Phase 2 terrain assets and print compact statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


REQUIRED_FILES = {
    "height_meters.npy",
    "height_u16.png",
    "surface_normal.npy",
    "slope.npy",
    "cliff_mask.png",
    "channel_mask.png",
    "source_mask.png",
    "metadata.json",
    "preview.png",
}


def validate(directory: Path) -> None:
    missing = REQUIRED_FILES - {p.name for p in directory.iterdir()}
    if missing:
        raise ValueError(f"{directory.name}: missing {sorted(missing)}")

    height = np.load(directory / "height_meters.npy")
    normal = np.load(directory / "surface_normal.npy")
    slope = np.load(directory / "slope.npy")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    resolution = tuple(metadata["resolution"])

    if height.shape != resolution or slope.shape != resolution:
        raise ValueError(f"{directory.name}: scalar field shape mismatch")
    if normal.shape != (*resolution, 3):
        raise ValueError(f"{directory.name}: normal shape mismatch")
    if not np.isfinite(height).all() or not np.isfinite(normal).all():
        raise ValueError(f"{directory.name}: non-finite data")
    normal_error = float(np.max(np.abs(np.linalg.norm(normal, axis=-1) - 1.0)))
    if normal_error >= 1e-4:
        raise ValueError(f"{directory.name}: normal error {normal_error}")

    cliff_count = int(np.count_nonzero(np.asarray(Image.open(directory / "cliff_mask.png"))))
    source_count = int(np.count_nonzero(np.asarray(Image.open(directory / "source_mask.png"))))
    if cliff_count == 0 or source_count == 0:
        raise ValueError(f"{directory.name}: empty routing mask")

    print(
        f"{directory.name:14s} "
        f"height={height.min():.3f}..{height.max():.3f}m "
        f"max_slope={slope.max():.2f} cliff_px={cliff_count} source_px={source_count}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1] / "terrains")
    args = parser.parse_args()
    terrain_dirs = sorted(p for p in args.root.iterdir() if p.is_dir())
    if not terrain_dirs:
        raise ValueError(f"no terrain directories under {args.root}")
    for directory in terrain_dirs:
        validate(directory)
    print(f"validated {len(terrain_dirs)} terrains")


if __name__ == "__main__":
    main()

