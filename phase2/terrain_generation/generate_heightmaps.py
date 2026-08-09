"""Generate deterministic height-map terrains for river/waterfall experiments.

Each terrain directory contains the exact floating-point height field used by
the simulator, an Unreal-importable 16-bit PNG, derived normals and masks, a
human-readable metadata file, and a shaded preview.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

CLIFF_SLOPE_THRESHOLD = 0.75


@dataclass(frozen=True)
class TerrainSpec:
    name: str
    world_width_m: float = 24.0
    world_length_m: float = 32.0
    lower_height_m: float = 0.8
    cliff_height_m: float = 7.0
    cliff_z_fraction: float = 0.56
    transition_width_fraction: float = 0.018
    river_width_m: float = 4.0
    river_depth_m: float = 0.45
    upstream_slope: float = 0.018
    lower_slope: float = 0.006
    roughness_m: float = 0.08
    source_flow_rate_m3s: float = 1.2
    seed: int = 0


def smooth_noise(shape: tuple[int, int], rng: np.random.Generator, passes: int = 6) -> np.ndarray:
    """Return dependency-free, low-frequency zero-mean terrain noise."""
    noise = rng.normal(0.0, 1.0, shape).astype(np.float32)
    for _ in range(passes):
        noise = (
            noise
            + np.roll(noise, 1, 0)
            + np.roll(noise, -1, 0)
            + np.roll(noise, 1, 1)
            + np.roll(noise, -1, 1)
        ) / 5.0
    noise -= float(noise.mean())
    scale = float(np.max(np.abs(noise)))
    return noise / max(scale, 1e-6)


def make_height_field(spec: TerrainSpec, resolution: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(spec.seed)
    x = np.linspace(-spec.world_width_m / 2, spec.world_width_m / 2, resolution, dtype=np.float32)
    z = np.linspace(0.0, spec.world_length_m, resolution, dtype=np.float32)
    xx, zz = np.meshgrid(x, z)

    cliff_z = spec.cliff_z_fraction * spec.world_length_m
    transition_m = max(spec.transition_width_fraction * spec.world_length_m, 0.05)
    upper_weight = 0.5 * (1.0 - np.tanh((zz - cliff_z) / transition_m))
    upper_height = spec.lower_height_m + spec.cliff_height_m
    height = spec.lower_height_m + spec.cliff_height_m * upper_weight

    # Give both plateaus a gentle downstream fall so shallow-water routing has
    # a well-defined direction even before water reaches the cliff.
    upstream_run = np.maximum(cliff_z - zz, 0.0)
    downstream_run = np.maximum(spec.world_length_m - zz, 0.0)
    height += upper_weight * spec.upstream_slope * upstream_run
    height += (1.0 - upper_weight) * spec.lower_slope * downstream_run

    # Carve a smooth channel which widens slightly toward the cliff.
    channel_sigma = spec.river_width_m / 2.355
    widening = 1.0 + 0.22 * np.clip(zz / max(cliff_z, 1e-6), 0.0, 1.0)
    channel = np.exp(-0.5 * (xx / (channel_sigma * widening)) ** 2)
    height -= spec.river_depth_m * channel * upper_weight

    if spec.name == "sloped_cliff":
        # A broader transition produces a wall-following waterfall case.
        wide = max(0.09 * spec.world_length_m, 0.2)
        upper_weight = 0.5 * (1.0 - np.tanh((zz - cliff_z) / wide))
        height = spec.lower_height_m + spec.cliff_height_m * upper_weight
        height += upper_weight * spec.upstream_slope * upstream_run
        height += (1.0 - upper_weight) * spec.lower_slope * downstream_run
        height -= spec.river_depth_m * channel * upper_weight
    elif spec.name == "rocky_cliff":
        # Height-field-compatible protrusions. True overhangs are introduced
        # later as meshes/SDFs, not encoded in this single-valued field.
        rocks = [(-2.2, cliff_z - 0.8, 0.9, 0.7), (1.5, cliff_z + 0.35, 1.2, 0.55), (0.1, cliff_z - 1.3, 0.7, 0.45)]
        for rock_x, rock_z, radius, amplitude in rocks:
            r2 = ((xx - rock_x) / radius) ** 2 + ((zz - rock_z) / (radius * 0.8)) ** 2
            height += amplitude * np.exp(-0.5 * r2)
    elif spec.name == "split_channel":
        divider = 0.7 * np.exp(-0.5 * (xx / 0.75) ** 2) * np.exp(-0.5 * ((zz - (cliff_z - 3.0)) / 3.0) ** 2)
        height += divider * upper_weight
    elif spec.name == "natural_waterfall":
        # Irregular escarpment, meandering upper channel and a receiving basin.
        # It remains a height field so SWE, WCSPH and Unreal Landscape can all
        # consume exactly the same geometry.
        edge = cliff_z + 0.65 * np.sin(xx * 0.55) + 0.28 * np.sin(xx * 1.45 + 0.8)
        transition = max(0.032 * spec.world_length_m, 0.35)
        upper_weight = 0.5 * (1.0 - np.tanh((zz - edge) / transition))
        center = 0.75 * np.sin(zz * 0.20 - 0.9) + 0.22 * np.sin(zz * 0.63)
        widening = 1.0 + 0.35 * np.clip(zz / max(cliff_z, 1e-6), 0.0, 1.0)
        sigma = (spec.river_width_m / 2.355) * widening
        channel = np.exp(-0.5 * ((xx - center) / sigma) ** 2)
        valley = 0.055 * np.abs(xx - center) ** 1.45
        rolling = 0.34 * smooth_noise(height.shape, rng, passes=18)
        height = spec.lower_height_m + spec.cliff_height_m * upper_weight
        height += upper_weight * spec.upstream_slope * np.maximum(edge - zz, 0.0)
        height += (1.0 - upper_weight) * spec.lower_slope * downstream_run
        height += valley * (0.45 + 0.55 * upper_weight)
        height -= (spec.river_depth_m * 1.35) * channel * upper_weight
        basin = np.exp(-0.5 * (((xx + 0.3) / 4.2) ** 2 + ((zz - (cliff_z + 4.4)) / 3.0) ** 2))
        height -= 0.72 * basin * (1.0 - upper_weight)
        height += rolling
        rocks = [
            (-4.6, cliff_z - 2.0, 1.45, 0.75), (4.1, cliff_z - 1.0, 1.2, 0.85),
            (-3.7, cliff_z + 3.0, 1.0, 0.65), (3.3, cliff_z + 4.0, 1.4, 0.9),
            (-5.0, cliff_z + 7.0, 1.7, 0.65), (5.2, cliff_z + 8.0, 1.3, 0.7),
            (0.8, cliff_z + 5.3, 0.75, 0.42),
        ]
        for rock_x, rock_z, radius, amplitude in rocks:
            r2 = ((xx - rock_x) / radius) ** 2 + ((zz - rock_z) / (radius * 0.78)) ** 2
            height += amplitude * np.exp(-0.5 * r2)

    height += spec.roughness_m * smooth_noise(height.shape, rng)
    height = height.astype(np.float32)

    dz, dx = np.gradient(
        height,
        spec.world_length_m / (resolution - 1),
        spec.world_width_m / (resolution - 1),
    )
    normals = np.stack((-dx, np.ones_like(height), -dz), axis=-1)
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-6)
    slope = np.sqrt(dx * dx + dz * dz)
    cliff_mask = slope >= CLIFF_SLOPE_THRESHOLD
    if spec.name == "natural_waterfall":
        # Rough banks and boulders may also be steep, but only the main
        # escarpment crossing the river is a 2D->3D waterfall boundary.
        cliff_mask &= (np.abs(zz - edge) < 1.8) & (channel > 0.08)
    channel_mask = (channel >= np.exp(-0.5 * 2.0**2)) & (zz <= cliff_z + transition_m)
    source_mask = channel_mask & (zz <= 0.06 * spec.world_length_m)

    derived = {
        "normal": normals.astype(np.float32),
        "slope": slope.astype(np.float32),
        "cliff_mask": cliff_mask,
        "channel_mask": channel_mask,
        "source_mask": source_mask,
    }
    return height, derived


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def shaded_preview(height: np.ndarray, derived: dict[str, np.ndarray]) -> np.ndarray:
    normal = derived["normal"]
    light = np.array([-0.45, 0.75, -0.48], dtype=np.float32)
    light /= np.linalg.norm(light)
    shade = np.clip(np.sum(normal * light, axis=-1), 0.12, 1.0)
    normalized = (height - height.min()) / max(float(np.ptp(height)), 1e-6)
    low = np.array([64, 88, 52], dtype=np.float32)
    high = np.array([170, 157, 126], dtype=np.float32)
    color = low + normalized[..., None] * (high - low)
    color *= (0.45 + 0.75 * shade[..., None])
    # Diagnostic overlays: blue=channel, cyan=source, orange=detected cliff.
    channel = derived["channel_mask"]
    cliff = derived["cliff_mask"]
    source = derived["source_mask"]
    color[channel] = 0.68 * color[channel] + 0.32 * np.array([45, 115, 175], dtype=np.float32)
    color[cliff] = 0.55 * color[cliff] + 0.45 * np.array([225, 120, 45], dtype=np.float32)
    color[source] = 0.25 * color[source] + 0.75 * np.array([25, 220, 235], dtype=np.float32)
    return np.clip(color, 0, 255).astype(np.uint8)


def write_terrain(root: Path, spec: TerrainSpec, resolution: int) -> None:
    output = root / spec.name
    output.mkdir(parents=True, exist_ok=True)
    height, derived = make_height_field(spec, resolution)

    np.save(output / "height_meters.npy", height)
    np.save(output / "surface_normal.npy", derived["normal"])
    np.save(output / "slope.npy", derived["slope"])

    height_min = float(height.min())
    height_max = float(height.max())
    encoded = np.round((height - height_min) / max(height_max - height_min, 1e-6) * 65535.0).astype(np.uint16)
    Image.fromarray(encoded).save(output / "height_u16.png")
    Image.fromarray(shaded_preview(height, derived), mode="RGB").save(output / "preview.png")
    save_mask(output / "cliff_mask.png", derived["cliff_mask"])
    save_mask(output / "channel_mask.png", derived["channel_mask"])
    save_mask(output / "source_mask.png", derived["source_mask"])

    cell_size = [spec.world_width_m / (resolution - 1), spec.world_length_m / (resolution - 1)]
    metadata = {
        "schema_version": 1,
        "spec": asdict(spec),
        "resolution": [resolution, resolution],
        "axis_convention": {"array_rows": "+z downstream", "array_columns": "+x right", "height": "+y up"},
        "cell_size_m": cell_size,
        "height_encoding": {
            "file": "height_u16.png",
            "minimum_m": height_min,
            "maximum_m": height_max,
            "decode": "minimum_m + pixel/65535 * (maximum_m-minimum_m)",
        },
        "water_source": {
            "mask": "source_mask.png",
            "flow_rate_m3s": spec.source_flow_rate_m3s,
            "initial_velocity_mps": [0.0, 0.0, 1.2],
        },
        "derived_fields": {
            "normal": "surface_normal.npy",
            "slope": "slope.npy",
            "cliff_mask": "cliff_mask.png",
            "channel_mask": "channel_mask.png",
        },
        "routing_thresholds": {"cliff_slope_gradient": CLIFF_SLOPE_THRESHOLD},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "terrains")
    parser.add_argument("--resolution", type=int, default=257)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument(
        "--types",
        nargs="+",
        default=["single_cliff", "sloped_cliff", "rocky_cliff", "split_channel", "natural_waterfall"],
        choices=["single_cliff", "sloped_cliff", "rocky_cliff", "split_channel", "natural_waterfall"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resolution < 33:
        raise ValueError("resolution must be at least 33")
    for index, terrain_type in enumerate(args.types):
        spec = TerrainSpec(
            name=terrain_type,
            seed=args.seed + index,
            source_flow_rate_m3s=2.0 if terrain_type == "natural_waterfall" else 1.2,
            river_width_m=5.2 if terrain_type == "natural_waterfall" else 4.0,
            roughness_m=0.14 if terrain_type == "natural_waterfall" else 0.08,
        )
        write_terrain(args.output, spec, args.resolution)
        print(f"generated {terrain_type}: {args.output / terrain_type}")


if __name__ == "__main__":
    main()
