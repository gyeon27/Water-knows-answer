"""Create SPlisHSPlasH-ready waterfall terrain meshes.

The command accepts either a real DEM (GeoTIFF/ASC/NPY) or creates a
deterministic natural-cliff test terrain. Coordinates written to OBJ use the
project convention X=downstream, Y=up, Z=cross-stream and SI metres.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, zoom


def _fractal_noise(shape: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.zeros(shape, np.float64)
    weight = 1.0
    total = 0.0
    for scale in (18.0, 9.0, 4.0, 1.8):
        layer = gaussian_filter(rng.standard_normal(shape), scale, mode="reflect")
        layer /= max(float(layer.std()), 1e-8)
        result += weight * layer
        total += weight
        weight *= 0.48
    return result / total


def natural_cliff(nx: int, nz: int, length: float, width: float, seed: int) -> np.ndarray:
    """Deterministic eroded-looking plateau, cliff, receiving basin and channel."""
    x = np.linspace(-1.0, 1.0, nx)[:, None]
    z = np.linspace(-1.0, 1.0, nz)[None, :]
    noise = _fractal_noise((nx, nz), seed)
    cliff_line = 0.08 + 0.07 * np.sin(2.4 * z) + 0.025 * gaussian_filter(noise, 6)
    plateau = 7.0 / (1.0 + np.exp(-34.0 * (x - cliff_line)))
    downstream_slope = 0.75 * (x + 1.0)
    upstream_slope = 0.45 * np.maximum(x, 0.0)
    height = plateau + downstream_slope + upstream_slope + 0.34 * noise

    # A broad catchment narrows into a rocky lip. It is a depression in the
    # solid terrain, not a painted water path.
    channel_width = 0.46 - 0.22 * np.clip(x, 0.0, 1.0)
    channel = np.exp(-((z - 0.07 * np.sin(3.0 * x)) / channel_width) ** 4)
    height -= channel * (0.55 + 0.35 * np.clip(x, 0.0, 1.0))

    # Bowl below the fall and asymmetric rock ledges create 3-D impact/splash.
    bowl = np.exp(-((x + 0.48) / 0.30) ** 2 - (z / 0.58) ** 2)
    height -= 0.72 * bowl
    for cx, cz, amp, sx, sz in [
        (-0.35, -0.25, 0.65, 0.10, 0.14),
        (-0.52, 0.18, 0.50, 0.13, 0.10),
        (0.02, 0.30, 0.42, 0.08, 0.12),
    ]:
        height += amp * np.exp(-((x - cx) / sx) ** 2 - ((z - cz) / sz) ** 2)
    height -= float(height.min())
    return height.astype(np.float32)


def load_dem(path: Path) -> tuple[np.ndarray, float | None, float | None]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path).astype(np.float32), None, None
    if suffix == ".npz":
        data = np.load(path)
        key = "height" if "height" in data else data.files[0]
        return np.asarray(data[key], np.float32), None, None
    if suffix == ".asc":
        header = {}
        with path.open("r", encoding="utf-8") as stream:
            for _ in range(6):
                key, value = stream.readline().split()[:2]
                header[key.lower()] = float(value)
            grid = np.loadtxt(stream, dtype=np.float32)
        cell = float(header.get("cellsize", 1.0))
        nodata = header.get("nodata_value")
        if nodata is not None:
            grid[grid == nodata] = np.nan
        return grid, cell, cell
    if suffix in {".tif", ".tiff"}:
        try:
            import rasterio
        except ImportError:
            # ArcGIS 3DEP exports are single-band floating-point TIFFs. Pillow
            # can read those without the heavyweight GDAL/rasterio runtime.
            from PIL import Image
            image = Image.open(path)
            grid = np.asarray(image, dtype=np.float32)
            scale = image.tag_v2.get(33550)
            cell_y = abs(float(scale[1])) if scale else None
            cell_x = abs(float(scale[0])) if scale else None
            return grid, cell_y, cell_x
        else:
            with rasterio.open(path) as src:
                grid = src.read(1).astype(np.float32)
                if src.nodata is not None:
                    grid[grid == src.nodata] = np.nan
                return grid, abs(float(src.transform.e)), abs(float(src.transform.a))
    raise ValueError(f"unsupported DEM format: {path}")


def repair_and_resize(height: np.ndarray, nx: int, nz: int) -> np.ndarray:
    height = np.asarray(height, np.float32)
    if height.ndim != 2:
        raise ValueError(f"DEM must be 2-D, got {height.shape}")
    finite = np.isfinite(height)
    if not finite.any():
        raise ValueError("DEM has no finite elevations")
    height = np.where(finite, height, np.nanmedian(height[finite]))
    resized = zoom(height, (nx / height.shape[0], nz / height.shape[1]), order=1)
    resized = resized[:nx, :nz]
    resized -= float(resized.min())
    return resized.astype(np.float32)


def write_heightfield_obj(path: Path, height: np.ndarray, length: float, width: float, base: float) -> dict:
    nx, nz = height.shape
    xs = np.linspace(-length / 2, length / 2, nx)
    zs = np.linspace(-width / 2, width / 2, nz)
    top = np.stack(np.meshgrid(xs, zs, indexing="ij") + (height,), axis=-1)
    # meshgrid order above is X,Z,Y; reorder to X,Y,Z.
    top = top[..., [0, 2, 1]].reshape(-1, 3)
    vertices = top.tolist()
    faces: list[tuple[int, int, int]] = []
    for i in range(nx - 1):
        for j in range(nz - 1):
            a = i * nz + j + 1
            b = (i + 1) * nz + j + 1
            faces.extend(((a, b, a + 1), (a + 1, b, b + 1)))

    # Close the terrain with a flat bottom. A watertight body is much more
    # reliable for SDF boundary generation than a single open height surface.
    perimeter = [(0, j) for j in range(nz)]
    perimeter += [(i, nz - 1) for i in range(1, nx)]
    perimeter += [(nx - 1, j) for j in range(nz - 2, -1, -1)]
    perimeter += [(i, 0) for i in range(nx - 2, 0, -1)]
    bottom_start = len(vertices) + 1
    for i, j in perimeter:
        vertices.append([float(xs[i]), float(base), float(zs[j])])
    ring_count = len(perimeter)
    for k, (i, j) in enumerate(perimeter):
        next_k = (k + 1) % ring_count
        ti = i * nz + j + 1
        ni, nj = perimeter[next_k]
        tn = ni * nz + nj + 1
        bi, bn = bottom_start + k, bottom_start + next_k
        faces.extend(((ti, bi, tn), (tn, bi, bn)))
    center = len(vertices) + 1
    vertices.append([0.0, float(base), 0.0])
    for k in range(ring_count):
        faces.append((center, bottom_start + (k + 1) % ring_count, bottom_start + k))

    # The grid construction above is consistently inward-wound. Reverse the
    # complete closed shell so the top points upward and SDF inside/outside as
    # well as OpenGL back-face handling agree.
    faces = [(a, c, b) for a, b, c in faces]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as out:
        out.write("# Water Knows Answer watertight height-field terrain\n")
        for x, y, z in vertices:
            out.write(f"v {x:.7f} {y:.7f} {z:.7f}\n")
        for a, b, c in faces:
            out.write(f"f {a} {b} {c}\n")
    return {"vertices": len(vertices), "triangles": len(faces), "bounds_m": [[-length/2, base, -width/2], [length/2, float(height.max()), width/2]]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem", type=Path, help="USGS GeoTIFF, ESRI ASC, NPY or NPZ; omit for deterministic demo")
    parser.add_argument("--output", type=Path, default=Path("phase3/external_teacher/generated"))
    parser.add_argument("--length", type=float, default=24.0)
    parser.add_argument("--width", type=float, default=16.0)
    parser.add_argument("--collision-resolution", type=int, default=129)
    parser.add_argument("--visual-resolution", type=int, default=385)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--crop", type=int, nargs=4, metavar=("ROW0", "ROW1", "COL0", "COL1"),
                        help="crop DEM before resizing")
    parser.add_argument("--height-scale", type=float, default=1.0,
                        help="uniform terrain scale applied to elevations after subtracting the minimum")
    parser.add_argument("--flip-downstream", action="store_true",
                        help="flip DEM rows so upstream is +X and downstream is -X")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.dem:
        source, cell_x, cell_z = load_dem(args.dem)
        if args.crop:
            r0, r1, c0, c1 = args.crop
            source = source[r0:r1, c0:c1]
        if args.flip_downstream:
            source = source[::-1].copy()
        source = source - float(np.nanmin(source))
        source *= args.height_scale
        source_name = str(args.dem.resolve())
    else:
        source = natural_cliff(args.visual_resolution, args.visual_resolution, args.length, args.width, args.seed)
        cell_x = cell_z = None
        source_name = "deterministic natural-cliff test terrain"
    visual = repair_and_resize(source, args.visual_resolution, args.visual_resolution)
    collision = gaussian_filter(repair_and_resize(source, args.collision_resolution, args.collision_resolution), 0.55)
    base = min(-1.5, -0.08 * float(collision.max()))
    np.savez_compressed(args.output / "terrain_height.npz", height=visual, length_m=args.length, width_m=args.width)
    coll_stats = write_heightfield_obj(args.output / "terrain_collision.obj", collision, args.length, args.width, base)
    vis_stats = write_heightfield_obj(args.output / "terrain_visual.obj", visual, args.length, args.width, base)
    manifest = {
        "source": source_name, "seed": args.seed, "axis": "X downstream, Y up, Z cross-stream",
        "units": "metres", "source_cell_m": [cell_x, cell_z], "crop": args.crop,
        "height_scale": args.height_scale, "flip_downstream": args.flip_downstream,
        "collision": coll_stats, "visual": vis_stats,
    }
    (args.output / "terrain_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
