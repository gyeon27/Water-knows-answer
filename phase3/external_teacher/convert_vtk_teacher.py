"""Convert SPlisHSPlasH VTK frames to the Phase 3 trajectory NPZ schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree


def _line(data: bytes, marker: bytes) -> tuple[list[str], int]:
    start = data.find(marker)
    if start < 0:
        raise ValueError(f"missing VTK marker {marker!r}")
    end = data.find(b"\n", start)
    return data[start:end].decode("ascii").split(), end + 1


def read_frame(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = path.read_bytes()
    words, begin = _line(data, b"POINTS ")
    count = int(words[1])
    positions = np.frombuffer(data[begin:begin + count * 12], dtype=">f4").astype(np.float32).reshape(count, 3)

    marker = b"LOOKUP_TABLE id_table\n"
    begin = data.find(marker)
    if begin < 0:
        raise ValueError(f"missing particle IDs in {path}")
    begin += len(marker)
    particle_ids = np.frombuffer(data[begin:begin + count * 4], dtype=">u4").astype(np.int64)

    words, begin = _line(data, b"velocity 3 ")
    velocity_count = int(words[2])
    if velocity_count != count:
        raise ValueError(f"velocity count {velocity_count} != point count {count}")
    velocities = np.frombuffer(data[begin:begin + count * 12], dtype=">f4").astype(np.float32).reshape(count, 3)
    return particle_ids, positions, velocities


def project_coordinates(values: np.ndarray, *, positions: bool) -> np.ndarray:
    """Native (downstream X, up Y, cross Z) -> project (cross X, up Y, downstream Z)."""
    result = np.empty_like(values)
    result[..., 0] = values[..., 2]
    result[..., 1] = values[..., 1]
    result[..., 2] = 12.0 - values[..., 0] if positions else -values[..., 0]
    return result


def classify_routing(positions: np.ndarray, velocities: np.ndarray, active: np.ndarray,
                     radius: float, dt: float, terrain_height: Path | None) -> tuple[np.ndarray, np.ndarray]:
    frames, particles, _ = positions.shape
    splash = np.zeros((frames, particles), bool)
    state = np.zeros((frames, particles), np.uint8)
    clearance = np.full((frames, particles), np.inf, np.float32)
    if terrain_height is not None:
        with np.load(terrain_height) as terrain:
            height = np.asarray(terrain["height"], np.float32)
            length, width = float(terrain["length_m"]), float(terrain["width_m"])
        native_x = 12.0 - positions[..., 2]
        row = (native_x / length + 0.5) * (height.shape[0] - 1)
        col = (positions[..., 0] / width + 0.5) * (height.shape[1] - 1)
        bed = map_coordinates(height, [row.ravel(), col.ravel()], order=1, mode="nearest").reshape(row.shape)
        clearance = positions[..., 1] - bed
    acceleration = np.zeros_like(velocities)
    acceleration[1:] = (velocities[1:] - velocities[:-1]) / max(dt, 1e-8)
    for frame in range(frames):
        ids = np.flatnonzero(active[frame])
        if not ids.size:
            continue
        p = positions[frame, ids]
        v = velocities[frame, ids]
        speed = np.linalg.norm(v, axis=1)
        counts = np.fromiter((len(x) for x in cKDTree(p).query_ball_point(p, radius * 1.8)), np.int32, count=len(ids))
        # Proxy labels use current-frame quantities only. Sparse, energetic
        # particles are SPLASH; slow particles in the lower receiving region
        # are POOL; all remaining coherent water is STREAM.
        local_clearance = clearance[frame, ids]
        accel = np.linalg.norm(acceleration[frame, ids], axis=1)
        if frame > 0:
            continuous = active[frame - 1, ids] & (
                np.linalg.norm(positions[frame, ids] - positions[frame - 1, ids], axis=1) < 0.60
            )
        else:
            continuous = np.ones(len(ids), bool)
        near_surface = (-0.15 < local_clearance) & (local_clearance < 0.90)
        rebound = near_surface & (v[:, 1] > 0.15) & (speed > 0.8) & ((accel > 4.0) | (counts <= 10))
        airborne_rebound = (0.20 < local_clearance) & (local_clearance < 1.80) & (v[:, 1] > 0.10) & (counts <= 12)
        is_splash = (rebound | airborne_rebound) & continuous
        is_pool = (~is_splash) & near_surface & (speed < 0.45)
        splash[frame, ids[is_splash]] = True
        state[frame, ids[is_splash]] = 1
        state[frame, ids[is_pool]] = 2
    # A rebound remains a droplet for a few frames after leaving the exact
    # contact band. Stop persistence once it is strongly descending again.
    impact_seed = splash.copy()
    for frame in range(frames):
        seeds = np.flatnonzero(impact_seed[frame])
        for lag in range(1, 6):
            target = frame + lag
            if target >= frames or not seeds.size:
                break
            keep = (active[target, seeds] & (velocities[target, seeds, 1] > -0.50) &
                    (np.linalg.norm(positions[target, seeds] - positions[target - 1, seeds], axis=1) < 0.60))
            seeds = seeds[keep]
            splash[target, seeds] = True
            state[target, seeds] = 1
    return splash, state


def convert(vtk_dir: Path, output: Path, terrain_manifest: Path, fps: float,
            terrain_height: Path | None = None) -> dict:
    files = sorted(vtk_dir.glob("ParticleData_Fluid_*.vtk"), key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
    if not files:
        raise FileNotFoundError(f"no VTK frames in {vtk_dir}")
    decoded = [read_frame(path) for path in files]
    max_id = max(int(ids.max(initial=-1)) for ids, _, _ in decoded)
    particles = max_id + 1
    frames = len(decoded)
    positions = np.zeros((frames, particles, 3), np.float32)
    velocities = np.zeros_like(positions)
    active = np.zeros((frames, particles), bool)
    for frame, (ids, native_p, native_v) in enumerate(decoded):
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"duplicate particle ID in {files[frame]}")
        positions[frame, ids] = project_coordinates(native_p, positions=True)
        velocities[frame, ids] = project_coordinates(native_v, positions=False)
        active[frame, ids] = True

    # Inactive coordinates are finite placeholders only; active_mask excludes
    # them from graph construction and losses. Use birth/last positions to
    # avoid discontinuities in tools that inspect full arrays.
    for particle in range(particles):
        live = np.flatnonzero(active[:, particle])
        if not live.size:
            continue
        positions[:live[0], particle] = positions[live[0], particle]
        for frame in range(live[0] + 1, frames):
            if not active[frame, particle]:
                positions[frame, particle] = positions[frame - 1, particle]

    radius = 0.22
    splash, routing = classify_routing(positions, velocities, active, radius, 1.0 / fps, terrain_height)
    particle_type = np.zeros(particles, np.int64)
    metadata = {
        "schema": "water-knows-answer.external-teacher.v1",
        "source": "SPlisHSPlasH 2.17.0 DFSPH",
        "teacher": "external 3-D DFSPH",
        "terrain": "deterministic natural-cliff test terrain",
        "terrain_manifest": json.loads(terrain_manifest.read_text(encoding="utf-8")),
        "coordinate_mapping": "native (downstream X,up Y,cross Z) -> project (cross X,up Y,downstream Z)",
        "units": "SI metres and metres/second",
        "dt": 1.0 / fps,
        "routing_labels": "current-frame proxy; 0 STREAM, 1 SPLASH, 2 POOL",
        "frames": frames,
        "particle_slots": particles,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        positions=positions,
        velocities=velocities,
        particle_type=particle_type,
        particle_id=np.arange(particles, dtype=np.int32),
        fluid_mask=np.ones(particles, bool),
        kinematic_mask=np.zeros(particles, bool),
        active_mask=active,
        splash_roi=splash,
        routing_state=routing,
        dt=np.asarray(1.0 / fps, np.float32),
        connectivity_radius=np.asarray(radius, np.float32),
        terrain_id=np.asarray("external_natural_cliff"),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    result = {
        "output": str(output.resolve()), "frames": frames, "particle_slots": particles,
        "active_samples": int(active.sum()), "max_active": int(active.sum(axis=1).max()),
        "splash_fraction": float(splash.sum() / max(active.sum(), 1)),
    }
    return result


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vtk-dir", type=Path, default=Path(tempfile.gettempdir()) / "wka_splish_teacher/output/vtk")
    parser.add_argument("--output", type=Path, default=here / "datasets/external_dfSPH_natural_cliff_001.npz")
    parser.add_argument("--terrain-manifest", type=Path, default=here / "generated/terrain_manifest.json")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--terrain-height", type=Path,
                        help="terrain_height.npz for surface-relative SPLASH routing")
    args = parser.parse_args()
    print(json.dumps(convert(args.vtk_dir, args.output, args.terrain_manifest, args.fps,
                             args.terrain_height), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
