"""Download, decode, convert, and validate DeepMind GNS trajectories.

This is the first Phase 3 Track-B adapter.  It intentionally preserves the
public dataset's native coordinate system and kinematic obstacle particles
instead of pretending that GNS coordinates are SI metres or that ramps are a
Phase-2 height field.

The decoder is dependency-light: TFRecord framing and the subset of protobuf
used by ``tf.train.SequenceExample`` are parsed directly, so TensorFlow is not
required.  The expected source schema follows DeepMind's official
``learning_to_simulate/reading_utils.py``.

Examples (from the repository root)::

    python phase3/gns_data_adapter.py all --dataset WaterDropSample --max-trajectories 2
    python phase3/gns_data_adapter.py all --dataset WaterRamps
    python phase3/gns_data_adapter.py validate --dataset WaterRamps

Converted files are written to ``phase3/datasets/gns/<dataset>/<split>/``.
They use schema ``gns_v1`` and are not silently accepted as WCSPH/SI data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time
from typing import Iterator
from urllib.request import Request, urlopen

import numpy as np

try:
    import google_crc32c
except ImportError:  # Small adapter tests can still run without the optional C extension.
    google_crc32c = None


BASE_URL = "https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets"
SUPPORTED_DATASETS = {
    "WaterDropSample",
    "WaterDrop",
    "WaterRamps",
    "Water-3D",
}
SPLITS = ("train", "valid", "test")
KINEMATIC_PARTICLE_ID = 3
SCHEMA_VERSION = "gns_v1"


# ---------------------------------------------------------------------------
# TFRecord + tf.train.SequenceExample decoder (no TensorFlow dependency)


def _varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(data) or shift >= 70:
            raise ValueError("invalid protobuf varint")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7


def _protobuf_fields(data: bytes) -> Iterator[tuple[int, int, int | bytes]]:
    offset = 0
    while offset < len(data):
        tag, offset = _varint(data, offset)
        number, wire = tag >> 3, tag & 7
        if number == 0:
            raise ValueError("protobuf field number zero")
        if wire == 0:
            value, offset = _varint(data, offset)
        elif wire == 1:
            if offset + 8 > len(data):
                raise ValueError("truncated fixed64 field")
            value, offset = data[offset : offset + 8], offset + 8
        elif wire == 2:
            size, offset = _varint(data, offset)
            if offset + size > len(data):
                raise ValueError("truncated length-delimited field")
            value, offset = data[offset : offset + size], offset + size
        elif wire == 5:
            if offset + 4 > len(data):
                raise ValueError("truncated fixed32 field")
            value, offset = data[offset : offset + 4], offset + 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield number, wire, value


def _parse_feature(payload: bytes) -> dict[str, list[bytes] | list[int] | list[float]]:
    result: dict[str, list[bytes] | list[int] | list[float]] = {}
    for number, wire, value in _protobuf_fields(payload):
        if wire != 2 or not isinstance(value, bytes):
            continue
        if number == 1:  # BytesList
            result["bytes"] = [v for n, w, v in _protobuf_fields(value) if n == 1 and w == 2 and isinstance(v, bytes)]
        elif number == 2:  # FloatList, packed float32
            packed = next((v for n, w, v in _protobuf_fields(value) if n == 1 and w == 2 and isinstance(v, bytes)), b"")
            if len(packed) % 4:
                raise ValueError("invalid packed float list")
            result["float"] = list(struct.unpack(f"<{len(packed) // 4}f", packed))
        elif number == 3:  # Int64List, normally packed varints
            integers: list[int] = []
            for n, w, item in _protobuf_fields(value):
                if n != 1:
                    continue
                if w == 0 and isinstance(item, int):
                    integers.append(item)
                elif w == 2 and isinstance(item, bytes):
                    cursor = 0
                    while cursor < len(item):
                        integer, cursor = _varint(item, cursor)
                        integers.append(integer)
            result["int64"] = integers
    return result


def _parse_feature_map(payload: bytes) -> dict[str, dict[str, list]]:
    output: dict[str, dict[str, list]] = {}
    for number, wire, entry in _protobuf_fields(payload):
        if number != 1 or wire != 2 or not isinstance(entry, bytes):
            continue
        key = ""
        feature = b""
        for field, entry_wire, value in _protobuf_fields(entry):
            if field == 1 and entry_wire == 2 and isinstance(value, bytes):
                key = value.decode("utf-8")
            elif field == 2 and entry_wire == 2 and isinstance(value, bytes):
                feature = value
        if key:
            output[key] = _parse_feature(feature)
    return output


def _parse_feature_lists(payload: bytes) -> dict[str, list[dict[str, list]]]:
    output: dict[str, list[dict[str, list]]] = {}
    for number, wire, entry in _protobuf_fields(payload):
        if number != 1 or wire != 2 or not isinstance(entry, bytes):
            continue
        key = ""
        feature_list = b""
        for field, entry_wire, value in _protobuf_fields(entry):
            if field == 1 and entry_wire == 2 and isinstance(value, bytes):
                key = value.decode("utf-8")
            elif field == 2 and entry_wire == 2 and isinstance(value, bytes):
                feature_list = value
        if key:
            output[key] = [
                _parse_feature(value)
                for field, item_wire, value in _protobuf_fields(feature_list)
                if field == 1 and item_wire == 2 and isinstance(value, bytes)
            ]
    return output


def decode_sequence_example(payload: bytes, metadata: dict) -> dict[str, np.ndarray | int]:
    """Decode one DeepMind GNS SequenceExample into NumPy arrays."""
    context_payload = b""
    lists_payload = b""
    for number, wire, value in _protobuf_fields(payload):
        if wire == 2 and isinstance(value, bytes):
            if number == 1:
                context_payload = value
            elif number == 2:
                lists_payload = value
    context = _parse_feature_map(context_payload)
    sequences = _parse_feature_lists(lists_payload)
    if "position" not in sequences or "particle_type" not in context:
        raise ValueError("record lacks position or particle_type")

    particle_blob = context["particle_type"].get("bytes", [])
    if not particle_blob:
        raise ValueError("particle_type is not byte encoded")
    particle_type = np.frombuffer(particle_blob[0], dtype="<i8").copy()
    dim = int(metadata["dim"])
    frames = []
    for feature in sequences["position"]:
        values = feature.get("bytes", [])
        if not values:
            raise ValueError("position frame is not byte encoded")
        frame = np.frombuffer(values[0], dtype="<f4")
        if frame.size != particle_type.size * dim:
            raise ValueError(f"position size {frame.size} != particles({particle_type.size}) * dim({dim})")
        frames.append(frame.reshape(particle_type.size, dim))
    positions = np.stack(frames).astype(np.float32, copy=False)
    expected = int(metadata["sequence_length"]) + 1
    if positions.shape[0] != expected:
        raise ValueError(f"frames {positions.shape[0]} != metadata sequence_length+1 ({expected})")

    step_context = None
    # Match DeepMind's official reader: some records contain a placeholder
    # step_context byte sequence even though the dataset has no global context.
    # It is meaningful only when normalization statistics are present.
    if "context_mean" in metadata and "step_context" in sequences:
        context_frames = []
        for feature in sequences["step_context"]:
            values = feature.get("bytes", [])
            if values:
                joined = b"".join(values)
                if len(joined) % 4:
                    raise ValueError("step_context byte length is not divisible by float32 size")
                context_frames.append(np.frombuffer(joined, dtype="<f4").copy())
        if context_frames:
            step_context = np.stack(context_frames).astype(np.float32, copy=False)
    key_values = context.get("key", {}).get("int64", [0])
    return {
        "key": int(key_values[0]) if key_values else 0,
        "particle_type": particle_type,
        "position": positions,
        "step_context": step_context,
    }


def _crc32c(data: bytes) -> int:
    if google_crc32c is not None:
        return int(google_crc32c.value(data))
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    return (~crc) & 0xFFFFFFFF


def _masked_crc32c(data: bytes) -> int:
    value = _crc32c(data)
    return (((value >> 15) | (value << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def iter_tfrecord(path: Path, verify_crc: bool = True) -> Iterator[bytes]:
    with path.open("rb") as stream:
        record = 0
        while True:
            length_bytes = stream.read(8)
            if not length_bytes:
                return
            if len(length_bytes) != 8:
                raise ValueError(f"truncated TFRecord length at record {record}")
            length_crc = stream.read(4)
            length = struct.unpack("<Q", length_bytes)[0]
            payload = stream.read(length)
            payload_crc = stream.read(4)
            if len(length_crc) != 4 or len(payload) != length or len(payload_crc) != 4:
                raise ValueError(f"truncated TFRecord payload at record {record}")
            if verify_crc:
                expected_length = struct.unpack("<I", length_crc)[0]
                expected_payload = struct.unpack("<I", payload_crc)[0]
                if expected_length != _masked_crc32c(length_bytes):
                    raise ValueError(f"length CRC mismatch at record {record}")
                if expected_payload != _masked_crc32c(payload):
                    raise ValueError(f"payload CRC mismatch at record {record}")
            record += 1
            yield payload


# ---------------------------------------------------------------------------
# Conversion and derived routing labels


def embed_3d(position: np.ndarray) -> np.ndarray:
    """Map GNS 2D (x, vertical-y) to this project's (x, y, z)."""
    if position.shape[-1] == 3:
        return position.astype(np.float32, copy=True)
    if position.shape[-1] != 2:
        raise ValueError(f"only 2D/3D datasets are supported, got dim={position.shape[-1]}")
    output = np.zeros(position.shape[:-1] + (3,), dtype=np.float32)
    output[..., 0] = position[..., 0]
    output[..., 1] = position[..., 1]
    return output


def derive_routing(position: np.ndarray, particle_type: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Derive STREAM/SPLASH/POOL candidates without claiming ground-truth labels.

    GNS does not publish these labels.  SPLASH is therefore a reproducible
    kinematic proxy: non-kinematic particles with both above-median fluid speed
    and acceleration, or particles moving quickly within 2 radii of an obstacle.
    POOL is the low-speed quartile; the remainder is STREAM.
    """
    velocity = np.diff(position, axis=0, prepend=position[:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[:1])
    speed = np.linalg.norm(velocity, axis=2)
    accel = np.linalg.norm(acceleration, axis=2)
    fluid = particle_type != KINEMATIC_PARTICLE_ID
    if not np.any(fluid):
        raise ValueError("trajectory contains no non-kinematic particles")
    speed_values = speed[:, fluid]
    accel_values = accel[:, fluid]
    speed_low, speed_high = np.quantile(speed_values, (0.25, 0.65))
    accel_high = float(np.quantile(accel_values, 0.70))

    near_obstacle = np.zeros_like(speed, dtype=bool)
    obstacle_ids = np.flatnonzero(~fluid)
    fluid_ids = np.flatnonzero(fluid)
    # Obstacles are kinematic, but fluid positions vary over time.  Chunk all
    # three axes to avoid a T x N_fluid x N_obstacle temporary on WaterRamps.
    if obstacle_ids.size and fluid_ids.size:
        obstacle = position[0, obstacle_ids]
        threshold2 = (2.0 * radius) ** 2
        for frame_start in range(0, position.shape[0], 16):
            frame_slice = slice(frame_start, min(frame_start + 16, position.shape[0]))
            for fluid_start in range(0, fluid_ids.size, 256):
                ids = fluid_ids[fluid_start : fluid_start + 256]
                minimum = np.full((position[frame_slice, ids].shape[0], ids.size), np.inf, dtype=np.float32)
                for obstacle_start in range(0, obstacle.shape[0], 256):
                    obstacle_chunk = obstacle[obstacle_start : obstacle_start + 256]
                    distance2 = np.sum(
                        (position[frame_slice, ids, None, :] - obstacle_chunk[None, None, :, :]) ** 2,
                        axis=3,
                    )
                    minimum = np.minimum(minimum, np.min(distance2, axis=2))
                near_obstacle[frame_slice, ids] = minimum <= threshold2

    moving_impact = (speed >= speed_high) & (accel >= accel_high)
    fast_obstacle_contact = near_obstacle & (speed >= speed_high)
    splash = fluid[None, :] & (moving_impact | fast_obstacle_contact)
    pool = fluid[None, :] & ~splash & (speed <= speed_low)
    state = np.zeros(speed.shape, dtype=np.uint8)  # 0=STREAM
    state[splash] = 1
    state[pool] = 2
    state[:, ~fluid] = 3  # KINEMATIC/obstacle
    return splash, state


def convert_record(record: dict, metadata: dict, dataset: str, split: str, index: int) -> dict[str, np.ndarray]:
    native = np.asarray(record["position"], dtype=np.float32)
    positions = embed_3d(native)
    particle_type = np.asarray(record["particle_type"], dtype=np.int64)
    velocity = np.diff(positions, axis=0, prepend=positions[:1]).astype(np.float32)
    radius = float(metadata["default_connectivity_radius"])
    splash, state = derive_routing(positions, particle_type, radius)
    frames, particles, _ = positions.shape
    fluid_mask = particle_type != KINEMATIC_PARTICLE_ID
    active = np.broadcast_to(fluid_mask, (frames, particles)).copy()
    context = record.get("step_context")
    if context is None:
        context = np.empty((frames, 0), dtype=np.float32)
    conversion = {
        "schema": SCHEMA_VERSION,
        "source": "Google DeepMind learning_to_simulate",
        "dataset": dataset,
        "split": split,
        "trajectory_index": index,
        "record_key": int(record["key"]),
        "native_dim": int(native.shape[-1]),
        "coordinate_mapping": "2D (x,y)->project (x,y,0); 3D unchanged",
        "units": "GNS normalized coordinate per simulation step (not SI metres/seconds)",
        "dt": 1.0,
        "kinematic_particle_id": KINEMATIC_PARTICLE_ID,
        "routing_labels": "derived proxy: 0 STREAM, 1 SPLASH, 2 POOL, 3 KINEMATIC; not supplied ground truth",
        "source_metadata": metadata,
    }
    return {
        "positions": positions,
        "velocities": velocity,
        "particle_type": particle_type,
        "particle_id": np.arange(particles, dtype=np.int32),
        "fluid_mask": fluid_mask,
        "kinematic_mask": ~fluid_mask,
        "active_mask": active,
        "splash_roi": splash,
        "routing_state": state,
        "step_context": np.asarray(context, dtype=np.float32),
        "dt": np.asarray(1.0, dtype=np.float32),
        "connectivity_radius": np.asarray(radius, dtype=np.float32),
        "metadata_json": np.asarray(json.dumps(conversion, ensure_ascii=False)),
    }


# ---------------------------------------------------------------------------
# CLI operations


def download_file(url: str, destination: Path, overwrite: bool = False) -> dict:
    if destination.exists() and destination.stat().st_size and not overwrite:
        return {"file": destination.name, "bytes": destination.stat().st_size, "status": "kept"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "Water-knows-answer/Phase3-GNS-adapter"})
    digest = hashlib.sha256()
    total = 0
    started = time.perf_counter()
    with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            output.write(block)
            digest.update(block)
            total += len(block)
    temporary.replace(destination)
    return {
        "file": destination.name,
        "bytes": total,
        "sha256": digest.hexdigest(),
        "elapsed_s": time.perf_counter() - started,
        "status": "downloaded",
    }


def download_dataset(dataset: str, root: Path, splits: list[str], overwrite: bool) -> None:
    source = root / dataset / "source"
    files = ["metadata.json", *(f"{split}.tfrecord" for split in splits)]
    manifest = []
    for name in files:
        print(f"download {dataset}/{name}", flush=True)
        manifest.append(download_file(f"{BASE_URL}/{dataset}/{name}", source / name, overwrite))
    (source / "download_manifest.json").write_text(
        json.dumps({"dataset": dataset, "base_url": BASE_URL, "files": manifest}, indent=2),
        encoding="utf-8",
    )


def convert_dataset(dataset: str, root: Path, splits: list[str], max_trajectories: int | None, no_crc: bool) -> None:
    dataset_root = root / dataset
    metadata_path = dataset_root / "source" / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing {metadata_path}; run the download command first")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest: dict[str, object] = {"schema": SCHEMA_VERSION, "dataset": dataset, "splits": {}}
    for split in splits:
        source = dataset_root / "source" / f"{split}.tfrecord"
        if not source.exists():
            raise FileNotFoundError(f"missing {source}; run the download command first")
        output = dataset_root / split
        output.mkdir(parents=True, exist_ok=True)
        files = []
        for index, payload in enumerate(iter_tfrecord(source, verify_crc=not no_crc)):
            if max_trajectories is not None and index >= max_trajectories:
                break
            record = decode_sequence_example(payload, metadata)
            converted = convert_record(record, metadata, dataset, split, index)
            path = output / f"trajectory_{index:05d}.npz"
            np.savez_compressed(path, **converted)
            files.append({
                "file": str(path.relative_to(dataset_root)),
                "frames": int(converted["positions"].shape[0]),
                "particles": int(converted["positions"].shape[1]),
                "fluid": int(converted["fluid_mask"].sum()),
                "kinematic": int(converted["kinematic_mask"].sum()),
                "splash_fraction": float(converted["splash_roi"].sum() / max(converted["active_mask"].sum(), 1)),
            })
            print(f"converted {split} trajectory {index}: {files[-1]}", flush=True)
        manifest["splits"][split] = files
    (dataset_root / "conversion_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_dataset(dataset: str, root: Path, splits: list[str]) -> dict:
    dataset_root = root / dataset
    report: dict[str, object] = {"dataset": dataset, "schema": SCHEMA_VERSION, "splits": {}, "valid": True}
    for split in splits:
        files = sorted((dataset_root / split).glob("trajectory_*.npz"))
        split_report = {"trajectories": len(files), "frames": 0, "particles": 0, "errors": []}
        for path in files:
            try:
                with np.load(path, allow_pickle=False) as data:
                    required = {"positions", "velocities", "particle_type", "fluid_mask", "kinematic_mask", "active_mask", "splash_roi", "routing_state", "metadata_json"}
                    missing = required.difference(data.files)
                    if missing:
                        raise ValueError(f"missing keys {sorted(missing)}")
                    position = data["positions"]
                    velocity = data["velocities"]
                    active = data["active_mask"]
                    splash = data["splash_roi"]
                    if position.ndim != 3 or position.shape[2] != 3:
                        raise ValueError(f"bad positions shape {position.shape}")
                    if velocity.shape != position.shape or active.shape != position.shape[:2] or splash.shape != position.shape[:2]:
                        raise ValueError("frame/particle shapes disagree")
                    if not np.isfinite(position).all() or not np.isfinite(velocity).all():
                        raise ValueError("NaN/Inf in positions or velocities")
                    if np.any(splash & ~active):
                        raise ValueError("kinematic/inactive particle marked as splash")
                    metadata = json.loads(str(data["metadata_json"].item()))
                    if metadata.get("schema") != SCHEMA_VERSION:
                        raise ValueError("schema mismatch")
                    split_report["frames"] += int(position.shape[0])
                    split_report["particles"] += int(position.shape[1])
            except Exception as error:  # report every broken file in one pass
                split_report["errors"].append(f"{path.name}: {error}")
        if not files:
            split_report["errors"].append("no converted trajectories")
        if split_report["errors"]:
            report["valid"] = False
        report["splits"][split] = split_report
    output = dataset_root / "validation_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "convert", "validate", "all"))
    parser.add_argument("--dataset", default="WaterDropSample", choices=sorted(SUPPORTED_DATASETS))
    parser.add_argument("--root", type=Path, default=repository / "phase3" / "datasets" / "gns")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--max-trajectories", type=int, help="limit each split for adapter/debug tests")
    parser.add_argument("--overwrite", action="store_true", help="redownload existing source files")
    parser.add_argument("--no-crc", action="store_true", help="skip TFRecord CRC32C checks (not recommended)")
    args = parser.parse_args()
    if args.max_trajectories is not None and args.max_trajectories < 1:
        parser.error("--max-trajectories must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    if args.command in ("download", "all"):
        download_dataset(args.dataset, args.root, args.splits, args.overwrite)
    if args.command in ("convert", "all"):
        convert_dataset(args.dataset, args.root, args.splits, args.max_trajectories, args.no_crc)
    if args.command in ("validate", "all"):
        report = validate_dataset(args.dataset, args.root, args.splits)
        if not report["valid"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
