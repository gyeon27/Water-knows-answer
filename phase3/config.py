"""Reproducible Phase 3 configuration and external-storage discovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import ctypes


SEED = 20260809
EXPERIMENT_DIRNAME = "WaterKnowsAnswer_Phase3"


@dataclass(frozen=True)
class Phase3Config:
    seed: int = SEED
    dataset: str = "Water-3D"
    train_trajectories: int = 1000
    valid_trajectories: int = 100
    test_trajectories: int = 100
    windows_per_train_trajectory: int = 5
    validation_windows: int = 8
    training_steps: int = 5_000
    history_positions: int = 6
    hidden_size: int = 128
    message_blocks: int = 10
    particle_embedding: int = 16
    max_neighbors: int = 48
    learning_rate: float = 2e-4
    weight_decay: float = 1e-6
    gradient_clip: float = 1.0
    validate_every: int = 500
    checkpoint_every: int = 500
    early_stopping_patience_validations: int = 4
    early_stopping_min_delta: float = 1e-4
    warmup_frames: int = 10
    measured_frames: int = 300
    particle_counts: tuple[int, ...] = (2_000, 5_000, 10_000, 20_000, 50_000)
    splash_fractions: tuple[float, ...] = (0.05, 0.25, 0.50, 1.00)
    target_fps: tuple[int, ...] = (30, 60, 120, 144)
    rollout_horizons: tuple[int, ...] = (1, 8, 16, 32, 100)

    def write(self, root: Path) -> None:
        (root / "experiment_config.json").write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8"
        )


def _windows_volumes() -> list[dict[str, object]]:
    if not hasattr(ctypes, "windll"):
        return []
    kernel = ctypes.windll.kernel32
    mask = kernel.GetLogicalDrives()
    volumes = []
    for index in range(26):
        if not mask & (1 << index):
            continue
        letter = chr(ord("A") + index)
        root = f"{letter}:\\"
        label = ctypes.create_unicode_buffer(261)
        filesystem = ctypes.create_unicode_buffer(261)
        serial = ctypes.c_uint32(); maximum = ctypes.c_uint32(); flags = ctypes.c_uint32()
        if not kernel.GetVolumeInformationW(root, label, len(label), ctypes.byref(serial), ctypes.byref(maximum), ctypes.byref(flags), filesystem, len(filesystem)):
            continue
        free = ctypes.c_ulonglong(); total = ctypes.c_ulonglong(); total_free = ctypes.c_ulonglong()
        if not kernel.GetDiskFreeSpaceExW(root, ctypes.byref(free), ctypes.byref(total), ctypes.byref(total_free)):
            continue
        volumes.append({"DriveLetter": letter, "FileSystemLabel": label.value, "SizeRemaining": int(free.value)})
    return volumes


def resolve_data_root(value: str | Path, minimum_free_gb: float = 300.0) -> Path:
    if str(value).lower() != "auto":
        root = Path(value).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root
    candidates = []
    for volume in _windows_volumes():
        label = str(volume.get("FileSystemLabel") or "")
        free = float(volume.get("SizeRemaining") or 0) / 2**30
        letter = str(volume.get("DriveLetter") or "")
        if "T7" in label.upper() and letter and free >= minimum_free_gb:
            candidates.append((free, Path(f"{letter}:/{EXPERIMENT_DIRNAME}")))
    if not candidates:
        raise RuntimeError(
            "Samsung T7 volume with at least 300 GB free was not found. "
            "Pass --data-root X:\\WaterKnowsAnswer_Phase3 explicitly."
        )
    root = max(candidates)[1]
    root.mkdir(parents=True, exist_ok=True)
    return root


def ensure_layout(root: Path) -> dict[str, Path]:
    paths = {name: root / name for name in ("raw", "indices", "checkpoints", "rollouts", "benchmark", "reports")}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
