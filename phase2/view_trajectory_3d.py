"""Interactive dependency-light 3D viewer for Phase 2 trajectory NPZ files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import numpy as np

try:
    from phase2.shallow_water import TerrainData
except ModuleNotFoundError:
    from shallow_water import TerrainData


class SceneData:
    def __init__(self, trajectory: Path, terrain_root: Path):
        with np.load(trajectory, allow_pickle=False) as data:
            self.positions = data["positions"].astype(np.float32)
            self.velocities = data["velocities"].astype(np.float32)
            self.active = data["active_mask"].astype(bool)
            self.splash = data["splash_roi"].astype(bool)
            self.terrain_id = str(data["terrain_id"].item())
            self.dt = float(data["dt"])
            self.metadata = json.loads(str(data["metadata_json"].item()))
        self.terrain = TerrainData.load(terrain_root / self.terrain_id)
        self.speed = np.linalg.norm(self.velocities, axis=2)

    @property
    def frames(self) -> int:
        return self.positions.shape[0]

    def particle_classes(self, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ids = np.flatnonzero(self.active[frame])
        if not ids.size:
            return ids, np.empty((0, 3)), np.empty(0), np.empty(0, dtype=np.uint8)
        position = self.positions[frame, ids]
        col = np.clip(np.rint((position[:, 0] + self.terrain.width_m * 0.5) / self.terrain.dx).astype(int), 0, self.terrain.height.shape[1] - 1)
        row = np.clip(np.rint(position[:, 2] / self.terrain.dz).astype(int), 0, self.terrain.height.shape[0] - 1)
        clearance = position[:, 1] - self.terrain.height[row, col]
        kind = np.zeros(ids.size, dtype=np.uint8)  # STREAM
        kind[clearance < 0.22] = 2  # POOL / surface flow
        kind[self.splash[frame, ids]] = 1  # impact SPLASH wins
        return ids, position, self.speed[frame, ids], kind


class Camera:
    def __init__(self):
        self.yaw = -0.72
        self.pitch = -0.43
        self.zoom = 35.0

    def transform(self, points: np.ndarray, center: np.ndarray) -> np.ndarray:
        p = points - center
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        x = cy * p[:, 0] - sy * p[:, 2]
        z = sy * p[:, 0] + cy * p[:, 2]
        y = cp * p[:, 1] - sp * z
        depth = sp * p[:, 1] + cp * z
        return np.column_stack((x, y, depth))


class TrajectoryViewer:
    COLORS = {0: "#22b8ff", 1: "#ff5364", 2: "#36d98b"}
    MAX_VISIBLE_PARTICLES = 900
    MAX_STREAKS = 320

    def __init__(self, root: tk.Tk, scene: SceneData):
        self.root, self.scene = root, scene
        self.camera = Camera()
        self.frame = 0
        self.playing = True
        self.last_mouse = None
        self.center = np.array([0.0, float((scene.terrain.height.min() + scene.terrain.height.max()) * 0.5), scene.terrain.length_m * 0.54], dtype=np.float32)
        self._build_mesh()
        self._build_ui()
        self.root.after(1, self._tick)

    def _build_mesh(self) -> None:
        terrain = self.scene.terrain
        rows, cols = terrain.height.shape
        stride = max(1, int(max(rows, cols) / 28))
        rr = np.arange(0, rows, stride)
        cc = np.arange(0, cols, stride)
        if rr[-1] != rows - 1:
            rr = np.append(rr, rows - 1)
        if cc[-1] != cols - 1:
            cc = np.append(cc, cols - 1)
        x = cc * terrain.dx - terrain.width_m * 0.5
        z = rr * terrain.dz
        xx, zz = np.meshgrid(x, z)
        self.vertices = np.column_stack((xx.ravel(), terrain.height[np.ix_(rr, cc)].ravel(), zz.ravel())).astype(np.float32)
        width = cc.size
        faces = []
        for r in range(rr.size - 1):
            for c in range(cc.size - 1):
                a = r * width + c
                faces.extend(((a, a + width, a + 1), (a + 1, a + width, a + width + 1)))
        self.faces = np.asarray(faces, dtype=np.int32)
        edge_a = self.vertices[self.faces[:, 1]] - self.vertices[self.faces[:, 0]]
        edge_b = self.vertices[self.faces[:, 2]] - self.vertices[self.faces[:, 0]]
        normal = np.cross(edge_a, edge_b)
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)
        light = np.array([-0.35, 0.82, -0.45], dtype=np.float32)
        light /= np.linalg.norm(light)
        self.face_light = np.clip(np.abs(normal @ light), 0.18, 1.0)

    def _build_ui(self) -> None:
        self.root.title("Phase 2 · WCSPH 3D Trajectory Viewer")
        self.root.geometry("1280x820")
        self.root.configure(bg="#111820")
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(fill="x")
        self.play_button = ttk.Button(toolbar, text="일시정지", command=self._toggle)
        self.play_button.pack(side="left")
        ttk.Button(toolbar, text="처음", command=lambda: self._set_frame(0)).pack(side="left", padx=5)
        self.frame_var = tk.IntVar(value=0)
        self.slider = ttk.Scale(toolbar, from_=0, to=self.scene.frames - 1, variable=self.frame_var, command=self._slider)
        self.slider.pack(side="left", fill="x", expand=True, padx=10)
        self.status = ttk.Label(toolbar)
        self.status.pack(side="right")
        self.canvas = tk.Canvas(self.root, bg="#111820", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<Button-1>", self._mouse_down)
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Configure>", lambda _e: self.draw(full=True))
        self.root.bind("<space>", lambda _e: self._toggle())
        self.root.bind("<Left>", lambda _e: self._set_frame(self.frame - 1))
        self.root.bind("<Right>", lambda _e: self._set_frame(self.frame + 1))

    def _mouse_down(self, event) -> None:
        self.last_mouse = (event.x, event.y)

    def _drag(self, event) -> None:
        if self.last_mouse:
            dx, dy = event.x - self.last_mouse[0], event.y - self.last_mouse[1]
            self.camera.yaw += dx * 0.008
            self.camera.pitch = float(np.clip(self.camera.pitch + dy * 0.008, -1.45, 1.45))
            self.last_mouse = (event.x, event.y)
            self.draw(full=True)

    def _wheel(self, event) -> None:
        self.camera.zoom = float(np.clip(self.camera.zoom * (1.12 if event.delta > 0 else 0.89), 12.0, 90.0))
        self.draw(full=True)

    def _toggle(self) -> None:
        self.playing = not self.playing
        self.play_button.configure(text="일시정지" if self.playing else "재생")

    def _slider(self, _value=None) -> None:
        self.frame = int(round(self.frame_var.get()))
        self.draw(full=False)

    def _set_frame(self, frame: int) -> None:
        self.frame = frame % self.scene.frames
        self.frame_var.set(self.frame)
        self.draw(full=False)

    def _screen(self, transformed: np.ndarray) -> np.ndarray:
        w, h = max(self.canvas.winfo_width(), 2), max(self.canvas.winfo_height(), 2)
        scale = self.camera.zoom
        return np.column_stack((w * 0.5 + transformed[:, 0] * scale, h * 0.52 - transformed[:, 1] * scale))

    @staticmethod
    def _lod_indices(kind: np.ndarray, limit: int) -> np.ndarray:
        """Keep all rare splash samples and uniformly subsample the remainder."""
        count = kind.size
        if count <= limit:
            return np.arange(count)
        splash = np.flatnonzero(kind == 1)
        splash = splash[np.linspace(0, splash.size - 1, min(splash.size, limit // 3), dtype=int)] if splash.size else splash
        remaining = np.setdiff1d(np.arange(count), splash, assume_unique=True)
        slots = max(0, limit - splash.size)
        sampled = remaining[np.linspace(0, remaining.size - 1, slots, dtype=int)] if slots and remaining.size else np.empty(0, int)
        return np.sort(np.concatenate((splash, sampled)))

    def draw(self, full: bool = False) -> None:
        if not hasattr(self, "canvas"):
            return
        if full or not self.canvas.find_withtag("terrain"):
            self.canvas.delete("all")
            tv = self.camera.transform(self.vertices, self.center)
            sv = self._screen(tv)
            face_depth = tv[self.faces, 2].mean(axis=1)
            face_height = self.vertices[self.faces, 1].mean(axis=1)
            lo, hi = float(self.scene.terrain.height.min()), float(self.scene.terrain.height.max())
            for fi in np.argsort(face_depth)[::-1]:
                pts = sv[self.faces[fi]].reshape(-1).tolist()
                altitude = (face_height[fi] - lo) / max(hi - lo, 1e-6)
                illumination = float(self.face_light[fi])
                base = np.array([76, 83, 77]) * (0.55 + 0.55 * illumination) + np.array([38, 27, 19]) * (0.25 * altitude)
                rgb = np.clip(base, 22, 155).astype(int)
                fill = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
                self.canvas.create_polygon(*pts, fill=fill, outline=fill, tags=("terrain",))
            legend = (("STREAM", self.COLORS[0]), ("SPLASH", self.COLORS[1]), ("POOL/표면류", self.COLORS[2]))
            x = 18
            for label, color in legend:
                self.canvas.create_oval(x, 18, x + 10, 28, fill=color, outline="", tags=("overlay",))
                self.canvas.create_text(x + 15, 23, text=label, fill="#eaf4f8", anchor="w", font=("Segoe UI", 10, "bold"), tags=("overlay",))
                x += 115
        self.canvas.delete("water")
        ids, position, speed, kind = self.scene.particle_classes(self.frame)
        counts = np.bincount(kind, minlength=3) if ids.size else np.zeros(3, int)
        if ids.size:
            visible = self._lod_indices(kind, self.MAX_VISIBLE_PARTICLES)
            ids, position, speed, kind = ids[visible], position[visible], speed[visible], kind[visible]
            tp = self.camera.transform(position, self.center)
            sp = self._screen(tp)
            # A short per-particle velocity streak conveys continuous flow
            # without inventing graph edges between unrelated particles.
            tail_world = position - self.scene.velocities[self.frame, ids] * 0.055
            tail = self._screen(self.camera.transform(tail_world, self.center))
            order = np.argsort(tp[:, 2])[::-1]
            streak_step = max(1, math.ceil(order.size / self.MAX_STREAKS))
            streak_ids = set(order[::streak_step].tolist())
            for i in order:
                radius = 3.0 + min(float(speed[i]), 12.0) * 0.13
                x, y = sp[i]
                color = self.COLORS[int(kind[i])]
                if i in streak_ids:
                    self.canvas.create_line(tail[i, 0], tail[i, 1], x, y, fill=color, width=max(1.2, radius * 0.62), capstyle="round", tags=("water",))
                self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline="#dffbff", width=0.7, tags=("water",))
        self.canvas.tag_raise("overlay")
        shown = min(int(counts.sum()), self.MAX_VISIBLE_PARTICLES)
        self.status.configure(text=f"{self.frame + 1}/{self.scene.frames} · {self.frame * self.scene.dt:.2f}s · 노드 {int(counts.sum())} (표시 {shown}) · S {counts[0]} / X {counts[1]} / P {counts[2]}")

    def _tick(self) -> None:
        if self.playing:
            self._set_frame(self.frame + 1)
        delay = max(12, int(self.scene.dt * 1000))
        self.root.after(delay, self._tick)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", nargs="?", type=Path, default=root / "datasets" / "wcsph" / "trajectory_001_natural_waterfall.npz")
    parser.add_argument("--terrain-root", type=Path, default=root / "terrains")
    parser.add_argument("--check", action="store_true", help="load and validate data without opening a window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene = SceneData(args.trajectory, args.terrain_root)
    if args.check:
        print(json.dumps({"trajectory": str(args.trajectory), "terrain": scene.terrain_id, "frames": scene.frames, "peak_particles": int(scene.active.sum(axis=1).max()), "finite": bool(np.isfinite(scene.positions).all() and np.isfinite(scene.velocities).all())}, ensure_ascii=False, indent=2))
        return
    root = tk.Tk()
    TrajectoryViewer(root, scene)
    root.mainloop()


if __name__ == "__main__":
    main()
