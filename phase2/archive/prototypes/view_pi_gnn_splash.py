"""Single-panel PI-GNN rollout viewer focused on the waterfall base.

Shows only the PI-GNN prediction (no WCSPH teacher side-by-side) and
colors particles purely by STREAM vs SPLASH state, since the splash /
impact region at the cliff base is the behaviour PI-GNN is meant to
approximate instead of the full WCSPH solve.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import numpy as np

from phase2.shallow_water import TerrainData
from phase2.view_trajectory_3d import Camera

# Same impact-zone rule used when the WCSPH teacher data was generated
# (see teacher/wcsph_writer.py): near the surface and moving fast enough
# to be considered a splash rather than smooth stream flow.
SPLASH_CLEARANCE_M = 0.45
SPLASH_SPEED_MPS = 1.0


class RolloutScene:
    def __init__(self, comparison: Path, terrain_root: Path):
        with np.load(comparison, allow_pickle=False) as data:
            self.positions = data["predicted_positions"].astype(np.float32)
            self.velocities = data["predicted_velocities"].astype(np.float32)
            self.active = data["predicted_active"].astype(bool)
            self.terrain_id = str(data["terrain_id"].item())
            self.dt = float(data["dt"])
        self.terrain = TerrainData.load(terrain_root / self.terrain_id)
        self.speed = np.linalg.norm(self.velocities, axis=2)
        self.base_center = self._find_ground_impact_center()

    def _find_ground_impact_center(self) -> np.ndarray:
        """Centroid of particles that actually hit the floor (not just the cliff face).

        A terrain-mask heuristic (cliff row + offset) puts the camera on the
        cliff face itself, not on the pool where water actually lands — that's
        why the old default view showed the slope filling the frame with the
        impact crammed into a corner. This instead looks at where particles
        are near the ground (small clearance) *and* low in absolute height
        (rules out particles clinging to the cliff face partway down) across
        every frame, and centers on that.
        """
        terrain = self.terrain
        hits = []
        for frame in range(self.frames):
            ids = np.flatnonzero(self.active[frame])
            if not ids.size:
                continue
            position = self.positions[frame, ids]
            col = np.clip(np.rint((position[:, 0] + terrain.width_m * 0.5) / terrain.dx).astype(int), 0, terrain.height.shape[1] - 1)
            row = np.clip(np.rint(position[:, 2] / terrain.dz).astype(int), 0, terrain.height.shape[0] - 1)
            clearance = position[:, 1] - terrain.height[row, col]
            speed = self.speed[frame, ids]
            ground_hit = (clearance < 0.2) & (speed > SPLASH_SPEED_MPS) & (position[:, 1] < 4.0)
            if ground_hit.any():
                hits.append(position[ground_hit])
        if hits:
            return np.concatenate(hits, axis=0).mean(axis=0).astype(np.float32)
        cliff_rows = np.flatnonzero(terrain.cliff.any(axis=1))
        base_row = int(cliff_rows.max()) if cliff_rows.size else terrain.height.shape[0] // 2
        cliff_cols = np.flatnonzero(terrain.cliff[base_row]) if cliff_rows.size else np.array([])
        base_col = int(round(cliff_cols.mean())) if cliff_cols.size else terrain.height.shape[1] // 2
        x = base_col * terrain.dx - terrain.width_m * 0.5
        z = min(base_row + 6, terrain.height.shape[0] - 1) * terrain.dz
        y = float(terrain.height[base_row, base_col])
        return np.array([x, y, z], dtype=np.float32)

    @property
    def frames(self) -> int:
        return self.positions.shape[0]

    def particle_states(self, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (positions, speeds, is_splash) for active particles."""
        ids = np.flatnonzero(self.active[frame])
        if not ids.size:
            return np.empty((0, 3)), np.empty(0), np.empty(0, dtype=bool)
        position = self.positions[frame, ids]
        terrain = self.terrain
        col = np.clip(np.rint((position[:, 0] + terrain.width_m * 0.5) / terrain.dx).astype(int), 0, terrain.height.shape[1] - 1)
        row = np.clip(np.rint(position[:, 2] / terrain.dz).astype(int), 0, terrain.height.shape[0] - 1)
        clearance = position[:, 1] - terrain.height[row, col]
        speed = self.speed[frame, ids]
        is_splash = (clearance < SPLASH_CLEARANCE_M) & (speed > SPLASH_SPEED_MPS)
        return position, speed, is_splash


class SplashViewer:
    STREAM_COLOR = "#22b8ff"
    SPLASH_COLOR = "#ff5364"
    MAX_VISIBLE_PARTICLES = 900

    def __init__(self, root: tk.Tk, scene: RolloutScene):
        self.root, self.scene = root, scene
        self.camera = Camera()
        self.camera.yaw = -0.15
        self.camera.pitch = -0.55  # steeper look-down angle onto the pool, not along the cliff face
        self.camera.zoom = 70.0  # zoomed in on the impact pool itself, not the whole cliff
        self.frame = 0
        self.playing = True
        self.last_mouse = None
        self.center = scene.base_center.copy()
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
        self.root.title("PI-GNN Rollout · Stream / Splash 하단부 뷰")
        self.root.geometry("1100x800")
        self.root.configure(bg="#111820")
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(fill="x")
        self.play_button = ttk.Button(toolbar, text="일시정지", command=self._toggle)
        self.play_button.pack(side="left")
        ttk.Button(toolbar, text="처음", command=lambda: self._set_frame(0)).pack(side="left", padx=5)
        ttk.Button(toolbar, text="폭포 하단 초점", command=self._recenter_base).pack(side="left", padx=5)
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

    def _recenter_base(self) -> None:
        self.center = self.scene.base_center.copy()
        self.camera.zoom = 58.0
        self.draw(full=True)

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
        self.camera.zoom = float(np.clip(self.camera.zoom * (1.12 if event.delta > 0 else 0.89), 12.0, 160.0))
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
    def _lod_indices(is_splash: np.ndarray, limit: int) -> np.ndarray:
        """Keep all rare splash samples and uniformly subsample the rest."""
        count = is_splash.size
        if count <= limit:
            return np.arange(count)
        splash = np.flatnonzero(is_splash)
        splash = splash[np.linspace(0, splash.size - 1, min(splash.size, limit // 2), dtype=int)] if splash.size else splash
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
            legend = (("STREAM", self.STREAM_COLOR), ("SPLASH", self.SPLASH_COLOR))
            x = 18
            for label, color in legend:
                self.canvas.create_oval(x, 18, x + 10, 28, fill=color, outline="", tags=("overlay",))
                self.canvas.create_text(x + 15, 23, text=label, fill="#eaf4f8", anchor="w", font=("Segoe UI", 10, "bold"), tags=("overlay",))
                x += 115
        self.canvas.delete("water")
        position, speed, is_splash = self.scene.particle_states(self.frame)
        if position.shape[0]:
            visible = self._lod_indices(is_splash, self.MAX_VISIBLE_PARTICLES)
            position, speed, is_splash = position[visible], speed[visible], is_splash[visible]
            tp = self.camera.transform(position, self.center)
            sp = self._screen(tp)
            order = np.argsort(tp[:, 2])[::-1]
            for i in order:
                radius = 3.0 + min(float(speed[i]), 12.0) * 0.13
                x, y = sp[i]
                color = self.SPLASH_COLOR if is_splash[i] else self.STREAM_COLOR
                self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline="#dffbff", width=0.7, tags=("water",))
        self.canvas.tag_raise("overlay")
        splash_count = int(is_splash.sum()) if position.shape[0] else 0
        stream_count = position.shape[0] - splash_count
        self.status.configure(text=f"{self.frame + 1}/{self.scene.frames} · {self.frame * self.scene.dt:.2f}s · STREAM {stream_count} / SPLASH {splash_count}")

    def _tick(self) -> None:
        if self.playing:
            self._set_frame(self.frame + 1)
        delay = max(12, int(self.scene.dt * 1000))
        self.root.after(delay, self._tick)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", nargs="?", type=Path, default=root / "outputs" / "pi_gnn_comparison.npz")
    parser.add_argument("--terrain-root", type=Path, default=root / "terrains")
    parser.add_argument("--check", action="store_true", help="load and validate data without opening a window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene = RolloutScene(args.comparison, args.terrain_root)
    if args.check:
        import json

        position, _, is_splash = scene.particle_states(0)
        print(json.dumps({
            "comparison": str(args.comparison),
            "terrain": scene.terrain_id,
            "frames": scene.frames,
            "base_center": scene.base_center.tolist(),
            "frame0_particles": int(position.shape[0]),
            "frame0_splash": int(is_splash.sum()),
        }, ensure_ascii=False, indent=2))
        return
    root = tk.Tk()
    SplashViewer(root, scene)
    root.mainloop()


if __name__ == "__main__":
    main()
