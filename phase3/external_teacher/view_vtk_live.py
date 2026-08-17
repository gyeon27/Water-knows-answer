"""Responsive live viewer for headless SPlisHSPlasH VTK exports."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import tkinter as tk
from tkinter import ttk

import numpy as np


def read_vtk_points(path: Path) -> np.ndarray:
    data = path.read_bytes()
    marker = b"POINTS "
    start = data.find(marker)
    if start < 0:
        raise ValueError("POINTS section not found")
    line_end = data.find(b"\n", start)
    words = data[start:line_end].decode("ascii").split()
    count = int(words[1])
    begin = line_end + 1
    end = begin + count * 3 * 4
    if len(data) < end:
        raise EOFError("VTK frame is still being written")
    return np.frombuffer(data[begin:end], dtype=">f4").astype(np.float32).reshape(count, 3)


def load_height(path: Path, stride: int = 16) -> tuple[np.ndarray, float, float]:
    with np.load(path) as data:
        height = np.asarray(data["height"], np.float32)[::stride, ::stride]
        length = float(data["length_m"])
        width = float(data["width_m"])
    return height, length, width


class LiveViewer:
    def __init__(self, vtk_dir: Path, terrain: Path, max_particles: int = 4500):
        self.vtk_dir = vtk_dir
        self.height, self.length, self.width = load_height(terrain)
        self.max_particles = max_particles
        self.root = tk.Tk()
        self.root.title("Phase 3 · External DFSPH Waterfall Teacher")
        self.root.geometry("1280x820")
        self.root.configure(bg="#101820")
        toolbar = ttk.Frame(self.root, padding=6)
        toolbar.pack(fill="x")
        self.paused = tk.BooleanVar(value=False)
        ttk.Button(toolbar, text="재생/일시정지", command=lambda: self.paused.set(not self.paused.get())).pack(side="left")
        ttk.Button(toolbar, text="처음", command=self.reset).pack(side="left", padx=5)
        self.status = ttk.Label(toolbar, text="VTK 프레임 대기 중…")
        self.status.pack(side="right")
        self.canvas = tk.Canvas(self.root, bg="#101820", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.frame = 0
        self.last_drawn_frame = -1
        self.last_file_count = 0
        self.unchanged_ticks = 0
        self.yaw = np.deg2rad(-38.0)
        self.pitch = np.deg2rad(23.0)
        self.drag = None
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.root.bind("<space>", lambda _: self.paused.set(not self.paused.get()))
        self.root.after(60, self.tick)

    def _press(self, event):
        self.drag = (event.x, event.y, self.yaw, self.pitch)

    def _drag(self, event):
        if self.drag is None:
            return
        x, y, yaw, pitch = self.drag
        self.yaw = yaw + (event.x - x) * 0.007
        self.pitch = np.clip(pitch + (event.y - y) * 0.005, -1.1, 1.1)
        self.draw(self._latest_points())

    def reset(self):
        self.frame = 0
        self.last_drawn_frame = -1
        self.paused.set(False)

    def files(self) -> list[Path]:
        return sorted(self.vtk_dir.glob("ParticleData_Fluid_*.vtk"), key=lambda p: int(p.stem.rsplit("_", 1)[-1]))

    def _latest_points(self) -> np.ndarray:
        files = self.files()
        if not files:
            return np.empty((0, 3), np.float32)
        index = min(max(self.frame - 1, 0), len(files) - 1)
        try:
            return read_vtk_points(files[index])
        except (OSError, ValueError, EOFError):
            return np.empty((0, 3), np.float32)

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        rotation = np.array([[cy, 0, -sy], [sy * sp, cp, cy * sp], [sy * cp, -sp, cy * cp]], np.float32)
        view = points @ rotation.T
        w = max(self.canvas.winfo_width(), 800)
        h = max(self.canvas.winfo_height(), 600)
        scale = min(w / 31.0, h / 18.0)
        screen = np.column_stack((w * 0.5 + view[:, 0] * scale, h * 0.60 - view[:, 1] * scale))
        return screen, view[:, 2]

    def draw(self, particles: np.ndarray):
        self.canvas.delete("all")
        nx, nz = self.height.shape
        xs = np.linspace(-self.length / 2, self.length / 2, nx)
        zs = np.linspace(-self.width / 2, self.width / 2, nz)
        grid = np.stack(np.meshgrid(xs, zs, indexing="ij") + (self.height,), axis=-1)[..., [0, 2, 1]]
        projected, depth = self.project(grid.reshape(-1, 3))
        projected = projected.reshape(nx, nz, 2)
        cell_depth = depth.reshape(nx, nz)
        cells = [(float(np.mean(cell_depth[i:i+2, j:j+2])), i, j) for i in range(nx-1) for j in range(nz-1)]
        for _, i, j in sorted(cells, reverse=True):
            q = projected[[i, i+1, i+1, i], [j, j, j+1, j+1]].reshape(-1)
            shade = int(66 + 35 * np.clip(self.height[i:i+2, j:j+2].mean() / max(self.height.max(), 1e-5), 0, 1))
            color = f"#{shade:02x}{shade+5:02x}{shade+8:02x}"
            self.canvas.create_polygon(*q, fill=color, outline="#58636a", width=0.35)
        if len(particles):
            if len(particles) > self.max_particles:
                ids = np.linspace(0, len(particles)-1, self.max_particles, dtype=np.int64)
                particles = particles[ids]
            screen, depth = self.project(particles)
            for idx in np.argsort(depth)[::-1]:
                x, y = screen[idx]
                r = 2.2
                self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="#26d7ff", outline="#d9f9ff", width=0.5)
        self.canvas.create_text(18, 18, anchor="nw", fill="white", font=("Segoe UI", 14, "bold"), text="EXTERNAL DFSPH WATERFALL TEACHER")
        self.canvas.create_text(18, 45, anchor="nw", fill="#9fb3c1", font=("Segoe UI", 10), text="드래그: 시점 회전  ·  Space: 일시정지")

    def tick(self):
        files = self.files()
        if len(files) == self.last_file_count:
            self.unchanged_ticks += 1
        else:
            self.unchanged_ticks = 0
            self.last_file_count = len(files)
        if files and not self.paused.get():
            if self.frame >= len(files) and self.unchanged_ticks >= 15:
                self.frame = 1
            else:
                self.frame = min(self.frame + 1, len(files))
        points = self._latest_points()
        if self.frame != self.last_drawn_frame:
            self.draw(points)
            self.last_drawn_frame = self.frame
        mode = "반복 재생" if self.unchanged_ticks >= 15 else "계산 추적"
        self.status.configure(text=f"{mode} · 생성 프레임 {len(files)} · 재생 {self.frame}/{len(files)} · 입자 {len(points)}")
        self.root.after(80, self.tick)

    def run(self):
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vtk-dir", type=Path, default=Path(tempfile.gettempdir()) / "wka_splish_teacher/output/vtk")
    parser.add_argument("--terrain", type=Path, default=Path(__file__).resolve().parent / "generated/terrain_height.npz")
    args = parser.parse_args()
    LiveViewer(args.vtk_dir, args.terrain).run()


if __name__ == "__main__":
    main()
