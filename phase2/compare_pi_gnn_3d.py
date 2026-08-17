"""Synchronized 3D comparison GUI for WCSPH teacher and PI-GNN rollout."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk

import numpy as np

from shallow_water import TerrainData
from view_trajectory_3d import Camera


class Comparison:
    def __init__(self, path: Path, terrain_root: Path):
        with np.load(path, allow_pickle=False) as data:
            self.teacher_p = data["teacher_positions"]
            self.teacher_v = data["teacher_velocities"]
            self.teacher_a = data["teacher_active"]
            self.predicted_p = data["predicted_positions"]
            self.predicted_v = data["predicted_velocities"]
            self.predicted_a = data["predicted_active"]
            self.one_p = data["one_step_positions"]
            self.one_v = data["one_step_velocities"]
            self.error = data["position_error"]
            self.one_error = data["one_step_error"]
            self.metrics = data["physics_metrics"]
            self.metric_names = [str(x) for x in data["metric_names"]]
            self.terrain_id = str(data["terrain_id"].item())
            self.dt = float(data["dt"])
        self.terrain = TerrainData.load(terrain_root / self.terrain_id)


class Panel:
    # Tk Canvas creates one item per particle.  Keeping this below 500 avoids
    # UI stalls; temporal frame skipping preserves the apparent flow speed.
    LIMIT = 450

    def __init__(self, canvas, data: Comparison, camera: Camera):
        self.canvas, self.data, self.camera = canvas, data, camera
        self.center = np.array([0, (data.terrain.height.min() + data.terrain.height.max()) * 0.5, data.terrain.length_m * 0.54])
        rows, cols = data.terrain.height.shape
        stride = max(1, max(rows, cols) // 20)
        rr, cc = np.arange(0, rows, stride), np.arange(0, cols, stride)
        rr, cc = np.unique(np.append(rr, rows - 1)), np.unique(np.append(cc, cols - 1))
        xx, zz = np.meshgrid(cc * data.terrain.dx - data.terrain.width_m * 0.5, rr * data.terrain.dz)
        self.vertices = np.column_stack((xx.ravel(), data.terrain.height[np.ix_(rr, cc)].ravel(), zz.ravel()))
        width, faces = cc.size, []
        for r in range(rr.size - 1):
            for c in range(cc.size - 1):
                a = r * width + c
                faces.extend(((a, a + width, a + 1), (a + 1, a + width, a + width + 1)))
        self.faces = np.asarray(faces)

    def screen(self, points):
        transformed = self.camera.transform(points, self.center)
        w, h = max(self.canvas.winfo_width(), 2), max(self.canvas.winfo_height(), 2)
        xy = np.column_stack((w * 0.5 + transformed[:, 0] * self.camera.zoom, h * 0.52 - transformed[:, 1] * self.camera.zoom))
        return transformed, xy

    def terrain(self):
        self.canvas.delete("all")
        transformed, screen = self.screen(self.vertices)
        depth = transformed[self.faces, 2].mean(1)
        height = self.vertices[self.faces, 1].mean(1)
        lo, hi = self.data.terrain.height.min(), self.data.terrain.height.max()
        for fi in np.argsort(depth)[::-1]:
            level = (height[fi] - lo) / max(hi - lo, 1e-6)
            shade = int(43 + 48 * level)
            color = f"#{shade:02x}{min(shade + 8, 130):02x}{min(shade + 5, 125):02x}"
            self.canvas.create_polygon(*screen[self.faces[fi]].reshape(-1).tolist(), fill=color, outline=color, tags="terrain")

    @staticmethod
    def color(error, heatmap, teacher=False):
        if teacher:
            return "#24d9ff"
        if not heatmap:
            return "#ff5ca8"
        value = min(float(error) / 1.5, 1.0)
        return f"#{int(40 + 215 * value):02x}{int(220 - 170 * value):02x}{int(255 - 205 * value):02x}"

    def particles(self, position, active, error, heatmap=False, teacher=False, tag="water"):
        ids = np.flatnonzero(active)
        if ids.size > self.LIMIT:
            ids = ids[np.linspace(0, ids.size - 1, self.LIMIT, dtype=int)]
        if not ids.size:
            return
        transformed, screen = self.screen(position[ids])
        for local in np.argsort(transformed[:, 2])[::-1]:
            i, (x, y) = ids[local], screen[local]
            color = self.color(error[i], heatmap, teacher)
            self.canvas.create_oval(x - 3.2, y - 3.2, x + 3.2, y + 3.2, fill=color, outline="#eaffff", width=.5, tags=tag)

    def draw(self, frame, position, active, error, heatmap=False, teacher=False, overlay=None, full=False):
        if full or not self.canvas.find_withtag("terrain"):
            self.terrain()
        self.canvas.delete("water")
        if overlay is not None:
            self.particles(overlay[0], overlay[1], error, teacher=True)
        self.particles(position, active, error, heatmap, teacher)


class App:
    def __init__(self, root, data):
        self.root, self.data, self.frame, self.playing = root, data, 0, True
        self.last_tick = time.perf_counter()
        self.simulation_time = 0.0
        self.camera, self.last_mouse = Camera(), None
        root.title("WCSPH Teacher ↔ PI-GNN 3D Comparison")
        root.geometry("1500x850")
        bar = ttk.Frame(root, padding=7); bar.pack(fill="x")
        self.button = ttk.Button(bar, text="일시정지", command=self.toggle); self.button.pack(side="left")
        self.mode = tk.StringVar(value="자율 rollout")
        ttk.Combobox(bar, textvariable=self.mode, state="readonly", values=("자율 rollout", "1-step"), width=14).pack(side="left", padx=7)
        self.speed = tk.StringVar(value="2x")
        ttk.Label(bar, text="재생 속도").pack(side="left", padx=(8, 2))
        ttk.Combobox(bar, textvariable=self.speed, state="readonly",
                     values=("0.5x", "1x", "2x", "4x"), width=5).pack(side="left")
        self.overlay, self.heatmap = tk.BooleanVar(), tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="오버레이", variable=self.overlay, command=self.draw).pack(side="left")
        ttk.Checkbutton(bar, text="오차 heatmap", variable=self.heatmap, command=self.draw).pack(side="left")
        self.frame_var = tk.IntVar()
        ttk.Scale(bar, from_=0, to=data.teacher_p.shape[0] - 1, variable=self.frame_var, command=self.slide).pack(side="left", fill="x", expand=True, padx=10)
        self.label = ttk.Label(bar); self.label.pack(side="right")
        titles = ttk.Frame(root); titles.pack(fill="x")
        ttk.Label(titles, text="WCSPH TEACHER", anchor="center", font=("Segoe UI", 12, "bold")).pack(side="left", expand=True, fill="x")
        ttk.Label(titles, text="PI-GNN", anchor="center", font=("Segoe UI", 12, "bold")).pack(side="left", expand=True, fill="x")
        body = ttk.Frame(root); body.pack(fill="both", expand=True)
        self.left = tk.Canvas(body, bg="#101820", highlightthickness=0); self.left.pack(side="left", fill="both", expand=True)
        self.right = tk.Canvas(body, bg="#101820", highlightthickness=0); self.right.pack(side="left", fill="both", expand=True)
        self.panels = (Panel(self.left, data, self.camera), Panel(self.right, data, self.camera))
        for canvas in (self.left, self.right):
            canvas.bind("<Button-1>", self.down); canvas.bind("<B1-Motion>", self.drag); canvas.bind("<MouseWheel>", self.wheel)
            canvas.bind("<Configure>", lambda _e: self.draw(full=True))
        root.bind("<space>", lambda _e: self.toggle())
        root.after(1, self.tick)

    def down(self, event): self.last_mouse = event.x, event.y
    def drag(self, event):
        if self.last_mouse:
            self.camera.yaw += (event.x - self.last_mouse[0]) * .008
            self.camera.pitch = float(np.clip(self.camera.pitch + (event.y - self.last_mouse[1]) * .008, -1.45, 1.45))
            self.last_mouse = event.x, event.y; self.draw(full=True)
    def wheel(self, event): self.camera.zoom = float(np.clip(self.camera.zoom * (1.12 if event.delta > 0 else .89), 12, 90)); self.draw(full=True)
    def toggle(self): self.playing = not self.playing; self.button.configure(text="일시정지" if self.playing else "재생")
    def slide(self, _=None): self.frame = int(self.frame_var.get()); self.draw()
    def draw(self, full=False):
        f = self.frame
        predicted_p = self.data.predicted_p if self.mode.get() == "자율 rollout" else self.data.one_p
        predicted_a = self.data.predicted_a if self.mode.get() == "자율 rollout" else self.data.teacher_a
        error = self.data.error if self.mode.get() == "자율 rollout" else self.data.one_error
        self.panels[0].draw(f, self.data.teacher_p[f], self.data.teacher_a[f], self.data.error[f], teacher=True, full=full)
        overlay = (self.data.teacher_p[f], self.data.teacher_a[f]) if self.overlay.get() else None
        self.panels[1].draw(f, predicted_p[f], predicted_a[f], error[f], heatmap=self.heatmap.get(), overlay=overlay, full=full)
        m = self.data.metrics[f]
        self.label.configure(text=f"{f+1}/{self.data.teacher_p.shape[0]} · t={f*self.data.dt:.2f}s · RMSE {m[0]:.3f}m · 침투 {m[1]*100:.2f}% · 밀도 {m[2]:.3f} · 운동량 {m[3]:.3f} · 에너지+ {m[4]:.3f}")
    def tick(self):
        now = time.perf_counter()
        elapsed = min(now - self.last_tick, 0.25)
        self.last_tick = now
        if self.playing:
            rate = float(self.speed.get().removesuffix("x"))
            self.simulation_time += elapsed * rate
            advance = int(self.simulation_time / max(self.data.dt, 1e-6))
            if advance:
                self.simulation_time -= advance * self.data.dt
                self.frame = (self.frame + advance) % self.data.teacher_p.shape[0]
                self.frame_var.set(self.frame); self.draw()
        self.root.after(8, self.tick)


def main():
    root_path = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", nargs="?", type=Path, default=root_path / "outputs" / "pi_gnn_comparison.npz")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = Comparison(args.comparison, root_path / "terrains")
    if args.check:
        print({"frames": data.teacher_p.shape[0], "terrain": data.terrain_id, "finite": bool(np.isfinite(data.predicted_p).all())}); return
    window = tk.Tk(); App(window, data); window.mainloop()


if __name__ == "__main__": main()
