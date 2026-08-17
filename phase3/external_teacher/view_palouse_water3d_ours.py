"""3-D GUI: Palouse DFSPH teacher vs Water-3D Optimized-Ours SPLASH prediction."""

from __future__ import annotations

import argparse
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


COLORS = np.asarray(("#22bfff", "#ff4f7b", "#39d98a"))


class App:
    def __init__(self, root: tk.Tk, comparison: Path, terrain_path: Path):
        self.root = root
        with np.load(comparison, allow_pickle=False) as data:
            self.teacher = data["teacher_position"]
            self.baseline = data["baseline_position"]
            self.predicted = data["predicted_position"]
            self.active = data["active_mask"]
            self.state = data["routing_state"]
            self.frames = data["frames"]
            self.protocol = str(data["protocol"].item())
            self.blend_alpha = float(data["blend_alpha"]) if "blend_alpha" in data else 1.0
        with np.load(terrain_path, allow_pickle=False) as data:
            height = np.asarray(data["height"], np.float32)[::8, ::8]
            length, width = float(data["length_m"]), float(data["width_m"])
        # Project trajectory uses (cross X, up Y, downstream Z), whereas the
        # terrain file uses native (downstream X, cross Z).
        native_x = np.linspace(-length / 2, length / 2, height.shape[0])
        cross = np.linspace(-width / 2, width / 2, height.shape[1])
        native_x, cross = np.meshgrid(native_x, cross, indexing="ij")
        self.terrain_x = cross
        self.terrain_y = height
        self.terrain_z = 12.0 - native_x
        self.index = 0
        self.playing = True
        self.delay = 90
        root.title("Palouse Falls DEM · Water-3D Optimized Ours")
        root.geometry("1880x850")
        toolbar = ttk.Frame(root, padding=7); toolbar.pack(fill="x")
        self.play = ttk.Button(toolbar, text="일시정지", command=self.toggle); self.play.pack(side="left")
        ttk.Button(toolbar, text="처음", command=self.reset).pack(side="left", padx=5)
        ttk.Button(toolbar, text="폭포 정면", command=lambda: self.set_view(25, -55)).pack(side="left", padx=(10, 2))
        ttk.Button(toolbar, text="측면", command=lambda: self.set_view(18, -5)).pack(side="left", padx=2)
        ttk.Button(toolbar, text="위에서", command=lambda: self.set_view(72, -55)).pack(side="left", padx=2)
        ttk.Label(toolbar, text="재생 속도").pack(side="left", padx=(14, 3))
        self.speed = tk.DoubleVar(value=1.0)
        ttk.Scale(toolbar, from_=0.25, to=3.0, variable=self.speed, orient="horizontal", length=180).pack(side="left")
        self.slider = tk.IntVar()
        ttk.Scale(toolbar, from_=0, to=len(self.frames)-1, variable=self.slider,
                  command=self.slide).pack(side="left", fill="x", expand=True, padx=12)
        self.status = ttk.Label(toolbar); self.status.pack(side="right")
        self.figure = Figure(figsize=(18, 7), dpi=100, facecolor="#1b2329")
        self.axes = (self.figure.add_subplot(131, projection="3d", computed_zorder=False),
                     self.figure.add_subplot(132, projection="3d", computed_zorder=False),
                     self.figure.add_subplot(133, projection="3d", computed_zorder=False))
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_release_event", self.sync_camera)
        legend = ttk.Frame(root, padding=6); legend.pack(fill="x")
        ttk.Label(legend, text="● STREAM", foreground=COLORS[0]).pack(side="left", padx=8)
        ttk.Label(legend, text="● SPLASH / PI-GNN ROI", foreground=COLORS[1]).pack(side="left", padx=8)
        ttk.Label(legend, text="● POOL", foreground=COLORS[2]).pack(side="left", padx=8)
        ttk.Label(legend, text="teacher-forced 1-step 비교 · 빨강 입자만 Water-3D PI-GNN 예측", foreground="#555").pack(side="right", padx=8)
        self.scatters = []
        self._initialize_panels()
        self.draw()
        root.after(self.delay, self.tick)

    def toggle(self):
        self.playing = not self.playing
        self.play.configure(text="일시정지" if self.playing else "재생")

    def reset(self):
        self.index = 0; self.slider.set(0); self.draw()

    def slide(self, _=None):
        self.index = int(self.slider.get()); self.draw()

    def _initialize_panels(self):
        titles = ("EXTERNAL DFSPH TEACHER", "ANALYTIC BASE PHYSICS",
                  f"OPTIMIZED OURS · PI-GNN α={self.blend_alpha:.3f}")
        for axis, title in zip(self.axes, titles):
            axis.set_facecolor("#1b2329")
            axis.set_title(title, color="white", fontsize=12, fontweight="bold")
            # Display convention: X=downstream, Y=cross-stream, Z=height.
            # The stored project array remains (cross, height, downstream).
            axis.plot_surface(self.terrain_z, self.terrain_x, self.terrain_y,
                              color="#aeb8bd", edgecolor="#7d898f", linewidth=.18,
                              alpha=.82, antialiased=True, zorder=1)
            scatter = axis.scatter([], [], [], s=20, depthshade=False, linewidths=.8,
                                   edgecolors="#e8fbff", zorder=10)
            self.scatters.append(scatter)
            axis.set_xlabel("X · 하류 방향 (m)", color="#b8c4ca")
            axis.set_ylabel("Y · 좌우 방향 (m)", color="#b8c4ca")
            axis.set_zlabel("Z · 높이 (m)", color="#b8c4ca")
            axis.tick_params(colors="#8fa0a8", labelsize=7)
            axis.grid(False)
            axis.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
            axis.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
            axis.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
            axis.set_proj_type("ortho")
            axis.view_init(elev=24, azim=-55)
            axis.set_box_aspect((1, 1.25, .65))
        all_live = self.teacher[self.active]
        for axis in self.axes:
            axis.set_xlim(float(all_live[:, 2].min()) - 1, float(all_live[:, 2].max()) + 1)
            axis.set_ylim(float(all_live[:, 0].min()) - 1, float(all_live[:, 0].max()) + 1)
            axis.set_zlim(float(min(self.terrain_y.min(), all_live[:, 1].min())) - 1,
                          float(max(self.terrain_y.max(), all_live[:, 1].max())) + 1)
        self.figure.tight_layout()

    def set_view(self, elev: float, azim: float):
        for axis in self.axes:
            axis.view_init(elev=elev, azim=azim)
        self.canvas.draw_idle()

    def sync_camera(self, event):
        if event.inaxes not in self.axes:
            return
        # Never allow an underside camera: it makes correctly surface-bound
        # particles appear to pass through the semi-transparent height field.
        elev = float(np.clip(event.inaxes.elev, 5.0, 85.0))
        azim = float(event.inaxes.azim)
        for axis in self.axes:
            axis.view_init(elev=elev, azim=azim)
        self.canvas.draw_idle()

    def draw(self):
        ids = np.flatnonzero(self.active[self.index])
        state = np.minimum(self.state[self.index, ids], 2)
        colors = COLORS[state]
        sizes = np.where(state == 1, 48, np.where(state == 2, 30, 22))
        for scatter, position in zip(self.scatters, (self.teacher[self.index], self.baseline[self.index], self.predicted[self.index])):
            p = position[ids]
            scatter._offsets3d = (p[:, 2], p[:, 0], p[:, 1])
            scatter.set_color(colors)
            scatter.set_sizes(sizes)
            scatter.set_edgecolors(np.where(state == 1, "#ffffff", "#c9f6ff"))
        ids = self.active[self.index]
        roi = ids & (self.state[self.index] == 1)
        model_error = np.linalg.norm(self.predicted[self.index] - self.teacher[self.index], axis=1)
        base_error = np.linalg.norm(self.baseline[self.index] - self.teacher[self.index], axis=1)
        model_rmse = float(np.sqrt(np.mean(model_error[roi] ** 2))) if np.any(roi) else 0.0
        base_rmse = float(np.sqrt(np.mean(base_error[roi] ** 2))) if np.any(roi) else 0.0
        self.status.configure(text=f"frame {int(self.frames[self.index])} · SPLASH {int(roi.sum())} · 기본 {base_rmse:.4f} m · PI-GNN {model_rmse:.4f} m")
        self.canvas.draw_idle()

    def tick(self):
        if self.playing:
            step = max(1, int(round(float(self.speed.get()))))
            self.index = (self.index + step) % len(self.frames)
            self.slider.set(self.index); self.draw()
        self.root.after(max(16, int(self.delay / max(float(self.speed.get()), .25))), self.tick)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=here / "evaluation_palouse_water3d_ours/gui_comparison.npz")
    parser.add_argument("--terrain", type=Path, default=here / "palouse_generated/terrain_height.npz")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        with np.load(args.comparison, allow_pickle=False) as data:
            print({"frames": len(data["frames"]), "slots": data["teacher_position"].shape[1],
                   "finite": bool(np.isfinite(data["predicted_position"]).all()),
                   "protocol": str(data["protocol"].item())})
        return
    root = tk.Tk(); App(root, args.comparison, args.terrain); root.mainloop()


if __name__ == "__main__":
    main()
