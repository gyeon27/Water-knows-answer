"""Interactive 3D GUI for the routed SPLASH-ROI Optimized Ours rollout."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
import torch

from .config import Phase3Config, resolve_data_root
from .data import IndexedTFRecord, KINEMATIC_ID, deterministic_windows, graph_from_gns
from .evaluation import _condition_acceleration, _load_model


COLORS = {0: "#25c6ff", 1: "#ff4f81", 2: "#36e69a", 3: "#7d8792"}


class Camera:
    def __init__(self, zoom: float):
        self.yaw, self.pitch, self.zoom = -0.72, -0.40, zoom

    def transform(self, points: np.ndarray, center: np.ndarray) -> np.ndarray:
        p = points - center
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        x = cy * p[:, 0] - sy * p[:, 2]
        z = sy * p[:, 0] + cy * p[:, 2]
        y = cp * p[:, 1] - sp * z
        depth = sp * p[:, 1] + cp * z
        return np.column_stack((x, y, depth))


def _cache_path(root: Path, trajectory: int, start: int, steps: int) -> Path:
    directory = root / "rollouts" / "gui_cache"
    directory.mkdir(parents=True, exist_ok=True)
    # v2 deliberately matches evaluation condition G.  The previous cache
    # mixed an independently advanced SWE state into a model trained against
    # the analytic base acceleration, which caused routing feedback/divergence.
    return directory / f"optimized_ours_v2_t{trajectory:03d}_f{start:03d}_{steps}step.npz"


def generate_rollout(root: Path, trajectory: int, steps: int, cfg: Phase3Config, regenerate: bool = False) -> Path:
    raw = root / "raw" / cfg.dataset
    dataset = IndexedTFRecord(raw / "test.tfrecord", root / "indices" / cfg.dataset / "test.npy", raw / "metadata.json")
    source = dataset.read(trajectory)
    frames = int(source["position"].shape[0])
    start = deterministic_windows(frames - steps, 1, cfg.seed + 900_000 + trajectory)[0]
    cache = _cache_path(root, trajectory, start, steps)
    if cache.exists() and not regenerate:
        return cache
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to generate the optimized rollout cache")
    device = torch.device("cuda")
    model = _load_model(root, "ours", cfg, device)
    teacher_all = np.asarray(source["position"], np.float32)
    types = np.asarray(source["particle_type"], np.int64)
    fluid = types != KINEMATIC_ID
    working = teacher_all.copy()
    previous_v = working[start] - working[start - 1]
    teacher = np.empty((steps + 1, working.shape[1], 3), np.float32)
    predicted = np.empty_like(teacher)
    states = np.empty((steps + 1, working.shape[1]), np.uint8)
    rmse = np.empty(steps + 1, np.float32)
    velocity_rmse = np.empty(steps + 1, np.float32)
    penetration = np.empty(steps + 1, np.float32)
    roi_count = np.empty(steps + 1, np.int32)
    bounds = np.asarray(dataset.metadata["bounds"], np.float32)
    started = time.perf_counter()
    for offset in range(steps + 1):
        frame = start + offset
        local = {**source, "position": working}
        graph = graph_from_gns(local, dataset.metadata, frame, cfg, "ours", build_edges=False)
        teacher[offset] = teacher_all[frame]
        predicted[offset] = working[frame]
        states[offset] = graph.routing_state
        roi_count[offset] = int(np.sum(fluid & (graph.routing_state == 1)))
        difference = working[frame, fluid] - teacher_all[frame, fluid]
        rmse[offset] = float(np.sqrt(np.mean(difference * difference)))
        true_v = teacher_all[frame] - teacher_all[frame - 1]
        velocity_rmse[offset] = float(np.sqrt(np.mean((previous_v[fluid] - true_v[fluid]) ** 2)))
        p = working[frame, fluid]
        penetration[offset] = float(np.mean(np.any((p < bounds[:, 0]) | (p > bounds[:, 1]), axis=1)))
        if offset == steps:
            break
        acceleration = _condition_acceleration("G", graph, dataset.metadata, model, device)
        # Match the trained/evaluated Optimized-Ours distribution exactly:
        # analytic base acceleration for STREAM/POOL and ROI PI-GNN residual
        # for SPLASH.  A separately advanced SWE state must not be injected
        # here without interface-aware training/blending.
        next_v = previous_v + acceleration
        next_p = working[frame] + next_v
        next_p[~fluid] = teacher_all[frame + 1, ~fluid]
        next_v[~fluid] = teacher_all[frame + 1, ~fluid] - teacher_all[frame, ~fluid]
        working[frame + 1] = next_p
        previous_v = next_v
        if (offset + 1) % 10 == 0:
            print(f"GUI rollout {offset + 1}/{steps}", flush=True)
    np.savez_compressed(
        cache, teacher_position=teacher, predicted_position=predicted,
        particle_type=types, routing_state=states, position_rmse=rmse,
        velocity_rmse=velocity_rmse, penetration_rate=penetration,
        roi_count=roi_count, bounds=bounds, trajectory=np.int64(trajectory),
        start_frame=np.int64(start), generation_seconds=np.float64(time.perf_counter() - started),
    )
    return cache


class Scene:
    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as data:
            self.teacher = data["teacher_position"]
            self.predicted = data["predicted_position"]
            self.types = data["particle_type"]
            self.states = data["routing_state"]
            self.rmse = data["position_rmse"]
            self.velocity_rmse = data["velocity_rmse"]
            self.penetration = data["penetration_rate"]
            self.roi_count = data["roi_count"]
            self.bounds = data["bounds"]
            self.trajectory = int(data["trajectory"])
            self.start = int(data["start_frame"])
            self.generation_seconds = float(data["generation_seconds"])
        self.fluid = self.types != KINEMATIC_ID
        self.center = self.bounds.mean(axis=1)
        self.span = float(np.max(self.bounds[:, 1] - self.bounds[:, 0]))


class Panel:
    MAX_FLUID = 1100
    MAX_BOUNDARY = 450

    def __init__(self, canvas: tk.Canvas, scene: Scene, camera: Camera):
        self.canvas, self.scene, self.camera = canvas, scene, camera

    def _project(self, points):
        transformed = self.camera.transform(points, self.scene.center)
        w, h = max(self.canvas.winfo_width(), 2), max(self.canvas.winfo_height(), 2)
        screen = np.column_stack((w * 0.5 + transformed[:, 0] * self.camera.zoom,
                                  h * 0.52 - transformed[:, 1] * self.camera.zoom))
        return transformed, screen

    @staticmethod
    def _sample(ids: np.ndarray, limit: int) -> np.ndarray:
        return ids if ids.size <= limit else ids[np.linspace(0, ids.size - 1, limit, dtype=int)]

    def draw(self, position, states, mode: str, error, show_boundary: bool, overlay=None):
        self.canvas.delete("all")
        if overlay is not None:
            ids = self._sample(np.flatnonzero(self.scene.fluid), self.MAX_FLUID)
            _, xy = self._project(overlay[ids])
            for x, y in xy:
                self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, outline="#f4f8ff", width=1)
        ids = np.flatnonzero(self.scene.fluid)
        if mode == "SPLASH ROI만":
            ids = ids[states[ids] == 1]
        ids = self._sample(ids, self.MAX_FLUID)
        if show_boundary:
            boundary = self._sample(np.flatnonzero(~self.scene.fluid), self.MAX_BOUNDARY)
            ids = np.concatenate((ids, boundary))
        transformed, xy = self._project(position[ids])
        order = np.argsort(transformed[:, 2])[::-1]
        max_error = max(float(np.quantile(error[self.scene.fluid], 0.95)), 1e-8)
        for local in order:
            particle = ids[local]; x, y = xy[local]
            state = 3 if not self.scene.fluid[particle] else int(states[particle])
            if mode == "오차 Heatmap" and state != 3:
                value = min(float(error[particle]) / max_error, 1.0)
                color = f"#{int(40 + 215 * value):02x}{int(220 - 190 * value):02x}{int(255 - 210 * value):02x}"
            else:
                color = COLORS[state]
            radius = 3.6 if state == 1 else 2.7 if state != 3 else 1.7
            outline = "#ffffff" if state == 1 else color
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                    fill=color, outline=outline, width=1 if state == 1 else 0)


class App:
    def __init__(self, root: tk.Tk, scene: Scene):
        self.root, self.scene = root, scene
        self.frame, self.playing, self.last_mouse = 0, True, None
        self.camera = Camera(zoom=max(28.0, min(110.0, 560.0 / max(scene.span, 1e-3))))
        root.title("Phase 3 · Optimized Ours · SPLASH ROI PI-GNN")
        root.geometry("1500x880")
        toolbar = ttk.Frame(root, padding=8); toolbar.pack(fill="x")
        self.play_button = ttk.Button(toolbar, text="일시정지", command=self.toggle); self.play_button.pack(side="left")
        ttk.Button(toolbar, text="처음", command=self.reset).pack(side="left", padx=4)
        self.mode = tk.StringVar(value="상태 색상")
        ttk.Combobox(toolbar, textvariable=self.mode, state="readonly", width=14,
                     values=("상태 색상", "오차 Heatmap", "SPLASH ROI만")).pack(side="left", padx=8)
        self.overlay = tk.BooleanVar(value=False); self.boundary = tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar, text="Teacher 오버레이", variable=self.overlay, command=self.draw).pack(side="left")
        ttk.Checkbutton(toolbar, text="경계 입자", variable=self.boundary, command=self.draw).pack(side="left")
        self.frame_var = tk.IntVar()
        ttk.Scale(toolbar, from_=0, to=scene.predicted.shape[0] - 1, variable=self.frame_var,
                  command=self.slide).pack(side="left", fill="x", expand=True, padx=12)
        self.metric = ttk.Label(toolbar); self.metric.pack(side="right")
        titles = ttk.Frame(root); titles.pack(fill="x")
        ttk.Label(titles, text="WATER-3D TEACHER", anchor="center", font=("Segoe UI", 12, "bold")).pack(side="left", expand=True, fill="x")
        ttk.Label(titles, text="OPTIMIZED OURS · ANALYTIC BASE + SPLASH ROI PI-GNN", anchor="center", font=("Segoe UI", 12, "bold")).pack(side="left", expand=True, fill="x")
        body = ttk.Frame(root); body.pack(fill="both", expand=True)
        self.canvases = (tk.Canvas(body, bg="#101821", highlightthickness=0), tk.Canvas(body, bg="#101821", highlightthickness=0))
        for canvas in self.canvases:
            canvas.pack(side="left", fill="both", expand=True)
            canvas.bind("<Button-1>", self.down); canvas.bind("<B1-Motion>", self.drag)
            canvas.bind("<MouseWheel>", self.wheel); canvas.bind("<Configure>", lambda _e: self.draw())
        self.panels = tuple(Panel(c, scene, self.camera) for c in self.canvases)
        legend = ttk.Frame(root, padding=6); legend.pack(fill="x")
        ttk.Label(legend, text="● STREAM", foreground=COLORS[0]).pack(side="left", padx=8)
        ttk.Label(legend, text="● SPLASH / GNN ROI", foreground=COLORS[1]).pack(side="left", padx=8)
        ttk.Label(legend, text="● POOL", foreground=COLORS[2]).pack(side="left", padx=8)
        ttk.Label(legend, text="● KINEMATIC BOUNDARY", foreground=COLORS[3]).pack(side="left", padx=8)
        ttk.Label(legend, text=f"trajectory {scene.trajectory} · start {scene.start} · cache generation {scene.generation_seconds:.1f}s").pack(side="right")
        root.bind("<space>", lambda _e: self.toggle())
        root.after(1, self.tick)

    def down(self, event): self.last_mouse = (event.x, event.y)
    def drag(self, event):
        if self.last_mouse:
            self.camera.yaw += (event.x - self.last_mouse[0]) * .008
            self.camera.pitch = float(np.clip(self.camera.pitch + (event.y - self.last_mouse[1]) * .008, -1.45, 1.45))
            self.last_mouse = (event.x, event.y); self.draw()
    def wheel(self, event):
        self.camera.zoom = float(np.clip(self.camera.zoom * (1.12 if event.delta > 0 else .89), 10, 240)); self.draw()
    def toggle(self):
        self.playing = not self.playing; self.play_button.configure(text="일시정지" if self.playing else "재생")
    def reset(self): self.frame = 0; self.frame_var.set(0); self.draw()
    def slide(self, _=None): self.frame = int(self.frame_var.get()); self.draw()
    def draw(self):
        f = self.frame
        error = np.linalg.norm(self.scene.predicted[f] - self.scene.teacher[f], axis=1)
        self.panels[0].draw(self.scene.teacher[f], self.scene.states[f], "상태 색상", error, self.boundary.get())
        overlay = self.scene.teacher[f] if self.overlay.get() else None
        self.panels[1].draw(self.scene.predicted[f], self.scene.states[f], self.mode.get(), error, self.boundary.get(), overlay)
        self.metric.configure(text=f"{f:03d}/{self.scene.predicted.shape[0]-1} · RMSE {self.scene.rmse[f]:.5f} · vRMSE {self.scene.velocity_rmse[f]:.5f} · 침투 {100*self.scene.penetration[f]:.2f}% · ROI {self.scene.roi_count[f]}")
    def tick(self):
        if self.playing:
            self.frame = (self.frame + 1) % self.scene.predicted.shape[0]
            self.frame_var.set(self.frame); self.draw()
        self.root.after(33, self.tick)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="auto")
    parser.add_argument("--group", choices=("quiet", "complex", "violent"), default="violent")
    parser.add_argument("--trajectory", type=int)
    parser.add_argument("--steps", type=int, default=32,
                        help="rollout length; 32 matches the training curriculum, 100 is stress-test only")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    requested = Path(args.data_root)
    if args.data_root != "auto" and requested.drive and not Path(requested.drive + "\\").exists():
        raise SystemExit(
            f"데이터 드라이브 {requested.drive}가 연결되어 있지 않습니다. "
            "Samsung T7을 연결하거나 --data-root에 실제 WaterKnowsAnswer_Phase3 경로를 지정하세요."
        )
    root = resolve_data_root(args.data_root)
    if args.trajectory is None:
        results = json.loads((root / "rollouts" / "ablation_results.json").read_text(encoding="utf-8"))
        trajectory = int(results["representatives"][args.group])
    else:
        trajectory = args.trajectory
    cache = generate_rollout(root, trajectory, args.steps, Phase3Config(), args.regenerate)
    scene = Scene(cache)
    if args.check:
        print(json.dumps({"cache": str(cache), "frames": int(scene.predicted.shape[0]),
                          "particles": int(scene.predicted.shape[1]), "finite": bool(np.isfinite(scene.predicted).all()),
                          "max_roi": int(scene.roi_count.max()), "final_rmse": float(scene.rmse[-1]),
                          "final_penetration_rate": float(scene.penetration[-1])}, indent=2))
        return
    window = tk.Tk(); App(window, scene); window.mainloop()


if __name__ == "__main__":
    main()
