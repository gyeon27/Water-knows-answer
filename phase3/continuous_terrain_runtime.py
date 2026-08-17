"""Continuous height-map -> SWE -> 3-D SPLASH ROI PI-GNN game-runtime prototype."""

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

from phase2.gnn.runtime import terrain_sample
from phase2.shallow_water import FluxParticleEmitter, ShallowWaterSolver, TerrainData

from .config import Phase3Config, resolve_data_root
from .data import radius_graph
from .evaluation import _load_model


DT = 1.0 / 30.0
RADIUS = 0.32


def coarse_swe_terrain(source: TerrainData, stride: int) -> TerrainData:
    """Keep full terrain for collision/rendering, downsample only the 2-D SWE grid."""
    stride = max(1, int(stride))
    rows = np.unique(np.append(np.arange(0, source.height.shape[0], stride), source.height.shape[0] - 1))
    cols = np.unique(np.append(np.arange(0, source.height.shape[1], stride), source.height.shape[1] - 1))
    return TerrainData(
        height=source.height[np.ix_(rows, cols)],
        cliff=source.cliff[np.ix_(rows, cols)],
        channel=source.channel[np.ix_(rows, cols)],
        source=source.source[np.ix_(rows, cols)],
        dx=source.width_m / max(cols.size - 1, 1),
        dz=source.length_m / max(rows.size - 1, 1),
        width_m=source.width_m, length_m=source.length_m,
        source_flow_m3s=source.source_flow_m3s,
        source_velocity_xz=source.source_velocity_xz,
    )


def wcsph_statistics(dataset_dir: Path) -> dict[str, np.ndarray]:
    """Fixed train-domain statistics; never recomputed from runtime predictions."""
    paths = sorted(dataset_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no WCSPH training trajectories in {dataset_dir}")
    velocities, accelerations = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            v = data["velocities"].astype(np.float32)
            active = data["active_mask"].astype(bool)
            dt = float(data["dt"])
        displacement = v * dt
        velocities.append(displacement[active])
        dv = np.diff(displacement, axis=0)
        accelerations.append(dv[active[1:] & active[:-1]])
    v = np.concatenate(velocities); a = np.concatenate(accelerations)
    return {
        "vel_mean": v.mean(0).astype(np.float32),
        "vel_std": np.maximum(v.std(0), 1e-5).astype(np.float32),
        "acc_mean": a.mean(0).astype(np.float32),
        "acc_std": np.maximum(a.std(0), 1e-5).astype(np.float32),
    }


class ContinuousWaterfall:
    def __init__(self, terrain_dir: Path, data_root: Path, max_particles: int = 4000,
                 particle_mass: float = 4.0, swe_stride: int = 4, mode: str = "ours",
                 gnn_interval: int = 2):
        if mode not in {"simple", "ours"}:
            raise ValueError(f"unsupported continuous mode: {mode}")
        self.mode = mode
        self.terrain = TerrainData.load(terrain_dir)
        self.base_flow = self.terrain.source_flow_m3s
        self.swe = ShallowWaterSolver(coarse_swe_terrain(self.terrain, swe_stride), initial_depth_m=0.03)
        self.initial_swe_volume = self.swe.volume
        self.emitter = FluxParticleEmitter(particle_mass_kg=particle_mass, seed=20260809)
        self.mass, self.max_particles = float(particle_mass), int(max_particles)
        self.position = np.zeros((max_particles, 3), np.float32)
        self.velocity = np.zeros_like(self.position)
        self.history = np.zeros((5, max_particles, 3), np.float32)
        self.active = np.zeros(max_particles, bool)
        self.state = np.zeros(max_particles, np.uint8)
        self.splash_timer = np.zeros(max_particles, np.float32)
        self.gnn_correction = np.zeros_like(self.position)
        self.gnn_interval = max(1, int(gnn_interval))
        self.step_index = 0
        self.age = np.zeros(max_particles, np.float32)
        self.rng = np.random.default_rng(20260809)
        self.recycled = self.emitted = 0
        self.flow_multiplier = 1.8
        self.stats = wcsph_statistics(Path(__file__).resolve().parents[1] / "phase2" / "datasets" / "wcsph_pi")
        cfg = Phase3Config()
        self.device = torch.device("cuda")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for continuous PI-GNN runtime")
        self.model = _load_model(data_root, "wcsph_zero_shot", cfg, self.device) if mode == "ours" else None
        self.bounds = np.asarray((
            (-self.terrain.width_m / 2, self.terrain.width_m / 2),
            (float(self.terrain.height.min()), float(self.terrain.height.max() + 8.0)),
            (0.0, self.terrain.length_m),
        ), np.float32)
        self.roi_count = 0

    def _spawn(self, position: np.ndarray, velocity: np.ndarray) -> None:
        free = np.flatnonzero(~self.active)
        count = min(free.size, position.shape[0])
        if not count:
            return
        ids = free[:count]
        self.position[ids] = position[:count]
        self.velocity[ids] = velocity[:count]
        self.history[:, ids] = velocity[:count]
        self.active[ids] = True; self.age[ids] = 0.0; self.state[ids] = 0
        self.splash_timer[ids] = 0.0
        self.gnn_correction[ids] = 0.0
        self.emitted += count

    def _features(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = self.position[ids]
        displacement_history = self.history[:, ids] * DT
        normalized = ((displacement_history - self.stats["vel_mean"]) /
                      self.stats["vel_std"]).transpose(1, 0, 2).reshape(ids.size, 15)
        distances = np.clip(np.concatenate((p - self.bounds[:, 0], self.bounds[:, 1] - p), axis=1) / RADIUS, -1, 1)
        onehot = np.eye(3, dtype=np.float32)[np.minimum(self.state[ids], 2)]
        gravity = np.tile((0.0, -1.0, 0.0), (ids.size, 1)).astype(np.float32)
        node = np.concatenate((normalized, distances, onehot, gravity), axis=1).astype(np.float32)
        edge_index, edge = radius_graph(p, RADIUS, 48)
        return node, edge_index, edge

    @torch.inference_mode()
    def _gnn_acceleration(self, ids: np.ndarray) -> np.ndarray:
        if not ids.size:
            return np.empty((0, 3), np.float32)
        node, edge_index, edge = self._features(ids)
        with torch.autocast("cuda", dtype=torch.float16):
            value = self.model(
                torch.from_numpy(node).to(self.device),
                torch.zeros(ids.size, dtype=torch.int64, device=self.device),
                torch.from_numpy(edge).to(self.device),
                torch.from_numpy(edge_index).long().to(self.device),
            )
        return (value.float().cpu().numpy() * self.stats["acc_std"] + self.stats["acc_mean"])

    def _absorb(self, ids: np.ndarray) -> None:
        if not ids.size:
            return
        p, v = self.position[ids], self.velocity[ids]
        swe_terrain = self.swe.terrain
        cols = np.clip(((p[:, 0] + swe_terrain.width_m * .5) / swe_terrain.dx).astype(int), 0, self.swe.h.shape[1] - 1)
        rows = np.clip((p[:, 2] / swe_terrain.dz).astype(int), 0, self.swe.h.shape[0] - 1)
        depth = (self.mass / 1000.0) / (swe_terrain.dx * swe_terrain.dz)
        np.add.at(self.swe.h, (rows, cols), depth)
        np.add.at(self.swe.hu, (rows, cols), depth * v[:, 0])
        np.add.at(self.swe.hv, (rows, cols), depth * v[:, 2])
        # The volume previously left SWE through waterfall_volume.  Returning
        # settled 3-D particles makes that transfer internal again.
        self.swe.waterfall_volume -= ids.size * self.mass / 1000.0
        self.active[ids] = False
        self.gnn_correction[ids] = 0.0
        self.recycled += ids.size

    def step(self, dt: float = DT) -> None:
        self.step_index += 1
        object.__setattr__(self.swe.terrain, "source_flow_m3s", self.base_flow * self.flow_multiplier)
        for flux in self.swe.advance(dt):
            batch = self.emitter.emit(flux, flux.duration_s, spacing_m=0.12)
            self._spawn(batch.position.astype(np.float32), batch.velocity.astype(np.float32))
        ids = np.flatnonzero(self.active)
        if not ids.size:
            return
        self.history[:-1, ids] = self.history[1:, ids]
        self.history[-1, ids] = self.velocity[ids]
        self.splash_timer[ids] = np.maximum(0.0, self.splash_timer[ids] - dt)
        bed, normal, slope, cliff = terrain_sample(self.terrain, self.position[ids])
        clearance = self.position[ids, 1] - bed
        speed = np.linalg.norm(self.velocity[ids], axis=1)
        approach = np.sum(self.velocity[ids] * normal, axis=1)
        # PI-GNN is an impact/splitting corrector, not a replacement for the
        # ballistic waterfall.  Selecting the cliff lip itself caused learned
        # tangential acceleration to accumulate into long rigid rays.
        splash = ((clearance < .55) & (approach < -.10) & (speed > .60)) | ((clearance < .12) & (speed > 1.0))
        pool = (~splash) & (clearance < .10) & (speed < .5) & (slope < .20)
        recent_splash = self.splash_timer[ids] > 0.0
        self.state[ids] = 0; self.state[ids[pool]] = 2; self.state[ids[splash | recent_splash]] = 1

        next_v = (self.velocity[ids] + np.array((0, -9.81 * dt, 0), np.float32)) * np.exp(-.08 * dt)
        roi_ids = ids[splash]
        self.roi_count = roi_ids.size
        if roi_ids.size and self.mode == "ours":
            if self.step_index % self.gnn_interval == 1 % self.gnn_interval:
                learned_acc = self._gnn_acceleration(roi_ids)
                # Learned output is displacement acceleration.  Clamp only
                # catastrophic out-of-domain spikes; normal WCSPH values pass.
                magnitude = np.linalg.norm(learned_acc, axis=1)
                learned_acc *= np.minimum(1.0, 0.12 / np.maximum(magnitude, 1e-8))[:, None]
                self.gnn_correction[roi_ids] = learned_acc
            learned_acc = self.gnn_correction[roi_ids]
            learned_v = (self.velocity[roi_ids] * dt + learned_acc) / dt
            # Domain-robust residual blend: analytic gravity remains dominant
            # and the network supplies only a local impact correction.
            next_v[splash] = .80 * next_v[splash] + .20 * learned_v
            next_v[splash, 1] = np.minimum(next_v[splash, 1], 4.0)
        speed_next = np.linalg.norm(next_v, axis=1)
        next_v *= np.minimum(1.0, 12.0 / np.maximum(speed_next, 1e-8))[:, None]
        self.velocity[ids] = next_v
        self.position[ids] += next_v * dt
        self.age[ids] += dt

        bed2, normal2, slope2, _ = terrain_sample(self.terrain, self.position[ids])
        below = self.position[ids, 1] < bed2 + .02
        if np.any(below):
            hit = ids[below]; n = normal2[below]
            self.position[hit, 1] = bed2[below] + .02
            vn = np.sum(self.velocity[hit] * n, axis=1)
            inward = vn < 0
            self.velocity[hit[inward]] -= 1.08 * vn[inward, None] * n[inward]
            self.velocity[hit] *= .82
            impact = np.maximum(-vn, 0.0)
            probability = np.clip(.12 + .055 * impact, .12, .52)
            spray = inward & (impact > 1.4) & (self.rng.random(hit.size) < probability)
            if np.any(spray):
                spray_ids = hit[spray]; spray_n = n[spray]; strength = impact[spray]
                random_direction = self.rng.normal(size=(spray_ids.size, 3)).astype(np.float32)
                tangent = random_direction - np.sum(random_direction * spray_n, axis=1, keepdims=True) * spray_n
                tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-8)
                current_normal = np.sum(self.velocity[spray_ids] * spray_n, axis=1)
                rebound = strength * self.rng.uniform(.20, .38, spray_ids.size)
                self.velocity[spray_ids] += (rebound - current_normal)[:, None] * spray_n
                self.velocity[spray_ids] += tangent * (strength * self.rng.uniform(.10, .24, spray_ids.size))[:, None]
                self.splash_timer[spray_ids] = self.rng.uniform(.18, .34, spray_ids.size)
                self.state[spray_ids] = 1
            hit_speed = np.linalg.norm(self.velocity[hit], axis=1)
            self.velocity[hit] *= np.minimum(1.0, 12.0 / np.maximum(hit_speed, 1e-8))[:, None]
            self.velocity[hit, 1] = np.minimum(self.velocity[hit, 1], 4.0)
        ids = np.flatnonzero(self.active)
        bed3, _, slope3, _ = terrain_sample(self.terrain, self.position[ids])
        settled = ((self.position[ids, 1] - bed3 < .08) &
                   (np.linalg.norm(self.velocity[ids], axis=1) < .45) &
                   (slope3 < .28) & (self.age[ids] > .25))
        self._absorb(ids[settled])
        ids = np.flatnonzero(self.active)
        p = self.position[ids]
        outside = ((np.abs(p[:, 0]) > self.terrain.width_m * .55) |
                   (p[:, 2] < -.5) | (p[:, 2] > self.terrain.length_m + .5) |
                   (p[:, 1] < self.terrain.height.min() - 1.0) | (self.age[ids] > 12.0))
        self.active[ids[outside]] = False
        self.gnn_correction[ids[outside]] = 0.0
        self.recycled += int(np.sum(outside))


class Camera:
    def __init__(self): self.yaw, self.pitch, self.zoom = -.72, -.43, 36.0
    def transform(self, points, center):
        p = points - center; cy, sy = math.cos(self.yaw), math.sin(self.yaw); cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        x = cy*p[:,0]-sy*p[:,2]; z = sy*p[:,0]+cy*p[:,2]; y=cp*p[:,1]-sp*z; d=sp*p[:,1]+cp*z
        return np.column_stack((x,y,d))


class ContinuousGUI:
    COLORS = ("#29c7ff", "#ff507c", "#36df93")
    def __init__(self, root: tk.Tk, simulation: ContinuousWaterfall):
        self.root, self.sim, self.camera = root, simulation, Camera()
        self.playing, self.last, self.accumulator, self.mouse = True, time.perf_counter(), 0.0, None
        root.title("Continuous Terrain Water · SWE + SPLASH ROI PI-GNN")
        root.geometry("1320x850")
        bar=ttk.Frame(root,padding=7); bar.pack(fill="x")
        self.button=ttk.Button(bar,text="일시정지",command=self.toggle); self.button.pack(side="left")
        self.impact_focus=tk.BooleanVar(value=True)
        self.focus_locked=False
        ttk.Checkbutton(bar,text="충돌부 확대",variable=self.impact_focus,command=self.change_focus).pack(side="left",padx=7)
        ttk.Button(bar,text="전체 지형",command=self.show_all).pack(side="left")
        ttk.Label(bar,text="수원 유량").pack(side="left",padx=(12,3)); self.flow=tk.DoubleVar(value=1.8)
        ttk.Scale(bar,from_=.25,to=2.5,variable=self.flow,command=lambda _=None:self.set_flow()).pack(side="left",fill="x",expand=True)
        self.info=ttk.Label(bar); self.info.pack(side="right")
        self.canvas=tk.Canvas(root,bg="#101820",highlightthickness=0); self.canvas.pack(fill="both",expand=True)
        self.canvas.bind("<Button-1>",self.down); self.canvas.bind("<B1-Motion>",self.drag); self.canvas.bind("<MouseWheel>",self.wheel)
        self.center=np.array((0,(simulation.terrain.height.min()+simulation.terrain.height.max())*.5,simulation.terrain.length_m*.54))
        self.full_center=self.center.copy(); self.camera.zoom=62.0
        self._mesh(); root.after(1,self.tick)
    def _mesh(self):
        t=self.sim.terrain; stride=max(1,max(t.height.shape)//22); rr=np.arange(0,t.height.shape[0],stride); cc=np.arange(0,t.height.shape[1],stride)
        rr=np.unique(np.append(rr,t.height.shape[0]-1)); cc=np.unique(np.append(cc,t.height.shape[1]-1)); xx,zz=np.meshgrid(cc*t.dx-t.width_m*.5,rr*t.dz)
        self.vertices=np.column_stack((xx.ravel(),t.height[np.ix_(rr,cc)].ravel(),zz.ravel())); w=cc.size; faces=[]
        for r in range(rr.size-1):
            for c in range(cc.size-1): a=r*w+c; faces.extend(((a,a+w,a+1),(a+1,a+w,a+w+1)))
        self.faces=np.asarray(faces)
    def screen(self,p):
        q=self.camera.transform(p,self.center); w,h=max(self.canvas.winfo_width(),2),max(self.canvas.winfo_height(),2)
        return q,np.column_stack((w*.5+q[:,0]*self.camera.zoom,h*.52-q[:,1]*self.camera.zoom))
    def draw(self, full=False):
        if self.impact_focus.get() and not self.focus_locked:
            splash=np.flatnonzero(self.sim.active & (self.sim.state==1))
            if splash.size:
                p=self.sim.position[splash]
                low=p[:,1] <= np.quantile(p[:,1],.45)
                target=np.median(p[low] if np.any(low) else p,axis=0)
                self.center=target; self.focus_locked=True; full=True
        if full or not self.canvas.find_withtag("terrain"):
            self.canvas.delete("all"); q,xy=self.screen(self.vertices); depth=q[self.faces,2].mean(1)
            for f in np.argsort(depth)[::-1]: self.canvas.create_polygon(*xy[self.faces[f]].reshape(-1).tolist(),fill="#48565a",outline="#526166",tags="terrain")
        else:
            self.canvas.delete("water")
        ids=np.flatnonzero(self.sim.active)
        if ids.size>900:
            splash_ids=ids[self.sim.state[ids]==1]
            ordinary_ids=ids[self.sim.state[ids]!=1]
            if splash_ids.size>320: splash_ids=splash_ids[np.linspace(0,splash_ids.size-1,320,dtype=int)]
            remaining=max(0,900-splash_ids.size)
            if ordinary_ids.size>remaining: ordinary_ids=ordinary_ids[np.linspace(0,ordinary_ids.size-1,remaining,dtype=int)]
            ids=np.concatenate((ordinary_ids,splash_ids))
        if ids.size:
            q,xy=self.screen(self.sim.position[ids])
            for j in np.argsort(q[:,2])[::-1]:
                i=ids[j]; x,y=xy[j]; color=self.COLORS[min(int(self.sim.state[i]),2)]; r=4.8 if self.sim.state[i]==1 else 2.8
                if self.sim.state[i]==1:
                    self.canvas.create_oval(x-7,y-7,x+7,y+7,outline="#ffd7e4",width=1,tags="water")
                self.canvas.create_oval(x-r,y-r,x+r,y+r,fill=color,outline="#f5ffff" if self.sim.state[i]==1 else color,width=1,tags="water")
        self.info.configure(text=f"t={self.sim.swe.time:6.1f}s · 3D {self.sim.active.sum()} · SPLASH {np.sum(self.sim.active & (self.sim.state==1))} · GNN ROI {self.sim.roi_count} · 재사용 {self.sim.recycled}")
    def set_flow(self): self.sim.flow_multiplier=float(self.flow.get())
    def change_focus(self):
        self.camera.zoom=62.0 if self.impact_focus.get() else 36.0
        self.focus_locked=False
        if not self.impact_focus.get(): self.center=self.full_center.copy(); self.focus_locked=True
        self.draw(full=True)
    def show_all(self):
        self.impact_focus.set(False); self.focus_locked=True; self.center=self.full_center.copy(); self.camera.zoom=36.0; self.draw(full=True)
    def toggle(self): self.playing=not self.playing; self.button.configure(text="일시정지" if self.playing else "재생")
    def down(self,e): self.mouse=(e.x,e.y)
    def drag(self,e):
        if self.mouse: self.camera.yaw+=(e.x-self.mouse[0])*.008; self.camera.pitch=float(np.clip(self.camera.pitch+(e.y-self.mouse[1])*.008,-1.45,1.45)); self.mouse=(e.x,e.y); self.draw(full=True)
    def wheel(self,e): self.camera.zoom=float(np.clip(self.camera.zoom*(1.12 if e.delta>0 else .89),10,100)); self.draw(full=True)
    def tick(self):
        now=time.perf_counter(); self.accumulator+=min(now-self.last,.15); self.last=now
        if self.playing:
            steps=min(int(self.accumulator/DT),3)
            for _ in range(steps): self.sim.step(DT); self.accumulator-=DT
            if steps: self.draw()
        self.root.after(8,self.tick)


def main():
    project=Path(__file__).resolve().parents[1]; parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain",type=Path,default=project/"phase2"/"terrains"/"natural_waterfall")
    parser.add_argument("--data-root",default="auto"); parser.add_argument("--max-particles",type=int,default=4000)
    parser.add_argument("--swe-stride",type=int,default=4,help="SWE-only terrain downsample stride; collision/rendering stay full resolution")
    parser.add_argument("--check-steps",type=int,default=0,help="run headless fixed steps and print runtime state")
    parser.add_argument("--mode",choices=("simple","ours"),default="ours")
    parser.add_argument("--gnn-interval",type=int,default=2,help="refresh SPLASH PI-GNN correction every N physics frames")
    parser.add_argument("--benchmark-modes",action="store_true",help="run Simple-3D and Ours with identical seeds")
    parser.add_argument("--benchmark-output",type=Path,default=project/"phase3"/"results_summary"/"continuous_runtime_benchmark.json")
    args=parser.parse_args(); data_root=resolve_data_root(args.data_root)
    if args.benchmark_modes:
        import time
        rows=[]
        steps=args.check_steps if args.check_steps else 3000
        for mode in ("simple","ours"):
            sim=ContinuousWaterfall(args.terrain,data_root,args.max_particles,swe_stride=args.swe_stride,
                                    mode=mode,gnn_interval=args.gnn_interval)
            try:
                import psutil
                process=psutil.Process()
            except ImportError:
                process=None
            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter(); peak_active=0; peak_speed=0.0; energy_peak=0.0
            active_samples=penetration_samples=0; min_clearance=float("inf"); rss_peak=0; frame_ms=[]
            for step_index in range(steps):
                frame_start=time.perf_counter(); sim.step(DT)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                if step_index >= min(30,steps//10): frame_ms.append((time.perf_counter()-frame_start)*1000.0)
                ids=np.flatnonzero(sim.active); peak_active=max(peak_active,int(ids.size))
                if ids.size:
                    speed=np.linalg.norm(sim.velocity[ids],axis=1); peak_speed=max(peak_speed,float(speed.max()))
                    bed,_,_,_=terrain_sample(sim.terrain,sim.position[ids])
                    clearance=sim.position[ids,1]-bed
                    active_samples+=int(ids.size); penetration_samples+=int(np.sum(clearance < -1e-5))
                    min_clearance=min(min_clearance,float(clearance.min()))
                    energy=np.mean(.5*speed*speed+9.81*np.maximum(sim.position[ids,1]-bed,0.0))
                    energy_peak=max(energy_peak,float(energy))
                if process is not None and step_index%10==0: rss_peak=max(rss_peak,int(process.memory_info().rss))
            elapsed=time.perf_counter()-start
            injected=float(sim.swe.injected_volume)
            mass_error=float(sim.swe.mass_balance_error(sim.initial_swe_volume))
            rows.append({"mode":mode,"steps":steps,"simulated_seconds":steps*DT,"wall_seconds":elapsed,
                         "simulation_fps":steps/elapsed,"realtime_factor":steps*DT/elapsed,
                         "mean_frame_ms":float(np.mean(frame_ms)),"p95_frame_ms":float(np.quantile(frame_ms,.95)),
                         "active_final":int(sim.active.sum()),"active_peak":peak_active,
                         "emitted":sim.emitted,"recycled":sim.recycled,
                         "particle_balance_error":int(sim.emitted-sim.recycled-sim.active.sum()),
                         "injected_volume_m3":injected,"swe_mass_error_m3":mass_error,
                         "swe_mass_relative_error":abs(mass_error)/max(injected,1e-12),
                         "penetration_rate":penetration_samples/max(active_samples,1),
                         "minimum_clearance_m":min_clearance if np.isfinite(min_clearance) else None,
                         "max_speed_mps":peak_speed,"peak_mean_mechanical_energy_jpkg":energy_peak,
                         "peak_process_memory_mb":rss_peak/2**20 if rss_peak else None,
                         "peak_cuda_memory_mb":torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else None,
                         "finite":bool(np.isfinite(sim.position[sim.active]).all() and np.isfinite(sim.velocity[sim.active]).all())})
        args.benchmark_output.parent.mkdir(parents=True,exist_ok=True)
        args.benchmark_output.write_text(json.dumps({"dt":DT,"same_seed":20260809,"rows":rows},ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps({"output":str(args.benchmark_output),"rows":rows},ensure_ascii=False,indent=2)); return
    sim=ContinuousWaterfall(args.terrain,data_root,args.max_particles,swe_stride=args.swe_stride,
                            mode=args.mode,gnn_interval=args.gnn_interval)
    if args.check_steps:
        for _ in range(args.check_steps): sim.step(DT)
        print(json.dumps({"steps":args.check_steps,"time_s":sim.swe.time,"active_3d":int(sim.active.sum()),
                          "emitted":sim.emitted,"recycled":sim.recycled,"gnn_roi":sim.roi_count,
                          "finite":bool(np.isfinite(sim.position[sim.active]).all() and np.isfinite(sim.velocity[sim.active]).all()),
                          "max_speed_mps":float(np.linalg.norm(sim.velocity[sim.active],axis=1).max()) if np.any(sim.active) else 0.0,
                          "max_upward_speed_mps":float(sim.velocity[sim.active,1].max()) if np.any(sim.active) else 0.0,
                          "swe_mass_error_m3":sim.swe.mass_balance_error(sim.initial_swe_volume)},ensure_ascii=False,indent=2)); return
    root=tk.Tk(); ContinuousGUI(root,sim); root.mainloop()


if __name__ == "__main__": main()
