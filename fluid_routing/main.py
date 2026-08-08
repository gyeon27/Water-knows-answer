"""진입점: 시뮬레이션 루프 + Taichi GGUI 실시간 3D 시각화.

표시 요소:
  - 입자를 상태별 색으로 표시 (STREAM=파랑, SPLASH=빨강, POOL=초록)
  - 프레임별 상태 분포 비율(%) 텍스트 오버레이
  - 자주 상태가 바뀌는 셀(깜빡임 의심 지점)은 노란색으로 하이라이트
"""

import argparse
import time

import numpy as np
import taichi as ti

import config as cfg
from particles import Particles
from grid import Grid
from features import Features
from routing import Router
from blending import Blender
from logging_utils import Logger
from solvers.stream_solver import StreamSolver
from solvers.pool_solver import PoolSolver
from solvers.splash_solver import SplashSolver


@ti.data_oriented
class Renderer:
    """입자 색상 계산만 담당 (상태별 색 + 깜빡임 의심 셀 하이라이트)."""

    def __init__(self, particles, grid, router):
        self.particles = particles
        self.grid = grid
        self.router = router
        self.colors = ti.Vector.field(3, dtype=ti.f32, shape=particles.n)

    @ti.kernel
    def compute_colors(self):
        for i in range(self.particles.n):
            lin = self.grid.cell_linear(self.particles.cell_index[i])
            s = self.particles.state[i]
            col = ti.Vector([0.2, 0.45, 0.9])  # STREAM = 파랑
            if s == cfg.STATE_SPLASH:
                col = ti.Vector([0.9, 0.2, 0.2])  # 빨강
            elif s == cfg.STATE_POOL:
                col = ti.Vector([0.2, 0.8, 0.35])  # 초록
            if self.router.recent_transition_count[lin] >= cfg.FLICKER_COUNT_THRESHOLD:
                col = ti.Vector([1.0, 0.9, 0.1])  # 깜빡임 의심 -> 노랑으로 덮어씀
            self.colors[i] = col


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: 위상 기반 적응형 솔버 라우팅 (신경망 없음)")
    parser.add_argument("--headless", action="store_true", help="GGUI 창 없이 실행 (로깅/검증용)")
    parser.add_argument("--max-frames", type=int, default=0, help="N프레임 후 종료 (0=계속 실행)")
    parser.add_argument("--arch", choices=["cpu", "gpu", "vulkan"], default="gpu")
    parser.add_argument("--n-particles", type=int, default=cfg.N_PARTICLES)
    return parser.parse_args()


def count_states(state_np):
    return {
        cfg.STATE_STREAM: int(np.sum(state_np == cfg.STATE_STREAM)),
        cfg.STATE_SPLASH: int(np.sum(state_np == cfg.STATE_SPLASH)),
        cfg.STATE_POOL: int(np.sum(state_np == cfg.STATE_POOL)),
    }


def main():
    args = parse_args()
    arch_map = {"cpu": ti.cpu, "gpu": ti.gpu, "vulkan": ti.vulkan}
    ti.init(arch=arch_map[args.arch])

    particles = Particles(n=args.n_particles)
    grid = Grid(particles)
    features = Features(grid, particles)
    router = Router(grid, features)
    stream_solver = StreamSolver(grid, particles, router)
    pool_solver = PoolSolver(grid, particles)
    splash_solver = SplashSolver(grid, particles, router)
    blender = Blender(grid, particles, router, stream_solver, pool_solver, splash_solver)
    logger = Logger()
    renderer = Renderer(particles, grid, router)

    def reset_sim():
        particles.init_water_column()
        router.init_states()
        features.init_buffers()
        pool_solver.init_buffers()

    reset_sim()

    window = canvas = scene = camera = None
    if not args.headless:
        window = ti.ui.Window("Fluid Routing - Phase 1 (no GNN)", (1280, 720), vsync=True)
        canvas = window.get_canvas()
        scene = window.get_scene()
        camera = ti.ui.Camera()
        center = [(a + b) / 2 for a, b in zip(cfg.DOMAIN_MIN, cfg.DOMAIN_MAX)]
        camera.position(center[0] + 6.0, center[1] + 4.0, center[2] + 7.0)
        camera.lookat(*center)
        camera.up(0, 1, 0)

    def render(overlay_lines):
        renderer.compute_colors()
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.6, 0.6, 0.6))
        scene.point_light(pos=(4, 8, 4), color=(1, 1, 1))
        scene.particles(particles.position, radius=0.035, per_vertex_color=renderer.colors)
        canvas.scene(scene)
        gui = window.get_gui()
        with gui.sub_window("state distribution / legend", 0.02, 0.02, 0.36, 0.38):
            for line in overlay_lines:
                gui.text(line)
        window.show()

    frame = 0
    paused = False
    try:
        while True:
            if window is not None and not window.running:
                break
            if args.max_frames and frame >= args.max_frames:
                break

            # --- GUI 전용: 리셋/일시정지 컨트롤 (헤드리스 모드에는 없음) ---
            if window is not None:
                if window.is_pressed("r") or window.is_pressed("R"):
                    reset_sim()
                    frame = 0
                    logger.close()
                    logger = Logger()
                if window.is_pressed(" "):
                    paused = not paused
                    # 스페이스바를 누르고 있는 동안 매 프레임 토글되는 것을 막기 위해
                    # 한 번 처리하고 나면 키를 뗄 때까지 잠시 대기한다.
                    while window.running and window.is_pressed(" "):
                        window.show()

            if paused:
                if window is not None:
                    render([f"frame: {frame} (PAUSED)", "Space: 재생/일시정지   R: 리셋(리플레이)"])
                continue

            t_frame_start = time.perf_counter()
            t0 = time.perf_counter()

            # 1) 격자 바인딩
            grid.build()
            # 2) 특징량 계산
            features.update(frame)
            if frame > 0 and frame % cfg.FLICKER_WINDOW_FRAMES == 0:
                router.reset_recent_transition_counts()
            # 3) 라우팅 (히스테리시스 + dilation + 갱신 주기)
            router.update(frame)
            # 4) 상태별 솔버 (모든 입자에 대해 후보 출력 계산 -> 블렌딩에서 선택/보간)
            stream_solver.step(cfg.DT, frame)
            pool_solver.step(cfg.DT)
            splash_solver.step(cfg.DT, frame)
            # 5) 경계 블렌딩으로 최종 위치/속도 확정
            blender.commit(frame)
            particles.enforce_bounds()

            frame_time_ms = (time.perf_counter() - t0) * 1000.0
            state_np = particles.state.to_numpy()
            counts = count_states(state_np)
            logger.log_frame(frame, counts, frame_time_ms)

            if window is not None:
                total = max(1, sum(counts.values()))
                render(
                    [
                        f"frame: {frame}",
                        f"STREAM: {100 * counts[cfg.STATE_STREAM] / total:.1f}%",
                        f"SPLASH: {100 * counts[cfg.STATE_SPLASH] / total:.1f}%",
                        f"POOL:   {100 * counts[cfg.STATE_POOL] / total:.1f}%",
                        "",
                        "Color legend:",
                        "Blue  = STREAM (falling / flowing water)",
                        "Red   = SPLASH (impact / droplets)",
                        "Green = POOL (settled water)",
                        "Yellow = frequent state changes (flicker)",
                        f"physics: {frame_time_ms:.2f} ms",
                        f"total(+render/vsync): {(time.perf_counter() - t_frame_start) * 1000.0:.2f} ms",
                        "Space: 일시정지   R: 리셋(리플레이)",
                    ]
                )

            frame += 1
    finally:
        logger.write_summary(router.transition_count.to_numpy(), frame)
        logger.close()


if __name__ == "__main__":
    main()
