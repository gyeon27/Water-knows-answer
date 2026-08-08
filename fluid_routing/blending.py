"""전환 경계 블렌딩.

상태가 바뀐 셀은 transition_frame부터 K 프레임 동안 이전 솔버(blend_from_state)와
신규 솔버(cell_state) 결과를 선형 보간한다:

    w(t) = min(1, (t - transition_frame) / K)
    x_blend = (1 - w) * x_prev_solver + w * x_new_solver

세 솔버 모두 매 프레임 전체 입자에 대해 후보 출력을 계산해두므로(각 solvers/*.py의
out_position/out_velocity), 여기서는 입자가 속한 셀의 현재 상태(및 전환 시점 이전 상태)에
맞는 두 후보를 골라 섞기만 하면 된다.
"""

import taichi as ti

import config as cfg


@ti.data_oriented
class Blender:
    def __init__(self, grid, particles, router, stream_solver, pool_solver, splash_solver):
        self.grid = grid
        self.particles = particles
        self.router = router
        self.stream_solver = stream_solver
        self.pool_solver = pool_solver
        self.splash_solver = splash_solver

    @ti.func
    def solver_output(self, state_id, i):
        pos = self.particles.position[i]
        vel = self.particles.velocity[i]
        if state_id == cfg.STATE_STREAM:
            pos = self.stream_solver.out_position[i]
            vel = self.stream_solver.out_velocity[i]
        elif state_id == cfg.STATE_SPLASH:
            pos = self.splash_solver.out_position[i]
            vel = self.splash_solver.out_velocity[i]
        elif state_id == cfg.STATE_POOL:
            pos = self.pool_solver.out_position[i]
            vel = self.pool_solver.out_velocity[i]
        return pos, vel

    @ti.kernel
    def commit(self, frame: ti.i32):
        for i in range(self.particles.n):
            lin = self.grid.cell_linear(self.particles.cell_index[i])
            cur_state = self.particles.state[i]
            new_pos, new_vel = self.solver_output(cur_state, i)

            frames_since = frame - self.router.transition_frame[lin]
            if 0 <= frames_since < cfg.BLEND_WINDOW_K:
                prev_state = self.router.blend_from_state[lin]
                old_pos, old_vel = self.solver_output(prev_state, i)
                w = ti.min(1.0, frames_since / cfg.BLEND_WINDOW_K)
                new_pos = (1.0 - w) * old_pos + w * new_pos
                new_vel = (1.0 - w) * old_vel + w * new_vel

            self.particles.position[i] = new_pos
            self.particles.velocity[i] = new_vel
