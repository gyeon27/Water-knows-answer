"""POOL 솔버: 2D 하이트필드(격자의 x-z 평면) 위에서 얕은 물 방정식을 유한차분으로 근사.

  d h/dt + div(h*u) = 0
  d u/dt + (u . grad)u = -g * grad h

POOL 입자들의 높이/속도를 하이트필드에 투영 -> 방정식 업데이트 -> 입자 위치 재투영.
"""

import taichi as ti

import config as cfg


@ti.data_oriented
class PoolSolver:
    def __init__(self, grid, particles):
        self.grid = grid
        self.particles = particles
        self.rx, self.rz = grid.res[0], grid.res[2]
        self.cell_size = grid.cell_size

        self.h = ti.field(dtype=ti.f32, shape=(self.rx, self.rz))
        self.h_new = ti.field(dtype=ti.f32, shape=(self.rx, self.rz))
        self.u = ti.Vector.field(2, dtype=ti.f32, shape=(self.rx, self.rz))  # (u_x, u_z)
        self.u_new = ti.Vector.field(2, dtype=ti.f32, shape=(self.rx, self.rz))

        self.proj_h_sum = ti.field(dtype=ti.f32, shape=(self.rx, self.rz))
        self.proj_u_sum = ti.Vector.field(2, dtype=ti.f32, shape=(self.rx, self.rz))
        self.proj_count = ti.field(dtype=ti.i32, shape=(self.rx, self.rz))

        self.out_position = ti.Vector.field(3, dtype=ti.f32, shape=particles.n)
        self.out_velocity = ti.Vector.field(3, dtype=ti.f32, shape=particles.n)

    @ti.kernel
    def init_buffers(self):
        for x, z in ti.ndrange(self.rx, self.rz):
            self.h[x, z] = 0.0
            self.u[x, z] = ti.Vector([0.0, 0.0])

    @ti.kernel
    def clear_projection(self):
        for x, z in ti.ndrange(self.rx, self.rz):
            self.proj_h_sum[x, z] = 0.0
            self.proj_u_sum[x, z] = ti.Vector([0.0, 0.0])
            self.proj_count[x, z] = 0

    @ti.kernel
    def project_particles(self):
        for i in range(self.particles.n):
            if self.particles.state[i] == cfg.STATE_POOL:
                c = self.particles.cell_index[i]
                depth = ti.max(self.particles.position[i][1] - cfg.POOL_FLOOR_Y, 0.0)
                v = self.particles.velocity[i]
                self.proj_h_sum[c[0], c[2]] += depth
                self.proj_u_sum[c[0], c[2]] += ti.Vector([v[0], v[2]])
                self.proj_count[c[0], c[2]] += 1

    @ti.kernel
    def commit_projection(self):
        for x, z in ti.ndrange(self.rx, self.rz):
            cnt = self.proj_count[x, z]
            if cnt > 0:
                self.h[x, z] = self.proj_h_sum[x, z] / cnt
                self.u[x, z] = self.proj_u_sum[x, z] / cnt
            # 입자가 없는(dry) 셀은 이전 값을 유지한 채 아래 파동 방정식으로 계속 전파된다.

    @ti.kernel
    def advance(self, dt: ti.f32):
        g_mag = -cfg.GRAVITY
        cs2 = 2.0 * self.cell_size
        for x, z in ti.ndrange(self.rx, self.rz):
            xm = ti.max(x - 1, 0)
            xp = ti.min(x + 1, self.rx - 1)
            zm = ti.max(z - 1, 0)
            zp = ti.min(z + 1, self.rz - 1)

            h_c = self.h[x, z]
            hu_xp = self.h[xp, z] * self.u[xp, z][0]
            hu_xm = self.h[xm, z] * self.u[xm, z][0]
            hu_zp = self.h[x, zp] * self.u[x, zp][1]
            hu_zm = self.h[x, zm] * self.u[x, zm][1]
            div_hu = (hu_xp - hu_xm) / cs2 + (hu_zp - hu_zm) / cs2
            h_new = ti.max(h_c - dt * div_hu, 0.0)

            u_c = self.u[x, z]
            dh_dx = (self.h[xp, z] - self.h[xm, z]) / cs2
            dh_dz = (self.h[x, zp] - self.h[x, zm]) / cs2
            du_dx = (self.u[xp, z] - self.u[xm, z]) / cs2
            du_dz = (self.u[x, zp] - self.u[x, zm]) / cs2
            advect = u_c[0] * du_dx + u_c[1] * du_dz
            pressure = ti.Vector([g_mag * dh_dx, g_mag * dh_dz])
            u_new = (u_c + dt * (-advect - pressure)) * cfg.POOL_VELOCITY_DAMPING

            self.h_new[x, z] = h_new
            self.u_new[x, z] = u_new

        for x, z in ti.ndrange(self.rx, self.rz):
            self.h[x, z] = self.h_new[x, z]
            self.u[x, z] = self.u_new[x, z]

    @ti.kernel
    def reproject_to_particles(self, dt: ti.f32):
        for i in range(self.particles.n):
            c = self.particles.cell_index[i]
            depth = self.h[c[0], c[2]]
            uxz = self.u[c[0], c[2]]
            pos = self.particles.position[i]
            target_y = cfg.POOL_FLOOR_Y + ti.max(depth, cfg.POOL_MIN_DEPTH)
            height_alpha = 1.0 - ti.exp(-cfg.POOL_HEIGHT_RELAXATION_RATE * dt)
            new_y = pos[1] + height_alpha * (target_y - pos[1])
            self.out_position[i] = ti.Vector([pos[0] + uxz[0] * dt, new_y, pos[2] + uxz[1] * dt])
            self.out_velocity[i] = ti.Vector([uxz[0], (new_y - pos[1]) / dt, uxz[1]])

    def step(self, dt):
        self.clear_projection()
        self.project_particles()
        self.commit_projection()
        self.advance(dt)
        self.reproject_to_particles(dt)
