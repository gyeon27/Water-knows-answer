"""SPLASH 임시 솔버 — 이번 단계에서는 정교할 필요 없음.

단순 중력 + 이웃 입자와의 soft collision(반발력)만 구현한다.
`splash_step_placeholder`는 입력(이전 위치/속도, 이웃 입자 목록) -> 출력(다음 위치/속도)
형태로 시그니처를 명확히 분리해뒀다. 나중에 이 함수의 몸통만 GNN 추론 호출로
교체하면 되고, 클래스 인터페이스(out_position/out_velocity 버퍼, step(dt))는 그대로 유지된다.
"""

import taichi as ti

import config as cfg


@ti.data_oriented
class SplashSolver:
    def __init__(self, grid, particles, router):
        self.grid = grid
        self.particles = particles
        self.router = router
        self.repulsion_radius = cfg.stream_particle_spacing(particles.n) * 1.5
        self.cohesion_radius = cfg.stream_particle_spacing(particles.n) * cfg.SPLASH_COHESION_RADIUS_FACTOR
        self.out_position = ti.Vector.field(3, dtype=ti.f32, shape=particles.n)
        self.out_velocity = ti.Vector.field(3, dtype=ti.f32, shape=particles.n)

    @ti.func
    def splash_step_placeholder(self, i, dt):
        # 입력: 이전 위치/속도(self.particles.position[i]/velocity[i]) + 이웃 입자 목록(격자 조회)
        # 출력: 다음 위치/속도 (x1, v1)
        # 나중에 GNN 추론으로 교체될 자리 — 지금은 중력 + 단순 반발력만 적용.
        x0 = self.particles.position[i]
        v0 = self.particles.velocity[i]

        repulsion = ti.Vector([0.0, 0.0, 0.0])
        cohesion = ti.Vector([0.0, 0.0, 0.0])
        neighbor_count = 0
        c = self.particles.cell_index[i]
        for dx, dy, dz in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
            nx, ny, nz = c[0] + dx, c[1] + dy, c[2] + dz
            if self.grid.in_bounds(nx, ny, nz):
                nlin = self.grid.cell_linear(ti.Vector([nx, ny, nz]))
                start = self.grid.cell_particle_offset[nlin]
                cnt = self.grid.cell_particle_count[nlin]
                for k in range(cnt):
                    j = self.grid.particle_id_sorted[start + k]
                    if j != i:
                        diff = x0 - self.particles.position[j]
                        dist = diff.norm()
                        if 1e-6 < dist < self.repulsion_radius:
                            overlap = self.repulsion_radius - dist
                            repulsion += (diff / dist) * overlap * cfg.SPLASH_REPULSION_STIFFNESS
                            neighbor_count += 1
                        elif self.repulsion_radius <= dist < self.cohesion_radius:
                            stretch = dist - self.repulsion_radius
                            cohesion += (-diff / dist) * stretch * cfg.SPLASH_COHESION_STIFFNESS
                            neighbor_count += 1
        if neighbor_count > 0:
            # 근접 이웃이 매우 많은 밀집 상태에서 힘이 폭발하지 않도록 평균을 낸다
            repulsion /= neighbor_count
            cohesion /= neighbor_count

        g = ti.Vector([0.0, cfg.GRAVITY, 0.0])
        accel = g + (repulsion + cohesion) / self.particles.mass[i]
        v1 = (v0 + accel * dt) * ti.exp(-cfg.SPLASH_VELOCITY_DAMPING_RATE * dt)
        x1 = x0 + v1 * dt
        # Give impact-region particles a small, bounded upward response.  This
        # is intentionally modest: it is a Phase-1 baseline, not a learned
        # splash model.
        if x1[1] < cfg.DOMAIN_MIN[1] and v1[1] < 0.0:
            impact_speed = -v1[1]
            x1[1] = cfg.DOMAIN_MIN[1]
            v1[1] = ti.min(impact_speed * cfg.SPLASH_FLOOR_RESTITUTION, cfg.SPLASH_MAX_UPWARD_SPEED)

            # Convert a small part of the downward momentum into an outward
            # sheet, as seen in WaterDrop-style impact trajectories.
            center_x = 0.5 * (cfg.DOMAIN_MIN[0] + cfg.DOMAIN_MAX[0])
            center_z = 0.5 * (cfg.DOMAIN_MIN[2] + cfg.DOMAIN_MAX[2])
            radial = ti.Vector([x0[0] - center_x, x0[2] - center_z])
            radial_norm = radial.norm()
            if radial_norm < 1e-5:
                angle = ti.cast(i, ti.f32) * 2.399963
                radial = ti.Vector([ti.cos(angle), ti.sin(angle)])
            else:
                radial /= radial_norm
            lateral_kick = ti.min(
                impact_speed * cfg.SPLASH_LATERAL_TRANSFER,
                cfg.SPLASH_MAX_LATERAL_KICK,
            )
            v1[0] += radial[0] * lateral_kick
            v1[2] += radial[1] * lateral_kick
        return x1, v1

    @ti.func
    def is_active(self, i, frame):
        active = self.particles.state[i] == cfg.STATE_SPLASH
        lin = self.grid.cell_linear(self.particles.cell_index[i])
        frames_since = frame - self.router.transition_frame[lin]
        if (0 <= frames_since < cfg.BLEND_WINDOW_K and
                self.router.blend_from_state[lin] == cfg.STATE_SPLASH):
            active = True
        return active

    @ti.kernel
    def step(self, dt: ti.f32, frame: ti.i32):
        for i in range(self.particles.n):
            if self.is_active(i, frame):
                x1, v1 = self.splash_step_placeholder(i, dt)
                self.out_position[i] = x1
                self.out_velocity[i] = v1
            else:
                self.out_position[i] = self.particles.position[i]
                self.out_velocity[i] = self.particles.velocity[i]
