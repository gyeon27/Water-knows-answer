"""STREAM 솔버: 간이 SPH (밀도 추정 + 상태방정식 압력 + 점성력).

질량-스프링 모델(고정된 "자연 길이"로 돌아가려는 탄성력)은 고체 역학 모델이라 압력/밀도
개념이 없어서, 물기둥이 낙하하는 동안 탄성 고체(젤리)처럼 부풀고 튕기는 문제가 있었다.
대신 압력 기반의 최소 SPH(Müller et al. 2003 스타일)를 쓴다:
  1) compute_density: 이웃 STREAM 입자와의 커널 가중합으로 밀도를 추정
  2) apply_forces: 기준밀도 대비 편차로 압력을 구해 압력기울기 힘 + 점성력을 가하고 적분

SPH 압력힘은 이 파이프라인의 프레임 간격(1/60초)에 비해 뻣뻣해서(stiff), 한 번에
1/60초씩 적분하면 불안정해진다(중력만으로는 나올 수 없는 속도가 관측됨). 그래서 한 프레임을
STREAM_SPH_SUBSTEPS개의 작은 스텝으로 쪼개 적분한다 — 실제 SPH 구현들이 흔히 쓰는 방식이다.
"""

import math

import taichi as ti

import config as cfg


@ti.data_oriented
class StreamSolver:
    def __init__(self, grid, particles, router):
        self.grid = grid
        self.particles = particles
        self.router = router
        self.sph_radius = cfg.stream_particle_spacing(particles.n) * cfg.STREAM_SPH_RADIUS_FACTOR
        self.rest_density = cfg.density_rate_ref(particles.n)
        self.density = ti.field(dtype=ti.f32, shape=particles.n)
        # 후보 출력 버퍼: 이 솔버가 담당했다면 나올 다음 위치/속도 (blending.py에서 사용).
        # 서브스텝 사이의 "현재" 작업 버퍼로도 재사용한다.
        self.out_position = ti.Vector.field(3, dtype=ti.f32, shape=particles.n)
        self.out_velocity = ti.Vector.field(3, dtype=ti.f32, shape=particles.n)
        # 서브스텝 결과를 쓰는 버퍼. out_position/out_velocity를 읽으면서 동시에 쓰면
        # (병렬 for에서) 경쟁 상태가 생기므로 이중 버퍼링한다.
        self._next_position = ti.Vector.field(3, dtype=ti.f32, shape=particles.n)
        self._next_velocity = ti.Vector.field(3, dtype=ti.f32, shape=particles.n)

    # ----- SPH 커널 (Müller et al. 2003) -----
    @ti.func
    def poly6(self, r, h):
        result = 0.0
        if 0.0 <= r < h:
            term = h * h - r * r
            result = (315.0 / (64.0 * math.pi * h**9)) * term * term * term
        return result

    @ti.func
    def spiky_grad(self, diff, r, h):
        # r이 아주 작은(거의 붙어있는) 입자쌍에서는 (h-r)^2가 최댓값(h^2)에 가까워져
        # 힘이 극단적으로 커진다. 초기 배치가 완벽한 격자가 아니라 지터를 준 것이라
        # 이런 근접쌍이 실제로 생기므로, 힘 계산에 쓰는 유효 거리에 하한을 둬서
        # (방향 벡터 diff/r 자체는 그대로 두고) 폭발적인 힘만 막는다.
        result = ti.Vector([0.0, 0.0, 0.0])
        if 1e-6 < r < h:
            r_eff = ti.max(r, cfg.STREAM_SPH_MIN_DIST_RATIO * h)
            coeff = -45.0 / (math.pi * h**6) * (h - r_eff) * (h - r_eff)
            result = coeff * (diff / r)
        return result

    @ti.func
    def visc_laplacian(self, r, h):
        result = 0.0
        if 0.0 <= r < h:
            result = 45.0 / (math.pi * h**6) * (h - r)
        return result

    @ti.kernel
    def _init_work_buffers(self):
        for i in range(self.particles.n):
            self.out_position[i] = self.particles.position[i]
            self.out_velocity[i] = self.particles.velocity[i]

    @ti.func
    def is_active(self, i, frame):
        active = self.particles.state[i] == cfg.STATE_STREAM
        lin = self.grid.cell_linear(self.particles.cell_index[i])
        frames_since = frame - self.router.transition_frame[lin]
        if (0 <= frames_since < cfg.BLEND_WINDOW_K and
                self.router.blend_from_state[lin] == cfg.STATE_STREAM):
            active = True
        return active

    @ti.kernel
    def compute_density(self, frame: ti.i32):
        h = self.sph_radius
        for i in range(self.particles.n):
            if self.is_active(i, frame):
                xi = self.out_position[i]
                c = self.particles.cell_index[i]
                rho = 0.0
                for dx, dy, dz in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
                    nx, ny, nz = c[0] + dx, c[1] + dy, c[2] + dz
                    if self.grid.in_bounds(nx, ny, nz):
                        nlin = self.grid.cell_linear(ti.Vector([nx, ny, nz]))
                        start = self.grid.cell_particle_offset[nlin]
                        cnt = self.grid.cell_particle_count[nlin]
                        for k in range(cnt):
                            j = self.grid.particle_id_sorted[start + k]
                            if self.is_active(j, frame):
                                r = (xi - self.out_position[j]).norm()
                                rho += self.particles.mass[j] * self.poly6(r, h)
                self.density[i] = rho
            else:
                self.density[i] = self.rest_density

    @ti.kernel
    def apply_forces(self, dt: ti.f32, frame: ti.i32):
        h = self.sph_radius
        g = ti.Vector([0.0, cfg.GRAVITY, 0.0])
        rho0 = self.rest_density
        warmup = ti.min(1.0, ti.cast(frame + 1, ti.f32) / cfg.STREAM_SPH_WARMUP_FRAMES)
        # Smoothstep avoids an abrupt pressure impulse at startup.
        warmup = warmup * warmup * (3.0 - 2.0 * warmup)
        k_stiff = cfg.STREAM_SPH_STIFFNESS * warmup
        mu = cfg.STREAM_SPH_VISCOSITY
        for i in range(self.particles.n):
            xi = self.out_position[i]
            vi = self.out_velocity[i]
            rho_i = ti.max(self.density[i], 1e-3)
            # 압력을 0 이상으로 클램프한다: 표면장력 항이 없는 상태에서 기준밀도보다 희박한
            # (자유표면/경계) 입자에 음압을 허용하면 p/rho^2 항이 저밀도에서 극단적으로 커져
            # 폭발적인 힘이 생긴다. 표준 SPH free-surface 처리 관례대로 음압은 0으로 자른다.
            p_i = ti.max(0.0, k_stiff * (rho_i - rho0))

            if self.is_active(i, frame):
                f_pressure = ti.Vector([0.0, 0.0, 0.0])
                f_visc = ti.Vector([0.0, 0.0, 0.0])
                f_surface = ti.Vector([0.0, 0.0, 0.0])
                c = self.particles.cell_index[i]
                for dx, dy, dz in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
                    nx, ny, nz = c[0] + dx, c[1] + dy, c[2] + dz
                    if self.grid.in_bounds(nx, ny, nz):
                        nlin = self.grid.cell_linear(ti.Vector([nx, ny, nz]))
                        start = self.grid.cell_particle_offset[nlin]
                        cnt = self.grid.cell_particle_count[nlin]
                        for k in range(cnt):
                            j = self.grid.particle_id_sorted[start + k]
                            if j != i and self.is_active(j, frame):
                                xj = self.out_position[j]
                                diff = xi - xj
                                r = diff.norm()
                                if 1e-6 < r < h:
                                    rho_j = ti.max(self.density[j], 1e-3)
                                    p_j = ti.max(0.0, k_stiff * (rho_j - rho0))
                                    grad = self.spiky_grad(diff, r, h)
                                    f_pressure += (
                                        -self.particles.mass[j] * (p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j)) * grad
                                    )
                                    vj = self.out_velocity[j]
                                    f_visc += mu * self.particles.mass[j] * (vj - vi) / rho_j * self.visc_laplacian(r, h)
                                    if rho_i < rho0 * cfg.STREAM_SURFACE_DENSITY_RATIO:
                                        # Weak cohesion only on under-dense
                                        # surface particles; interior pressure
                                        # remains unaffected.
                                        q = (h - r) / h
                                        f_surface += (
                                            cfg.STREAM_SURFACE_TENSION
                                            * self.particles.mass[j]
                                            * q * q
                                            * (-diff / r)
                                        )

                accel = g + (f_pressure + f_visc + f_surface) / self.particles.mass[i]
                accel_norm = accel.norm()
                if accel_norm > cfg.STREAM_MAX_ACCELERATION:
                    accel *= cfg.STREAM_MAX_ACCELERATION / accel_norm
                v1 = vi + accel * dt
                self._next_position[i] = xi + v1 * dt
                self._next_velocity[i] = v1
            else:
                self._next_position[i] = xi
                self._next_velocity[i] = vi

    def step(self, dt, frame):
        self._init_work_buffers()
        sub_dt = dt / cfg.STREAM_SPH_SUBSTEPS
        for _ in range(cfg.STREAM_SPH_SUBSTEPS):
            self.compute_density(frame)
            self.apply_forces(sub_dt, frame)
            self.out_position.copy_from(self._next_position)
            self.out_velocity.copy_from(self._next_velocity)
