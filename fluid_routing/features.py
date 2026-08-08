"""셀 단위 4개 특징량: 수직 정렬도(A_stream), 이웃 밀집도(N_density),
속도장 발산+밀도 변화율(D_splash), 정체도(S_pool)."""

import taichi as ti

import config as cfg


@ti.data_oriented
class Features:
    def __init__(self, grid, particles):
        self.grid = grid
        self.particles = particles
        self.res = grid.res
        self.num_cells = grid.num_cells
        self.cell_size = grid.cell_size
        self.cell_volume = cfg.CELL_VOLUME
        self.density_rate_ref = cfg.density_rate_ref(particles.n)
        self.g_hat = ti.Vector(cfg.G_HAT)

        # 셀 단위 집계량
        self.vbar = ti.Vector.field(3, dtype=ti.f32, shape=self.num_cells)
        self.align_accum = ti.field(dtype=ti.f32, shape=self.num_cells)
        self.mean_height = ti.field(dtype=ti.f32, shape=self.num_cells)

        # 4개 특징량
        self.A_stream = ti.field(dtype=ti.f32, shape=self.num_cells)
        self.N_density = ti.field(dtype=ti.f32, shape=self.num_cells)
        self.divergence = ti.field(dtype=ti.f32, shape=self.num_cells)
        self.D_splash = ti.field(dtype=ti.f32, shape=self.num_cells)
        self.S_pool = ti.field(dtype=ti.i32, shape=self.num_cells)  # 0/1, 이번 프레임 원시 신호

        # 이전 프레임 밀도 (D_splash의 밀도 변화율 항에 필요)
        self.density_prev = ti.field(dtype=ti.f32, shape=self.num_cells)

        # S_pool을 위한 시간창 링버퍼 (셀 좌표가 프레임 간 안정적으로 매핑되므로 셀 ID를 그대로 사용)
        self.speed_ring = ti.field(dtype=ti.f32, shape=(self.num_cells, cfg.POOL_WINDOW_FRAMES))
        self.height_ring = ti.field(dtype=ti.f32, shape=(self.num_cells, cfg.POOL_WINDOW_FRAMES))
        self.occupied_ring = ti.field(dtype=ti.i32, shape=(self.num_cells, cfg.POOL_WINDOW_FRAMES))
        self.frames_recorded = 0  # 파이썬 측 카운터: 링버퍼가 아직 안 찼으면 S_pool은 항상 False

    @ti.kernel
    def _clear_buffers_kernel(self):
        for c in range(self.num_cells):
            self.density_prev[c] = 0.0
            self.S_pool[c] = 0
            for w in range(cfg.POOL_WINDOW_FRAMES):
                self.speed_ring[c, w] = 0.0
                self.height_ring[c, w] = 0.0
                self.occupied_ring[c, w] = 0

    def init_buffers(self):
        self._clear_buffers_kernel()
        self.frames_recorded = 0

    # ----- 3.1, 3.2 정렬도 / 밀집도 -----
    @ti.kernel
    def compute_cell_aggregates(self):
        for c in range(self.num_cells):
            self.vbar[c] = ti.Vector([0.0, 0.0, 0.0])
            self.align_accum[c] = 0.0
            self.mean_height[c] = 0.0

        for i in range(self.particles.n):
            lin = self.grid.cell_linear(self.particles.cell_index[i])
            v = self.particles.velocity[i]
            speed = v.norm()
            align = ti.abs(v.dot(self.g_hat)) / (speed + cfg.EPS)
            self.vbar[lin] += v
            self.align_accum[lin] += align
            self.mean_height[lin] += self.particles.position[i][1]

        for c in range(self.num_cells):
            cnt = self.grid.cell_particle_count[c]
            if cnt > 0:
                self.vbar[c] /= cnt
                self.mean_height[c] /= cnt
                self.A_stream[c] = self.align_accum[c] / cnt
            else:
                self.A_stream[c] = 0.0
            self.N_density[c] = cnt / self.cell_volume

    # ----- 3.3 발산 + 밀도 변화율 -----
    @ti.kernel
    def compute_divergence_and_dsplash(self):
        for cx, cy, cz in ti.ndrange(self.res[0], self.res[1], self.res[2]):
            lin = self.grid.cell_linear(ti.Vector([cx, cy, cz]))
            vk = self.vbar[lin]
            div = 0.0
            # 이 셀 자체가 비어있으면(입자가 없음) 속도가 정의되지 않는다. 예전엔 이웃이
            # 비어있는 경우만 발산 합산에서 제외했는데, 그 반대 경우(이 셀은 비어있고
            # 이웃엔 물이 있는 경우)를 놓쳤었다 - 물기둥이 아직 도달하지 않은 바로 앞/옆의
            # 빈 셀이 "이웃의 실제 속도 vs 자신의 가짜 속도 0"을 비교해 큰 발산을 만들고,
            # 이게 SPLASH로 잘못 분류된 뒤 팽창(dilation)을 통해 진짜 물 입자 쪽으로 번져서
            # 닿기도 전에 물이 확 퍼지는 것처럼 보이는 원인이었다. 빈 셀은 아예 계산하지 않는다.
            if self.grid.cell_particle_count[lin] > 0:
                for o in ti.static([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]):
                    nx, ny, nz = cx + o[0], cy + o[1], cz + o[2]
                    if self.grid.in_bounds(nx, ny, nz):
                        nlin = self.grid.cell_linear(ti.Vector([nx, ny, nz]))
                        # 입자가 없는 빈 이웃 셀도 마찬가지 이유로 발산 합산에서 제외한다.
                        if self.grid.cell_particle_count[nlin] > 0:
                            n_dir = ti.Vector([float(o[0]), float(o[1]), float(o[2])])
                            div += (self.vbar[nlin] - vk).dot(n_dir) / self.cell_size
            self.divergence[lin] = div

            # N_density는 셀 부피로 나눈 입자수 밀도라 절대값 스케일이 매우 크다(수백~수천).
            # "이전 프레임 밀도" 대비 상대 변화율을 쓰면, 이전에 비어있던 셀(밀도 0)이 막 채워질
            # 때 분모가 0에 가까워 변화율이 폭발한다(물이 새 영역으로 흘러들기만 해도 항상
            # SPLASH로 오판되던 원인). 대신 물기둥의 대표 밀도라는 고정 기준값으로 정규화한다.
            density_rate = (self.N_density[lin] - self.density_prev[lin]) / cfg.DT / self.density_rate_ref
            self.D_splash[lin] = ti.abs(div) + cfg.DIVERGENCE_DENSITY_LAMBDA * ti.abs(density_rate)
            if self.grid.cell_particle_count[lin] == 0:
                # 빈 셀은 (물이 막 빠져나가 밀도 변화율만으로도) D_splash가 튈 수 있는데,
                # 입자가 하나도 없는 셀을 SPLASH로 분류해봤자 렌더링될 것도 없고, 팽창(dilation)을
                # 통해 진짜 이웃 셀에만 악영향을 준다. 아예 0으로 눌러 분류 후보에서 제외한다.
                self.D_splash[lin] = 0.0

    @ti.kernel
    def store_density_prev(self):
        for c in range(self.num_cells):
            self.density_prev[c] = self.N_density[c]

    # ----- 3.4 정체도 (시간창) -----
    @ti.kernel
    def _write_ring(self, cursor: ti.i32):
        for c in range(self.num_cells):
            self.speed_ring[c, cursor] = self.vbar[c].norm()
            self.height_ring[c, cursor] = self.mean_height[c]
            self.occupied_ring[c, cursor] = self.grid.cell_particle_count[c] > 0

    @ti.kernel
    def _evaluate_pool_signal(self):
        for c in range(self.num_cells):
            speed_sum = 0.0
            h_min = 1e18
            h_max = -1e18
            occupied_frames = 0
            for w in range(cfg.POOL_WINDOW_FRAMES):
                speed_sum += self.speed_ring[c, w]
                h = self.height_ring[c, w]
                h_min = ti.min(h_min, h)
                h_max = ti.max(h_max, h)
                occupied_frames += self.occupied_ring[c, w]
            avg_speed = speed_sum / cfg.POOL_WINDOW_FRAMES
            height_range = h_max - h_min
            if (occupied_frames == cfg.POOL_WINDOW_FRAMES and
                    avg_speed < cfg.POOL_VEL_THRESHOLD and
                    height_range < cfg.POOL_HEIGHT_THRESHOLD):
                self.S_pool[c] = 1
            else:
                self.S_pool[c] = 0

    @ti.kernel
    def _clear_pool_signal(self):
        for c in range(self.num_cells):
            self.S_pool[c] = 0

    def update_pool_signal(self, frame: int):
        cursor = frame % cfg.POOL_WINDOW_FRAMES
        self._write_ring(cursor)
        self.frames_recorded = min(self.frames_recorded + 1, cfg.POOL_WINDOW_FRAMES)
        if self.frames_recorded >= cfg.POOL_WINDOW_FRAMES:
            self._evaluate_pool_signal()
        else:
            self._clear_pool_signal()

    def update(self, frame: int):
        self.compute_cell_aggregates()
        if frame == 0:
            # 첫 프레임은 비교할 이전 밀도가 없으므로 밀도 변화율 항이 허구로 폭발하지 않도록
            # density_prev를 현재 값으로 초기화해 rate=0에서 시작한다.
            self.store_density_prev()
        self.compute_divergence_and_dsplash()
        self.update_pool_signal(frame)
        self.store_density_prev()
