"""이중 임계값 분류 + 히스테리시스 + 전파 마진(dilation) + 갱신 주기.

상태 결정 규칙 요약 (config.py 표 참고):
  - SPLASH: D_splash가 진입 임계값을 넘으면 진입, 이탈 임계값 아래로 내려가면 이탈.
    그 사이 구간은 현재 SPLASH였으면 유지, 아니었으면 진입하지 않음 (self-hysteresis).
  - STREAM: A_stream(+밀도 조건)으로 진입, A_stream이 이탈 임계값 아래면 이탈. 마찬가지 방식.
  - POOL: S_pool 원시 신호가 4프레임 연속 True여야 진입, False가 한 프레임이라도 나오면 즉시 이탈.
  - 위 세 조건 중 아무것도 "활성"이 아니면 STREAM으로 폴백한다 (기본 유동 상태).
  - 동시에 여러 조건이 활성이면 SPLASH > STREAM > POOL 우선순위로 최종 상태를 정한다.
"""

import taichi as ti

import config as cfg


@ti.data_oriented
class Router:
    def __init__(self, grid, features):
        self.grid = grid
        self.features = features
        self.particles = features.particles
        self.res = grid.res
        self.num_cells = grid.num_cells

        self.cell_state = ti.field(dtype=ti.i32, shape=self.num_cells)
        self.new_state = ti.field(dtype=ti.i32, shape=self.num_cells)
        self.new_state_dilated = ti.field(dtype=ti.i32, shape=self.num_cells)
        self.is_boundary = ti.field(dtype=ti.i32, shape=self.num_cells)

        self.pool_streak_true = ti.field(dtype=ti.i32, shape=self.num_cells)
        self.pool_streak_false = ti.field(dtype=ti.i32, shape=self.num_cells)

        # 5단계 블렌딩을 위한 전환 기록
        self.transition_frame = ti.field(dtype=ti.i32, shape=self.num_cells)
        self.blend_from_state = ti.field(dtype=ti.i32, shape=self.num_cells)

        # 로깅/시각화용 전환 카운터
        self.transition_count = ti.field(dtype=ti.i32, shape=self.num_cells)       # 누적
        self.recent_transition_count = ti.field(dtype=ti.i32, shape=self.num_cells)  # 윈도우 내 (깜빡임 감지)

        if cfg.DILATION_NEIGHBORHOOD == 26:
            self._dilation_offsets = [
                (dx, dy, dz)
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                for dz in (-1, 0, 1)
                if not (dx == 0 and dy == 0 and dz == 0)
            ]
        else:
            self._dilation_offsets = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    @ti.kernel
    def init_states(self):
        for c in range(self.num_cells):
            self.cell_state[c] = cfg.STATE_STREAM
            self.new_state[c] = cfg.STATE_STREAM
            self.new_state_dilated[c] = cfg.STATE_STREAM
            self.blend_from_state[c] = cfg.STATE_STREAM
            self.transition_frame[c] = -10 * cfg.BLEND_WINDOW_K  # 시작부터 블렌딩이 걸리지 않도록 충분히 과거
            self.transition_count[c] = 0
            self.recent_transition_count[c] = 0
            self.pool_streak_true[c] = 0
            self.pool_streak_false[c] = 0

    # ----- 4.1 POOL 연속 프레임 스트릭 (원시 S_pool 신호는 매 프레임 갱신) -----
    @ti.kernel
    def update_pool_streaks(self):
        for c in range(self.num_cells):
            if self.features.S_pool[c] == 1:
                self.pool_streak_true[c] += 1
                self.pool_streak_false[c] = 0
            else:
                self.pool_streak_false[c] += 1
                self.pool_streak_true[c] = 0

    # ----- 4.1/4.2 이중 임계값 + 히스테리시스 + 우선순위 -----
    @ti.func
    def desired_state(self, c):
        cur = self.cell_state[c]
        d = self.features.D_splash[c]
        a = self.features.A_stream[c]
        nd = self.features.N_density[c]

        splash_ok = False
        if d > cfg.SPLASH_ENTER_THRESHOLD:
            splash_ok = True
        elif d >= cfg.SPLASH_EXIT_THRESHOLD:
            splash_ok = (cur == cfg.STATE_SPLASH)

        # D_splash alone often reacts one frame too late at a solid boundary.
        # Mark fast downward flow in the bottom grid layer as impact SPLASH so
        # the splash solver is active before enforce_bounds absorbs the hit.
        cy = (c // self.res[0]) % self.res[1]
        if (cy == 0 and self.grid.cell_particle_count[c] > 0 and
                self.features.vbar[c][1] < -cfg.IMPACT_SPLASH_MIN_DOWNWARD_SPEED):
            splash_ok = True

        stream_ok = False
        if a > cfg.STREAM_ALIGN_ENTER_THRESHOLD and nd < cfg.STREAM_DENSITY_HIGH_THRESHOLD:
            stream_ok = True
        elif a >= cfg.STREAM_ALIGN_EXIT_THRESHOLD:
            stream_ok = (cur == cfg.STATE_STREAM)

        pool_ok = False
        if cur == cfg.STATE_POOL:
            pool_ok = self.pool_streak_false[c] < cfg.POOL_EXIT_STREAK
        else:
            pool_ok = self.pool_streak_true[c] >= cfg.POOL_ENTER_STREAK

        result = cfg.STATE_STREAM  # 아무 조건도 활성이 아니면 기본 유동 상태로 폴백
        if splash_ok:
            result = cfg.STATE_SPLASH
        elif stream_ok:
            result = cfg.STATE_STREAM
        elif pool_ok:
            result = cfg.STATE_POOL
        return result

    @ti.kernel
    def classify_all_cells(self):
        for c in range(self.num_cells):
            self.new_state[c] = self.desired_state(c)

    @ti.kernel
    def compute_boundary_mask(self):
        for cx, cy, cz in ti.ndrange(self.res[0], self.res[1], self.res[2]):
            lin = self.grid.cell_linear(ti.Vector([cx, cy, cz]))
            s = self.cell_state[lin]
            boundary = 0
            for o in ti.static([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]):
                nx, ny, nz = cx + o[0], cy + o[1], cz + o[2]
                if self.grid.in_bounds(nx, ny, nz):
                    nlin = self.grid.cell_linear(ti.Vector([nx, ny, nz]))
                    if self.cell_state[nlin] != s:
                        boundary = 1
            self.is_boundary[lin] = boundary

    @ti.kernel
    def classify_boundary_cells_only(self):
        for c in range(self.num_cells):
            if self.is_boundary[c] == 1:
                self.new_state[c] = self.desired_state(c)
            else:
                self.new_state[c] = self.cell_state[c]

    # ----- 4.3 전파 마진 (SPLASH dilation) -----
    @ti.kernel
    def dilate_splash(self):
        for c in range(self.num_cells):
            self.new_state_dilated[c] = self.new_state[c]
        for cx, cy, cz in ti.ndrange(self.res[0], self.res[1], self.res[2]):
            lin = self.grid.cell_linear(ti.Vector([cx, cy, cz]))
            if self.new_state[lin] == cfg.STATE_SPLASH:
                for o in ti.static(self._dilation_offsets):
                    nx, ny, nz = cx + o[0], cy + o[1], cz + o[2]
                    if self.grid.in_bounds(nx, ny, nz):
                        nlin = self.grid.cell_linear(ti.Vector([nx, ny, nz]))
                        # Dilation marks physically disturbed occupied cells,
                        # not empty space or calm water. This prevents one
                        # impact cell from turning the whole body into SPLASH.
                        if (self.grid.cell_particle_count[nlin] > 0 and
                                self.features.D_splash[nlin] >= cfg.DILATION_MIN_D_SPLASH):
                            self.new_state_dilated[nlin] = cfg.STATE_SPLASH

    # ----- 상태 확정 + 전환 기록 -----
    @ti.kernel
    def apply_transition(self, frame: ti.i32):
        for c in range(self.num_cells):
            old = self.cell_state[c]
            new = self.new_state_dilated[c]
            if new != old:
                self.transition_frame[c] = frame
                self.transition_count[c] += 1
                self.recent_transition_count[c] += 1
                self.blend_from_state[c] = old
            self.cell_state[c] = new

    @ti.kernel
    def propagate_state_to_particles(self):
        for i in range(self.particles.n):
            lin = self.grid.cell_linear(self.particles.cell_index[i])
            self.particles.state[i] = self.cell_state[lin]

    @ti.kernel
    def reset_recent_transition_counts(self):
        for c in range(self.num_cells):
            self.recent_transition_count[c] = 0

    # ----- 4.4 갱신 주기 -----
    def update(self, frame: int):
        self.update_pool_streaks()
        if frame % cfg.FULL_RECLASSIFY_PERIOD == 0:
            self.classify_all_cells()
        else:
            self.compute_boundary_mask()
            self.classify_boundary_cells_only()
        self.dilate_splash()
        self.apply_transition(frame)
        self.propagate_state_to_particles()
