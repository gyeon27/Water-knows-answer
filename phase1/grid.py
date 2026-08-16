"""성긴 보조 격자 바인딩. 카운팅 정렬 방식으로 cell -> particle 리스트를 구성한다."""

import taichi as ti

import config as cfg


@ti.data_oriented
class Grid:
    def __init__(self, particles):
        self.particles = particles
        self.res = cfg.GRID_RES  # (rx, ry, rz), 파이썬 튜플 (컴파일 타임 상수로 취급됨)
        self.cell_size = cfg.CELL_SIZE
        self.num_cells = self.res[0] * self.res[1] * self.res[2]

        # 카운팅 정렬용 버퍼
        self.cell_particle_count = ti.field(dtype=ti.i32, shape=self.num_cells)
        self.cell_particle_offset = ti.field(dtype=ti.i32, shape=self.num_cells)
        self.cell_current_index = ti.field(dtype=ti.i32, shape=self.num_cells)
        self.particle_id_sorted = ti.field(dtype=ti.i32, shape=particles.n)

    # ----- 격자 셀 좌표/인덱스 유틸 (다른 모듈에서도 호출) -----
    @ti.func
    def cell_coord(self, pos):
        c = ti.floor(pos / self.cell_size, ti.i32)
        c[0] = ti.max(0, ti.min(self.res[0] - 1, c[0]))
        c[1] = ti.max(0, ti.min(self.res[1] - 1, c[1]))
        c[2] = ti.max(0, ti.min(self.res[2] - 1, c[2]))
        return c

    @ti.func
    def cell_linear(self, c):
        return c[0] + self.res[0] * (c[1] + self.res[1] * c[2])

    @ti.func
    def in_bounds(self, cx, cy, cz):
        return 0 <= cx < self.res[0] and 0 <= cy < self.res[1] and 0 <= cz < self.res[2]

    # ----- 카운팅 정렬 파이프라인 -----
    @ti.kernel
    def bind_particles_to_grid(self):
        for i in range(self.particles.n):
            self.particles.cell_index[i] = self.cell_coord(self.particles.position[i])

    @ti.kernel
    def clear_counts(self):
        for c in range(self.num_cells):
            self.cell_particle_count[c] = 0

    @ti.kernel
    def count_particles(self):
        for i in range(self.particles.n):
            lin = self.cell_linear(self.particles.cell_index[i])
            self.cell_particle_count[lin] += 1  # 병렬 for에서 자동으로 atomic add 처리됨

    @ti.kernel
    def prefix_sum_offsets(self):
        ti.loop_config(serialize=True)
        acc = 0
        for c in range(self.num_cells):
            self.cell_particle_offset[c] = acc
            acc += self.cell_particle_count[c]

    @ti.kernel
    def reset_cursor(self):
        for c in range(self.num_cells):
            self.cell_current_index[c] = self.cell_particle_offset[c]

    @ti.kernel
    def scatter_particles(self):
        for i in range(self.particles.n):
            lin = self.cell_linear(self.particles.cell_index[i])
            idx = ti.atomic_add(self.cell_current_index[lin], 1)
            self.particle_id_sorted[idx] = i

    def build(self):
        """매 프레임 호출: 입자를 셀에 바인딩하고 셀->입자 정렬 리스트를 구성한다."""
        self.bind_particles_to_grid()
        self.clear_counts()
        self.count_particles()
        self.prefix_sum_offsets()
        self.reset_cursor()
        self.scatter_particles()
