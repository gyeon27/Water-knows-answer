"""입자 상태: 위치, 속도, 질량, 상태 라벨, 소속 격자 셀."""

import taichi as ti

import config as cfg


def _lattice_dims(n, size):
    """물기둥 부피/입자수에 맞는 격자 칸 수(nx, ny, nz)를 구한다 (초기 배치용)."""
    sx, sy, sz = size
    spacing = (sx * sy * sz / n) ** (1 / 3)
    nx = max(1, round(sx / spacing))
    ny = max(1, round(sy / spacing))
    nz = max(1, round(sz / spacing))
    dims = [nx, ny, nz]
    longest_axis = max(range(3), key=lambda axis: size[axis])
    while dims[0] * dims[1] * dims[2] < n:
        dims[longest_axis] += 1
    nx, ny, nz = dims
    return nx, ny, nz


@ti.data_oriented
class Particles:
    def __init__(self, n: int = cfg.N_PARTICLES):
        self.n = n

        self.position = ti.Vector.field(3, dtype=ti.f32, shape=n)
        self.velocity = ti.Vector.field(3, dtype=ti.f32, shape=n)
        self.mass = ti.field(dtype=ti.f32, shape=n)
        self.state = ti.field(dtype=ti.i32, shape=n)  # 0=STREAM, 1=SPLASH, 2=POOL
        self.cell_index = ti.Vector.field(3, dtype=ti.i32, shape=n)  # 소속 격자 셀

        col_size = tuple(b - a for a, b in zip(cfg.COLUMN_MIN, cfg.COLUMN_MAX))
        self._lattice_nx, self._lattice_ny, self._lattice_nz = _lattice_dims(n, col_size)

    @ti.kernel
    def init_water_column(self):
        """물기둥 낙하 초기 씬: 지터를 준 격자(jittered lattice)로 입자를 배치한다.

        순수 무작위(uniform random) 배치는 일부 입자쌍이 우연히 극도로 가깝게 생성될 수 있어,
        SPH 압력힘(stream_solver.py)이 첫 프레임부터 폭발적인 힘을 내는 원인이 됐다
        (중력만으로는 나올 수 없는 속도가 관측됨). 최소 간격이 보장되는 격자 위에 작은
        지터만 주면 이 문제가 없어진다 — 실제 SPH 구현들도 보통 이렇게 초기화한다.
        """
        col_min = ti.Vector(cfg.COLUMN_MIN)
        col_size = ti.Vector(cfg.COLUMN_MAX) - col_min
        nx, ny, nz = self._lattice_nx, self._lattice_ny, self._lattice_nz
        dims = ti.Vector([float(nx), float(ny), float(nz)])
        for i in range(self.n):
            ix = i % nx
            iy = (i // nx) % ny
            iz = i // (nx * ny)
            idx = ti.Vector([ti.cast(ix, ti.f32), ti.cast(iy, ti.f32), ti.cast(iz, ti.f32)])
            cell_center = (idx + 0.5) / dims
            jitter = (ti.Vector([ti.random(), ti.random(), ti.random()]) - 0.5) / dims * cfg.LATTICE_JITTER_FRACTION
            r = cell_center + jitter
            self.position[i] = col_min + r * col_size
            self.velocity[i] = ti.Vector([0.0, 0.0, 0.0])
            self.mass[i] = 1.0
            self.state[i] = cfg.STATE_STREAM
            self.cell_index[i] = ti.Vector([0, 0, 0])

    @ti.kernel
    def enforce_bounds(self):
        """도메인 경계를 벗어난 입자를 감쇠 반발시켜 안쪽으로 되돌린다."""
        lo = ti.Vector(cfg.DOMAIN_MIN)
        hi = ti.Vector(cfg.DOMAIN_MAX)
        for i in range(self.n):
            p = self.position[i]
            v = self.velocity[i]
            for d in ti.static(range(3)):
                if p[d] < lo[d]:
                    p[d] = lo[d]
                    if v[d] < 0.0:
                        v[d] = -v[d] * cfg.BOUNDARY_RESTITUTION
                elif p[d] > hi[d]:
                    p[d] = hi[d]
                    if v[d] > 0.0:
                        v[d] = -v[d] * cfg.BOUNDARY_RESTITUTION
            self.position[i] = p
            self.velocity[i] = v
