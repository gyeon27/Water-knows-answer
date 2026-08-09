"""Taichi WCSPH backend for offline 3D waterfall teacher trajectories."""

import math

import numpy as np
import taichi as ti


_TAICHI_READY = False


def ensure_taichi() -> None:
    global _TAICHI_READY
    if not _TAICHI_READY:
        ti.init(arch=ti.cpu, default_fp=ti.f32, offline_cache=False)
        _TAICHI_READY = True


@ti.data_oriented
class WCSPHSolver:
    """Dynamic-particle WCSPH with a counting-sort neighbor grid."""

    def __init__(
        self,
        terrain_height: np.ndarray,
        dx: float,
        dz: float,
        width_m: float,
        length_m: float,
        max_particles: int = 4096,
        particle_mass_kg: float = 2.0,
        smoothing_radius_m: float = 0.28,
        air_drag_rate: float = 0.08,
        restitution: float = 0.18,
    ):
        ensure_taichi()
        self.max_particles = int(max_particles)
        self.mass = float(particle_mass_kg)
        self.radius = float(smoothing_radius_m)
        self.air_drag_rate = float(air_drag_rate)
        self.restitution = float(restitution)
        self.dx, self.dz = float(dx), float(dz)
        self.width_m, self.length_m = float(width_m), float(length_m)
        self.max_y = float(np.max(terrain_height) + 4.0)
        self.grid_res = (
            max(1, math.ceil(width_m / self.radius)),
            max(1, math.ceil(self.max_y / self.radius)),
            max(1, math.ceil(length_m / self.radius)),
        )
        self.num_cells = math.prod(self.grid_res)
        self.position = ti.Vector.field(3, ti.f32, shape=self.max_particles)
        self.velocity = ti.Vector.field(3, ti.f32, shape=self.max_particles)
        self.density = ti.field(ti.f32, shape=self.max_particles)
        self.pressure = ti.field(ti.f32, shape=self.max_particles)
        self.active = ti.field(ti.i32, shape=self.max_particles)
        self.age = ti.field(ti.f32, shape=self.max_particles)
        self.cell_index = ti.field(ti.i32, shape=self.max_particles)
        self.cell_count = ti.field(ti.i32, shape=self.num_cells)
        self.cell_offset = ti.field(ti.i32, shape=self.num_cells)
        self.cell_cursor = ti.field(ti.i32, shape=self.num_cells)
        self.sorted_id = ti.field(ti.i32, shape=self.max_particles)
        self.terrain = ti.field(ti.f32, shape=terrain_height.shape)
        self.terrain.from_numpy(terrain_height.astype(np.float32))
        self.terrain_rows, self.terrain_cols = terrain_height.shape
        self._clear_particles()

    @ti.kernel
    def _clear_particles(self):
        for i in range(self.max_particles):
            self.active[i] = 0
            self.age[i] = 0.0
            self.position[i] = ti.Vector([0.0, 0.0, 0.0])
            self.velocity[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def activate(
        self,
        ids: ti.types.ndarray(dtype=ti.i32, ndim=1),
        positions: ti.types.ndarray(dtype=ti.f32, ndim=2),
        velocities: ti.types.ndarray(dtype=ti.f32, ndim=2),
        count: ti.i32,
    ):
        for k in range(count):
            i = ids[k]
            self.position[i] = ti.Vector([positions[k, 0], positions[k, 1], positions[k, 2]])
            self.velocity[i] = ti.Vector([velocities[k, 0], velocities[k, 1], velocities[k, 2]])
            self.active[i] = 1
            self.age[i] = 0.0

    @ti.func
    def _coord(self, p):
        gx = ti.max(0, ti.min(self.grid_res[0] - 1, ti.cast((p[0] + 0.5 * self.width_m) / self.radius, ti.i32)))
        gy = ti.max(0, ti.min(self.grid_res[1] - 1, ti.cast(p[1] / self.radius, ti.i32)))
        gz = ti.max(0, ti.min(self.grid_res[2] - 1, ti.cast(p[2] / self.radius, ti.i32)))
        return ti.Vector([gx, gy, gz])

    @ti.func
    def _linear(self, c):
        return c[0] + self.grid_res[0] * (c[1] + self.grid_res[1] * c[2])

    @ti.func
    def _inside_grid(self, c):
        return 0 <= c[0] < self.grid_res[0] and 0 <= c[1] < self.grid_res[1] and 0 <= c[2] < self.grid_res[2]

    @ti.kernel
    def _clear_grid(self):
        for c in range(self.num_cells):
            self.cell_count[c] = 0

    @ti.kernel
    def _count(self):
        for i in range(self.max_particles):
            if self.active[i] == 1:
                lin = self._linear(self._coord(self.position[i]))
                self.cell_index[i] = lin
                ti.atomic_add(self.cell_count[lin], 1)

    @ti.kernel
    def _prefix(self):
        ti.loop_config(serialize=True)
        total = 0
        for c in range(self.num_cells):
            self.cell_offset[c] = total
            self.cell_cursor[c] = total
            total += self.cell_count[c]

    @ti.kernel
    def _scatter(self):
        for i in range(self.max_particles):
            if self.active[i] == 1:
                index = ti.atomic_add(self.cell_cursor[self.cell_index[i]], 1)
                self.sorted_id[index] = i

    @ti.func
    def _poly6(self, r):
        value = 0.0
        if r < self.radius:
            term = self.radius * self.radius - r * r
            value = 315.0 / (64.0 * math.pi * self.radius**9) * term**3
        return value

    @ti.func
    def _spiky_gradient(self, diff, r):
        result = ti.Vector([0.0, 0.0, 0.0])
        if 1e-5 < r < self.radius:
            result = -45.0 / (math.pi * self.radius**6) * (self.radius - r) ** 2 * diff / r
        return result

    @ti.func
    def _viscosity_laplacian(self, r):
        result = 0.0
        if r < self.radius:
            result = 45.0 / (math.pi * self.radius**6) * (self.radius - r)
        return result

    @ti.kernel
    def _density_pressure(self):
        for i in range(self.max_particles):
            if self.active[i] == 1:
                xi = self.position[i]
                base = self._coord(xi)
                rho = 0.0
                for ox, oy, oz in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
                    cell = base + ti.Vector([ox, oy, oz])
                    if self._inside_grid(cell):
                        lin = self._linear(cell)
                        for cursor in range(self.cell_count[lin]):
                            j = self.sorted_id[self.cell_offset[lin] + cursor]
                            rho += self.mass * self._poly6((xi - self.position[j]).norm())
                self.density[i] = ti.max(rho, 1.0)
                self.pressure[i] = ti.max(0.0, 900.0 * (self.density[i] / 1000.0 - 1.0))

    @ti.func
    def _bed_height(self, x, z):
        col = ti.max(0.0, ti.min(self.terrain_cols - 1.001, (x + 0.5 * self.width_m) / self.dx))
        row = ti.max(0.0, ti.min(self.terrain_rows - 1.001, z / self.dz))
        c0, r0 = ti.cast(ti.floor(col), ti.i32), ti.cast(ti.floor(row), ti.i32)
        c1, r1 = ti.min(c0 + 1, self.terrain_cols - 1), ti.min(r0 + 1, self.terrain_rows - 1)
        tx, tz = col - c0, row - r0
        return (
            self.terrain[r0, c0] * (1.0 - tx) * (1.0 - tz)
            + self.terrain[r0, c1] * tx * (1.0 - tz)
            + self.terrain[r1, c0] * (1.0 - tx) * tz
            + self.terrain[r1, c1] * tx * tz
        )

    @ti.kernel
    def _integrate(self, dt: ti.f32, max_age: ti.f32):
        for i in range(self.max_particles):
            if self.active[i] == 1:
                xi, vi = self.position[i], self.velocity[i]
                rhoi = ti.max(self.density[i], 1.0)
                pressure_force = ti.Vector([0.0, 0.0, 0.0])
                viscosity_force = ti.Vector([0.0, 0.0, 0.0])
                base = self._coord(xi)
                for ox, oy, oz in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
                    cell = base + ti.Vector([ox, oy, oz])
                    if self._inside_grid(cell):
                        lin = self._linear(cell)
                        for cursor in range(self.cell_count[lin]):
                            j = self.sorted_id[self.cell_offset[lin] + cursor]
                            if j != i:
                                diff = xi - self.position[j]
                                r = diff.norm()
                                if 1e-5 < r < self.radius:
                                    rhoj = ti.max(self.density[j], 1.0)
                                    pressure_force += -self.mass * (self.pressure[i] + self.pressure[j]) / (2.0 * rhoj) * self._spiky_gradient(diff, r)
                                    viscosity_force += 0.08 * self.mass * (self.velocity[j] - vi) / rhoj * self._viscosity_laplacian(r)
                accel = ti.Vector([0.0, -9.81, 0.0]) + pressure_force / rhoi + viscosity_force
                accel_norm = accel.norm()
                if accel_norm > 120.0:
                    accel *= 120.0 / accel_norm
                vi += accel * dt
                vi *= ti.exp(-self.air_drag_rate * dt)
                xi += vi * dt
                bed = self._bed_height(xi[0], xi[2]) + 0.015
                if xi[1] < bed:
                    epsx, epsz = self.dx, self.dz
                    nx = -(self._bed_height(xi[0] + epsx, xi[2]) - self._bed_height(xi[0] - epsx, xi[2])) / (2.0 * epsx)
                    nz = -(self._bed_height(xi[0], xi[2] + epsz) - self._bed_height(xi[0], xi[2] - epsz)) / (2.0 * epsz)
                    normal = ti.Vector([nx, 1.0, nz]).normalized()
                    vn = vi.dot(normal)
                    if vn < 0.0:
                        vi -= (1.0 + self.restitution) * vn * normal
                    vi *= 0.985
                    xi[1] = bed
                self.age[i] += dt
                if (
                    ti.abs(xi[0]) > 0.5 * self.width_m
                    or xi[2] < 0.0
                    or xi[2] > self.length_m
                    or xi[1] > self.max_y
                    or self.age[i] > max_age
                ):
                    self.active[i] = 0
                self.position[i], self.velocity[i] = xi, vi

    def build_grid(self) -> None:
        self._clear_grid()
        self._count()
        self._prefix()
        self._scatter()

    def step(self, dt: float, substeps: int = 3, max_age: float = 4.0) -> None:
        sub_dt = float(dt) / substeps
        for _ in range(substeps):
            self.build_grid()
            self._density_pressure()
            self._integrate(sub_dt, float(max_age))

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.position.to_numpy(), self.velocity.to_numpy(), self.active.to_numpy().astype(bool)
