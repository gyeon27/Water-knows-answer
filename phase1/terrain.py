"""Analytic waterfall terrain shared by rendering and particle collision."""

import numpy as np
import taichi as ti

import config as cfg


@ti.data_oriented
class WaterfallTerrain:
    def __init__(self, resolution=65):
        self.resolution = resolution
        vertex_count = resolution * resolution
        triangle_count = (resolution - 1) * (resolution - 1) * 2
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=vertex_count)
        self.colors = ti.Vector.field(3, dtype=ti.f32, shape=vertex_count)
        self.indices = ti.field(dtype=ti.i32, shape=triangle_count * 3)
        self._build_mesh()

    @staticmethod
    def height_numpy(x, z):
        upper = 0.5 * (1.0 - np.tanh((z - cfg.WATERFALL_CLIFF_Z) / cfg.WATERFALL_TRANSITION_WIDTH))
        height = cfg.WATERFALL_LOWER_Y + (cfg.WATERFALL_UPPER_Y - cfg.WATERFALL_LOWER_Y) * upper
        channel = np.exp(-0.5 * ((x - cfg.WATERFALL_CHANNEL_CENTER_X) / 0.65) ** 2)
        return height - 0.08 * channel * upper

    def _build_mesh(self):
        r = self.resolution
        vertices = np.zeros((r * r, 3), dtype=np.float32)
        colors = np.zeros((r * r, 3), dtype=np.float32)
        for iz in range(r):
            z = cfg.DOMAIN_MIN[2] + (cfg.DOMAIN_MAX[2] - cfg.DOMAIN_MIN[2]) * iz / (r - 1)
            for ix in range(r):
                x = cfg.DOMAIN_MIN[0] + (cfg.DOMAIN_MAX[0] - cfg.DOMAIN_MIN[0]) * ix / (r - 1)
                y = float(self.height_numpy(x, z))
                index = iz * r + ix
                vertices[index] = (x, y, z)
                rock_mix = np.clip((y - cfg.WATERFALL_LOWER_Y) / max(cfg.WATERFALL_UPPER_Y, 1e-6), 0.0, 1.0)
                colors[index] = (0.22 + 0.18 * rock_mix, 0.32 + 0.12 * rock_mix, 0.18 + 0.06 * rock_mix)
        indices = []
        for iz in range(r - 1):
            for ix in range(r - 1):
                a = iz * r + ix
                b = a + 1
                c = a + r
                d = c + 1
                indices.extend((a, c, b, b, c, d))
        self.vertices.from_numpy(vertices)
        self.colors.from_numpy(colors)
        self.indices.from_numpy(np.asarray(indices, dtype=np.int32))

    @ti.func
    def height_at(self, x, z):
        upper = 0.5 * (1.0 - ti.tanh((z - cfg.WATERFALL_CLIFF_Z) / cfg.WATERFALL_TRANSITION_WIDTH))
        height = cfg.WATERFALL_LOWER_Y + (cfg.WATERFALL_UPPER_Y - cfg.WATERFALL_LOWER_Y) * upper
        channel_dx = (x - cfg.WATERFALL_CHANNEL_CENTER_X) / 0.65
        channel = ti.exp(-0.5 * channel_dx * channel_dx)
        return height - 0.08 * channel * upper

    @ti.kernel
    def collide(self, particles: ti.template()):
        for i in range(particles.n):
            p = particles.position[i]
            v = particles.velocity[i]
            floor = self.height_at(p[0], p[2]) + 0.025
            if p[1] < floor:
                p[1] = floor
                if v[1] < 0.0:
                    v[1] = -v[1] * cfg.TERRAIN_COLLISION_RESTITUTION
                v[0] *= cfg.TERRAIN_TANGENTIAL_DAMPING
                v[2] *= cfg.TERRAIN_TANGENTIAL_DAMPING
            particles.position[i] = p
            particles.velocity[i] = v
