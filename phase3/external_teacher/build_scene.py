"""Build an official-format SPlisHSPlasH DFSPH waterfall scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain", type=Path, default=Path("phase3/external_teacher/generated/terrain_collision.obj"))
    parser.add_argument("--output", type=Path, default=Path("phase3/external_teacher/generated/waterfall_dfSPH.json"))
    parser.add_argument("--duration", type=float, default=18.0)
    parser.add_argument("--flow-speed", type=float, default=0.8)
    parser.add_argument("--particle-radius", type=float, default=0.055)
    parser.add_argument("--heightfield", type=Path,
                        help="terrain_height.npz used to place the emitter on the upstream surface")
    parser.add_argument("--emitter-x", type=float)
    parser.add_argument("--emitter-y", type=float)
    parser.add_argument("--emitter-z", type=float, default=0.0)
    parser.add_argument("--emitter-angle", type=float, default=3.141592653589793)
    parser.add_argument("--emitter-width", type=int, default=12)
    args = parser.parse_args()
    terrain = args.terrain.resolve()
    emitter_x, emitter_y = 9.8, 9.80
    if args.heightfield:
        with np.load(args.heightfield) as hf:
            height = np.asarray(hf["height"], np.float32)
            length = float(hf["length_m"])
        # +X is upstream. Start directly above the surface, with only one
        # particle diameter of clearance to avoid initial SDF penetration.
        emitter_x = 0.37 * length
        row = int(np.clip(round((emitter_x / length + 0.5) * (height.shape[0] - 1)), 0, height.shape[0] - 1))
        band = height[row, height.shape[1] // 2 - 4:height.shape[1] // 2 + 5]
        emitter_y = float(np.max(band) + 2.05 * args.particle_radius)
    if args.emitter_x is not None:
        emitter_x = args.emitter_x
    if args.emitter_y is not None:
        emitter_y = args.emitter_y
    scene = {
        "Configuration": {
            "cameraPosition": [-20, 12, 20], "cameraLookat": [0, 3.8, 0],
            "timeStepSize": 0.0025, "numberOfStepsPerRenderUpdate": 4,
            "particleRadius": args.particle_radius, "simulationMethod": 4,
            "gravitation": [0, -9.81, 0], "cflMethod": 2, "cflFactor": 0.5,
            "cflMaxTimeStepSize": 0.004, "boundaryHandlingMethod": 0,
            "particleAttributes": "density;velocity",
            "enablePartioExport": False, "enableVTKExport": True, "dataExportFPS": 30.0,
            "renderWalls": 3, "renderMinValue": 0.0, "renderMaxValue": 6.0,
            "DFSPH": {"minIterations": 2, "maxIterations": 100, "maxError": 0.05,
                      "maxIterationsV": 100, "maxErrorV": 0.1, "enableDivergenceSolver": True},
        },
        "Materials": [{
            "id": "Fluid", "density0": 1000, "maxEmitterParticles": 180000,
            "emitterReuseParticles": True, "emitterBoxMin": [-15, -3, -10],
            "emitterBoxMax": [15, 14, 10], "viscosityMethod": 1,
            "Standard viscosity": {"viscosity": 0.012},
        }],
        "RigidBodies": [{
            "geometryFile": terrain.as_posix(), "translation": [0, 0, 0],
            "rotationAxis": [1, 0, 0], "rotationAngle": 0, "scale": [1, 1, 1],
            "color": [0.28, 0.34, 0.36, 1], "isDynamic": False, "isWall": False,
            "mapInvert": False, "mapThickness": 0.0, "mapResolution": [120, 80, 80],
            "samplingMode": 1,
        }],
        "Emitters": [{
            "width": args.emitter_width, "height": 1,
            "translation": [emitter_x, emitter_y, args.emitter_z],
            "rotationAxis": [0, 1, 0], "rotationAngle": args.emitter_angle,
            "velocity": args.flow_speed, "type": 0, "emitStartTime": 0.0,
            "emitEndTime": args.duration,
        }],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scene, indent=2), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
