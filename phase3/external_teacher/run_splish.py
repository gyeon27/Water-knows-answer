"""Run SPlisHSPlasH through an ASCII-only staging directory on Windows."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=18.0)
    parser.add_argument("--smoke", action="store_true", help="run only 0.05 simulation seconds")
    parser.add_argument("--gui", action="store_true", help="open the native interactive 3-D GUI")
    parser.add_argument("--keep-stage", action="store_true")
    parser.add_argument("--generated", type=Path,
                        help="terrain/scene directory (defaults to external_teacher/generated)")
    parser.add_argument("--stage-name", default="wka_splish_teacher")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    executable = root / ".venv-splish" / "Scripts" / "splash.exe"
    generated = args.generated.resolve() if args.generated else Path(__file__).resolve().parent / "generated"
    if not executable.exists():
        raise FileNotFoundError(f"missing {executable}; install pysplishsplash in .venv-splish")
    stage = Path(tempfile.gettempdir()) / args.stage_name
    if stage.exists() and not args.keep_stage:
        shutil.rmtree(stage)
    (stage / "scene").mkdir(parents=True, exist_ok=True)
    (stage / "output" / "log").mkdir(parents=True, exist_ok=True)
    for pattern in ("output/vtk/*.vtk", "output/partio/*.bgeo"):
        for old_frame in stage.glob(pattern):
            old_frame.unlink(missing_ok=True)
    shutil.copy2(generated / "terrain_collision.obj", stage / "scene" / "terrain_collision.obj")
    scene = json.loads((generated / "waterfall_dfSPH.json").read_text(encoding="utf-8"))
    scene["RigidBodies"][0]["geometryFile"] = (stage / "scene" / "terrain_collision.obj").as_posix()
    scene_path = stage / "scene" / "waterfall_dfSPH.json"
    scene_path.write_text(json.dumps(scene, indent=2), encoding="utf-8")
    stop = 0.05 if args.smoke else args.seconds
    python = root / ".venv-splish" / "Scripts" / "python.exe"
    command = [str(python), "-m", "phase3.external_teacher.splish_worker", str(scene_path)]
    if not args.gui:
        command.append("--no-gui")
    command.extend(["--no-initial-pause", "--stopAt", str(stop),
                    "--output-dir", str(stage / "output")])
    print(" ".join(command), flush=True)
    completed = subprocess.run(command, check=False)
    manifest = {"returncode": completed.returncode, "stage": str(stage), "seconds": stop, "command": command}
    (generated / "last_run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if completed.returncode:
        raise SystemExit(completed.returncode)
    if not args.keep_stage and args.smoke:
        shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    main()
