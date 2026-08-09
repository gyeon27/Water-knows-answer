"""Generate the fixed 4-terrain x 3-condition PI-GNN WCSPH dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--max-particles", type=int, default=8192)
    parser.add_argument("--output", type=Path, default=root / "datasets" / "wcsph_pi")
    args = parser.parse_args()
    terrains = ["single_cliff", "sloped_cliff", "rocky_cliff", "natural_waterfall"]
    conditions = [(0.8, 0.20, 101), (1.2, 0.15, 202), (1.6, 0.10, 303)]
    args.output.mkdir(parents=True, exist_ok=True)
    index = 0
    for terrain in terrains:
        for flow, mass, seed in conditions:
            output = args.output / f"trajectory_{index:03d}_{terrain}_q{flow:.1f}.npz"
            command = [sys.executable, str(root / "generate_wcsph_teacher.py"), "--terrain", terrain, "--frames", str(args.frames), "--max-particles", str(args.max_particles), "--particle-mass", str(mass), "--flow-rate", str(flow), "--seed", str(seed), "--output", str(output)]
            print(f"[{index + 1}/12] {terrain} flow={flow} mass={mass}", flush=True)
            subprocess.run(command, check=True)
            index += 1


if __name__ == "__main__":
    main()
