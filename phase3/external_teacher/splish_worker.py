"""Minimal headless SPlisHSPlasH worker with deterministic process teardown.

The 2.17.0 Windows wheel can access invalid native state while Python destroys
the simulator module at interpreter shutdown. Simulation/export already ended
at that point. Exiting immediately after ``base.run`` lets Windows reclaim the
entire process and avoids calling the faulty native teardown path.
"""

from __future__ import annotations

import os
import sys

import pysplishsplash as sph


def main() -> None:
    base = sph.Exec.SimulatorBase()
    base.init(sys.argv, "[WaterKnowsAnswer] SPlisHSPlasH worker")
    if "--no-gui" not in sys.argv:
        gui = sph.GUI.Simulator_GUI_imgui(base)
        base.setGui(gui)
    base.run()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
