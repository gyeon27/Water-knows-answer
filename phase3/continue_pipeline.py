"""Wait for GNN-only, then finish every Phase-3 model and result artifact."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

import torch


TARGET_STEP = 5_000


def log(stream, message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    stream.write(line + "\n"); stream.flush()


def checkpoint_step(path: Path) -> int:
    try:
        return int(torch.load(path, map_location="cpu", weights_only=False).get("step", 0))
    except (FileNotFoundError, EOFError, RuntimeError):
        return 0


def run(command: list[str], repository: Path, stream) -> None:
    log(stream, "RUN " + subprocess.list2cmdline(command))
    process = subprocess.Popen(command, cwd=repository, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                               errors="replace", bufsize=1)
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        stream.write(line); stream.flush()
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    root = Path(sys.argv[1] if len(sys.argv) > 1 else r"E:\WaterKnowsAnswer_Phase3")
    log_path = root / "reports" / "phase3_continuation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "checkpoints" / "gnn_only" / "latest.pt"
    with log_path.open("a", encoding="utf-8") as stream:
        log(stream, f"Waiting for {checkpoint} to reach step {TARGET_STEP}")
        last = -1
        while True:
            step = checkpoint_step(checkpoint)
            if step != last:
                log(stream, f"gnn_only checkpoint step={step}/{TARGET_STEP}")
                last = step
            if step >= TARGET_STEP:
                break
            time.sleep(30)
        python = str(repository / ".venv-gpu" / "Scripts" / "python.exe")
        base = [python, "-m", "phase3.run_phase3"]
        run(base + ["train", "--data-root", str(root), "--models", "reversed", "ours", "baseline_gns"], repository, stream)
        run(base + ["evaluate", "--data-root", str(root)], repository, stream)
        run(base + ["benchmark", "--data-root", str(root)], repository, stream)
        run(base + ["report", "--data-root", str(root)], repository, stream)
        manifest = {"completed_unix_s": time.time(), "target_step": TARGET_STEP,
                    "stages": ["models", "evaluation", "benchmark", "report"]}
        (root / "reports" / "phase3_complete.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        log(stream, "PHASE 3 PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
