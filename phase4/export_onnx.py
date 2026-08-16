"""Export full-graph and routed-ROI PI-GNN packages for Unreal Engine NNE."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch import nn

from phase3.config import Phase3Config
from phase3.models import UnifiedGNS


class PhysicalAccelerationWrapper(nn.Module):
    """Denormalize network output so Unreal receives acceleration per data step."""

    def __init__(self, model: UnifiedGNS, acceleration_mean: list[float], acceleration_std: list[float]):
        super().__init__()
        self.model = model
        self.register_buffer("acceleration_mean", torch.tensor(acceleration_mean, dtype=torch.float32))
        self.register_buffer("acceleration_std", torch.tensor(acceleration_std, dtype=torch.float32))

    def forward(self, node_features, particle_type, edge_features, edge_index):
        normalized = self.model(node_features, particle_type, edge_features, edge_index)
        return normalized * self.acceleration_std + self.acceleration_mean


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(mode: str, onnx_name: str, sha256: str, cfg: Phase3Config, metadata: dict) -> dict:
    common = {
        "format": "ONNX", "opset": 18, "model": onnx_name, "sha256": sha256,
        "network": {"hidden": cfg.hidden_size, "message_passing_blocks": cfg.message_blocks,
                    "particle_embedding": cfg.particle_embedding},
        "inputs": {
            "node_features": {"dtype": "float32", "shape": ["N", 27]},
            "particle_type": {"dtype": "int64", "shape": ["N"]},
            "edge_features": {"dtype": "float32", "shape": ["E", 4]},
            "edge_index": {"dtype": "int64", "shape": [2, "E"], "order": ["sender", "receiver"]},
        },
        "output": {"name": "acceleration", "dtype": "float32", "shape": ["N", 3],
                   "units": "Water-3D coordinate displacement per simulation-step squared"},
        "connectivity_radius": metadata["default_connectivity_radius"],
        "routing": {
            "STREAM": "2D shallow-water equation solver",
            "POOL": "2D shallow-water equation solver",
            "SPLASH": "PI-GNN ONNX inference",
            "transition": "blend SWE and PI-GNN acceleration across the routing boundary",
            "future_frame_features_forbidden": True,
        },
        "coordinates": {"training": "Y-up dataset coordinates", "unreal": "Z-up centimeters",
                        "conversion": "(x,y,z)_data -> (100*x,100*z,100*y)_Unreal"},
    }
    if mode == "full_graph":
        common["runtime"] = {
            "graph_nodes": "all active fluid and boundary particles",
            "use_output_for": "SPLASH nodes only",
            "discard_output_for": "STREAM and POOL; advance these with SWE",
        }
    else:
        common["runtime"] = {
            "order": ["route all particles", "advance STREAM/POOL with SWE", "gather SPLASH particle IDs",
                      "build radius graph only for SPLASH ROI", "run ONNX", "scatter acceleration by saved IDs", "blend boundary"],
            "graph_nodes": "SPLASH ROI only",
            "empty_roi": "skip ONNX inference",
            "index_mapping": "preserve roi_to_global[N] outside ONNX and scatter output back to global particles",
        }
    return common


def export_packages(data_root: Path, output: Path) -> list[Path]:
    cfg = Phase3Config()
    checkpoint = data_root / "checkpoints" / "ours" / "best.pt"
    metadata = json.loads((data_root / "raw" / cfg.dataset / "metadata.json").read_text(encoding="utf-8"))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = UnifiedGNS(hidden=cfg.hidden_size, blocks=cfg.message_blocks, type_embedding=cfg.particle_embedding)
    model.load_state_dict(state["model"]); model.eval()
    wrapper = PhysicalAccelerationWrapper(model, metadata["acc_mean"], metadata["acc_std"]).eval()
    output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(cfg.seed)
    n, e = 64, 256
    sample = (
        torch.randn(n, 27, generator=generator),
        torch.zeros(n, dtype=torch.int64),
        torch.randn(e, 4, generator=generator),
        torch.randint(0, n, (2, e), generator=generator, dtype=torch.int64),
    )
    primary = output / "ours_full_graph.onnx"
    torch.onnx.export(
        wrapper, sample, primary, input_names=["node_features", "particle_type", "edge_features", "edge_index"],
        output_names=["acceleration"], opset_version=18, dynamo=False, do_constant_folding=True,
        dynamic_axes={"node_features": {0: "N"}, "particle_type": {0: "N"},
                      "edge_features": {0: "E"}, "edge_index": {1: "E"}, "acceleration": {0: "N"}},
    )
    optimized = output / "ours_roi_splash.onnx"
    shutil.copy2(primary, optimized)
    np.savez_compressed(output / "onnx_parity_input.npz", node_features=sample[0].numpy(),
                        particle_type=sample[1].numpy(), edge_features=sample[2].numpy(), edge_index=sample[3].numpy(),
                        pytorch_acceleration=wrapper(*sample).detach().numpy())
    for mode, path in (("full_graph", primary), ("roi_splash", optimized)):
        manifest = _manifest(mode, path.name, _sha256(path), cfg, metadata)
        (output / f"{path.stem}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return [primary, optimized]


def verify(output: Path) -> dict:
    import onnx
    import onnxruntime as ort
    sample = np.load(output / "onnx_parity_input.npz", allow_pickle=False)
    result = {}
    for name in ("ours_full_graph", "ours_roi_splash"):
        path = output / f"{name}.onnx"
        onnx.checker.check_model(onnx.load(path))
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
        session = ort.InferenceSession(str(path), providers=providers)
        actual = session.run(["acceleration"], {key: sample[key] for key in ("node_features", "particle_type", "edge_features", "edge_index")})[0]
        expected = sample["pytorch_acceleration"]
        result[name] = {"provider": session.get_providers()[0], "max_abs_error": float(np.max(np.abs(actual - expected))),
                        "mean_abs_error": float(np.mean(np.abs(actual - expected))), "valid": bool(np.allclose(actual, expected, rtol=1e-4, atol=1e-5))}
    (output / "onnx_parity_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("phase4/onnx"))
    args = parser.parse_args()
    paths = export_packages(args.data_root, args.output)
    print(json.dumps({"models": list(map(str, paths)), "parity": verify(args.output)}, indent=2))


if __name__ == "__main__":
    main()
