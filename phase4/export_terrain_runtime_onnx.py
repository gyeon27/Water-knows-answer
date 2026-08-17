"""Export the WCSPH-domain SPLASH ROI model used by the continuous terrain runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from phase3.config import Phase3Config
from phase3.continuous_terrain_runtime import RADIUS, wcsph_statistics
from phase3.models import UnifiedGNS


class TerrainPhysicalWrapper(nn.Module):
    def __init__(self, model: UnifiedGNS, mean: np.ndarray, std: np.ndarray):
        super().__init__(); self.model = model
        self.register_buffer("acceleration_mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("acceleration_std", torch.as_tensor(std, dtype=torch.float32))
    def forward(self, node_features, particle_type, edge_features, edge_index):
        return self.model(node_features, particle_type, edge_features, edge_index) * self.acceleration_std + self.acceleration_mean


def sha256(path: Path) -> str:
    value=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""): value.update(block)
    return value.hexdigest()


def main():
    project=Path(__file__).resolve().parents[1]; parser=argparse.ArgumentParser()
    parser.add_argument("--data-root",type=Path,required=True); parser.add_argument("--output",type=Path,default=project/"phase4"/"onnx")
    args=parser.parse_args(); cfg=Phase3Config(); stats=wcsph_statistics(project/"phase2"/"datasets"/"wcsph_pi")
    checkpoint=torch.load(args.data_root/"checkpoints"/"wcsph_zero_shot"/"best.pt",map_location="cpu",weights_only=False)
    model=UnifiedGNS(hidden=cfg.hidden_size,blocks=cfg.message_blocks,type_embedding=cfg.particle_embedding)
    model.load_state_dict(checkpoint["model"]); wrapper=TerrainPhysicalWrapper(model.eval(),stats["acc_mean"],stats["acc_std"]).eval()
    generator=torch.Generator().manual_seed(cfg.seed); n,e=64,256
    sample=(torch.randn(n,27,generator=generator),torch.zeros(n,dtype=torch.int64),torch.randn(e,4,generator=generator),torch.randint(0,n,(2,e),generator=generator,dtype=torch.int64))
    args.output.mkdir(parents=True,exist_ok=True); path=args.output/"ours_terrain_roi_splash.onnx"
    torch.onnx.export(wrapper,sample,path,input_names=["node_features","particle_type","edge_features","edge_index"],output_names=["acceleration"],opset_version=18,dynamo=False,do_constant_folding=True,dynamic_axes={"node_features":{0:"N"},"particle_type":{0:"N"},"edge_features":{0:"E"},"edge_index":{1:"E"},"acceleration":{0:"N"}})
    manifest={
      "format":"ONNX","opset":18,"model":path.name,"sha256":sha256(path),"checkpoint":"wcsph_zero_shot/best.pt",
      "purpose":"continuous arbitrary-height-map waterfall runtime; SPLASH ROI inference only",
      "inputs":{"node_features":{"dtype":"float32","shape":["N",27]},"particle_type":{"dtype":"int64","shape":["N"]},"edge_features":{"dtype":"float32","shape":["E",4]},"edge_index":{"dtype":"int64","shape":[2,"E"]}},
      "output":{"name":"acceleration","shape":["N",3],"units":"meter displacement per 30-Hz step squared"},
      "normalization":{key:value.tolist() for key,value in stats.items()},"connectivity_radius_m":RADIUS,
      "runtime_order":["inject source volume into SWE","advance SWE","extract cliff flux","emit/reuse 3-D slots","route present-state SPLASH ROI","build ROI radius graph","run this ONNX","scatter learned acceleration","terrain collision","absorb settled particles into SWE","recycle exited slots"],
      "not_in_onnx":["stateful SWE grid","source injection","dynamic routing","graph construction","2D/3D mass transfer","particle pool/recycling","terrain collision"],
      "unreal_contract":{"SWE":"C++/Compute Shader","routing_and_particle_pool":"Niagara/C++","inference":"NNE ONNX","coordinate_conversion":"Y-up meters -> Unreal Z-up centimeters"}
    }
    manifest_path=args.output/"ours_terrain_roi_splash.json"; manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    import onnx, onnxruntime as ort
    onnx.checker.check_model(onnx.load(path)); session=ort.InferenceSession(str(path),providers=["CPUExecutionProvider"])
    actual=session.run(["acceleration"],dict(zip(["node_features","particle_type","edge_features","edge_index"],[x.numpy() for x in sample])))[0]
    expected=wrapper(*sample).detach().numpy(); parity={"max_abs_error":float(np.max(np.abs(actual-expected))),"valid":bool(np.allclose(actual,expected,rtol=1e-4,atol=1e-5))}
    (args.output/"ours_terrain_roi_splash_parity.json").write_text(json.dumps(parity,indent=2),encoding="utf-8")
    print(json.dumps({"onnx":str(path),"manifest":str(manifest_path),"parity":parity},ensure_ascii=False,indent=2))


if __name__=="__main__": main()
