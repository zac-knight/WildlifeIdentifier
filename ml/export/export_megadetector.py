"""Export MegaDetector V6 (MIT, YOLOv9-based) so that it runs in the browser.

    uv run --with omegaconf --with rich --with loguru --with einops --with hydra-core \
        python export/export_megadetector.py

Requires, in models/megadetector/, the official weights from Zenodo record 15398270 and config from
record 15178680, and Microsoft's model code unpacked from the PytorchWildlife 1.3.0 wheel under
data/tools/megadetector/. Non-maximum suppression runs in TypeScript, because ONNX's version of it
is poorly supported on WebGPU. The graph returns boxes and class scores for every anchor.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch import nn

from wildlens.config import DATA_DIR, MODELS_DIR, ROOT
from wildlens.onnx_export import export_onnx, max_absolute_difference, session, to_int8

CODE = DATA_DIR / "tools" / "megadetector" / "PytorchWildlife" / "models" / "detection" / "yolo_mit"
WEIGHTS = MODELS_DIR / "megadetector" / "MDV6-mit-yolov9-c.ckpt"
CONFIG = MODELS_DIR / "megadetector" / "config_v9s.yaml"
SIZE = 640
OUT = MODELS_DIR / "web" / "detector"


def load_detector():
    sys.path.insert(0, str(CODE))
    from yolo import create_converter, create_model  # noqa: PLC0415

    config = OmegaConf.create(yaml.safe_load(CONFIG.read_text()))
    config.image_size = [SIZE, SIZE]
    model = create_model(config.model, weight_path=False, class_num=3).eval()
    # weights_only prevents a checkpoint from running code when it loads. A checkpoint that needs
    # more than this raises an error, which is the safe outcome.
    checkpoint = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    state = {key.replace("model.model.", ""): value for key, value in state.items()}
    expected = model.model.state_dict()
    matching = {
        key: value
        for key, value in state.items()
        if key in expected and value.shape == expected[key].shape
    }
    if len(matching) != len(expected):
        raise RuntimeError(
            f"Loaded {len(matching)} of {len(expected)} weights from the checkpoint."
        )
    model.model.load_state_dict(matching)
    converter = create_converter(config.model.name, model, config.model.anchor, [SIZE, SIZE], "cpu")
    return model, converter


class DetectorForExport(nn.Module):
    """Outputs boxes in input pixels, plus per-class scores, for every anchor."""

    def __init__(self, model: nn.Module, converter) -> None:
        super().__init__()
        self.model = model
        self.converter = converter

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.model(image)
        classes, _, boxes = self.converter(output["Main"])[:3]
        return boxes, classes.sigmoid()


def entry(path: Path) -> dict:
    return {
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def main() -> None:
    model, converter = load_detector()
    wrapper = DetectorForExport(model, converter).eval()
    parameters = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    print(f"parameters: {parameters:.1f} M")
    OUT.mkdir(parents=True, exist_ok=True)
    fp32, int8 = OUT / "model.fp32.onnx", OUT / "model.int8.onnx"
    example = torch.rand(1, 3, SIZE, SIZE, generator=torch.Generator().manual_seed(0))
    export_onnx(wrapper, example, fp32, ["image"], ["boxes", "scores"], {})
    with torch.inference_mode():
        boxes, _ = wrapper(example)
    difference = max_absolute_difference(fp32, {"image": example.numpy()}, boxes.numpy())
    assert difference < 0.05, difference
    outputs = session(fp32).run(None, {"image": example.numpy()})
    print("anchors:", outputs[0].shape, outputs[1].shape, "box difference:", difference)
    calibration = [
        torch.rand(1, 3, SIZE, SIZE, generator=torch.Generator().manual_seed(i)).numpy()
        for i in range(8)
    ]
    int8_ok = False
    try:
        to_int8(fp32, int8, "image", calibration)
        int8_ok = True
    except Exception as error:  # noqa: BLE001
        print("int8 conversion failed, so fp32 is used for WebAssembly:", error)
    metadata = {
        "version": "mdv6-mit-yolov9-c",
        "inputSize": SIZE,
        "classes": ["animal", "person", "vehicle"],
        "confidence": 0.2,
        "iou": 0.5,
        "files": {"webgpu": entry(fp32), "wasm": entry(int8 if int8_ok else fp32)},
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata))
    report = {
        "parameters_millions": parameters,
        "fp32_megabytes": fp32.stat().st_size / 1e6,
        "int8_megabytes": int8.stat().st_size / 1e6 if int8_ok else None,
        "max_box_difference_pixels": difference,
        "licence": "Model zoo lists MIT. The Zenodo record is marked CC BY 4.0.",
    }
    path = ROOT / "docs" / "metrics" / "export.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing["detector"] = report
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=1))
    print(json.dumps(report, indent=1), np.__name__)


if __name__ == "__main__":
    main()
