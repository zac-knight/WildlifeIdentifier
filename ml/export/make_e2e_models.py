"""Build tiny untrained ONNX models, so browser tests can run the real pipeline in CI.

    uv run python export/make_e2e_models.py

They use the real model interfaces and metadata format, but their predictions mean nothing. Tests
that use them check the plumbing only. Trained models are too large for CI.
"""

from __future__ import annotations

import hashlib
import json

import torch
from torch import nn

from wildlens.config import ROOT
from wildlens.onnx_export import export_onnx

OUT = ROOT / "web" / "public" / "e2e-models"
SPECIES = json.loads((ROOT / "web" / "data" / "species.json").read_text())
NAMES = [row["name"] for row in SPECIES]
CHANNELS = 4
SIZE = 7


class TinyImage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, CHANNELS, 3, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(CHANNELS, CHANNELS, 3, stride=8, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(CHANNELS, len(NAMES))

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(image)
        pooled = features.mean(dim=(2, 3))
        return self.classifier(pooled), features


class TinyPrior(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(3, len(NAMES))

    def forward(self, place: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.layer(place / 100))


def entry(path) -> dict:
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    torch.manual_seed(0)
    model = TinyImage().eval()
    folder = OUT / "image"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "model.onnx"
    export_onnx(model, torch.zeros(1, 3, 224, 224), path, ["image"], ["logits", "features"])
    assert model(torch.zeros(1, 3, 224, 224))[1].shape[-1] == SIZE
    weights = folder / "classifier_weights.bin"
    model.classifier.weight.detach().numpy().tofile(weights)
    file = entry(path)
    (folder / "metadata.json").write_text(
        json.dumps(
            {
                "version": "e2e",
                "classes": NAMES,
                "genera": [row["genus"] or "Unknown" for row in SPECIES],
                "families": [row["family"] or "Unknown" for row in SPECIES],
                "temperature": 1.0,
                "fusedTemperature": 1.0,
                "noLocationThresholds": {"species": 0.9, "genus": 0.9, "family": 0.9},
                "thresholds": {"species": 0.9, "genus": 0.9, "family": 0.9},
                "priorWeight": 1.0,
                "priorFloor": 0.001,
                "energyThreshold": -1000.0,
                "featureChannels": CHANNELS,
                "featureSize": SIZE,
                "weightsFile": entry(weights),
                "files": {"webgpu": file, "wasm": file},
            }
        )
    )

    folder = OUT / "prior"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "model.onnx"
    export_onnx(TinyPrior().eval(), torch.zeros(1, 3), path, ["place"], ["presence"])
    file = entry(path)
    (folder / "metadata.json").write_text(
        json.dumps({"version": "e2e", "classes": NAMES, "files": {"webgpu": file, "wasm": file}})
    )
    size = sum(item.stat().st_size for item in OUT.rglob("*") if item.is_file())
    print(f"wrote {size / 1e3:.0f} KB to {OUT}")


if __name__ == "__main__":
    main()
