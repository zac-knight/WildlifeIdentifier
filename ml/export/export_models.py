"""Export trained models for the browser, with accuracy and parity checks.

    uv run python export/export_models.py image prior
    uv run python export/export_models.py audio
    uv run python export/export_models.py fewshot

Each model produces a metadata JSON with calibration values and file checksums. A file ships
only if it passes the checks: PyTorch parity for fp32 and fp16, and a measured accuracy loss of
no more
than one percentage point for every variant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from wildlens import imagedata as d
from wildlens import metrics as m
from wildlens.audio import CLIP_SAMPLES, LogMel
from wildlens.config import DATA_DIR, MANIFEST_DIR, MODELS_DIR, ROOT, SEED
from wildlens.onnx_export import (
    HalfImageForExport,
    ImageForExport,
    build_for_export,
    export_onnx,
    max_absolute_difference,
    session,
    to_int8,
)
from wildlens.prior import PriorForExport, SpeciesPrior

OUT = MODELS_DIR / "web"
MAX_ACCURACY_LOSS = 0.01
PARITY_TOLERANCE = {"fp32": 1e-3, "fp16": 0.1}
REPORT = ROOT / "docs" / "metrics" / "export.json"
VERSION = "2026-11-v1"


def file_entry(path: Path, root: Path) -> dict:
    return {
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def update_report(name: str, values: dict) -> None:
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    report[name] = values
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1))


def validation_sample(frame: pd.DataFrame, index: dict, count: int) -> pd.DataFrame:
    return (
        frame[(frame["role"] == "classifier") & (frame["split_random"] == "val")]
        .sample(count, random_state=SEED)
        .assign(label=lambda rows: rows["species"].map(index))
    )


def onnx_accuracy(path: Path, sample: pd.DataFrame) -> tuple[float, float]:
    """Accuracy of the exported file itself. A real file may behave unlike its source model."""
    runner = session(path)
    dataset = d.PhotoDataset(
        sample["file_path"].tolist(), sample["label"].to_numpy(), d.eval_transform()
    )
    logits = []
    for images, _ in tqdm(DataLoader(dataset, batch_size=1, num_workers=4), leave=False):
        logits.append(runner.run(None, {"image": images.numpy()})[0])
    return (
        m.topk_accuracy(np.concatenate(logits), sample["label"].to_numpy(), 1),
        m.macro_accuracy(np.concatenate(logits), sample["label"].to_numpy()),
    )


def export_image() -> None:
    run = MODELS_DIR / "runs" / "image_random"
    config = json.loads((run / "config.json").read_text())
    names = config["classes"]
    index = {name: i for i, name in enumerate(names)}
    model = build_for_export(config["architecture"], len(names))
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True))
    model.eval()

    folder = OUT / "image"
    folder.mkdir(parents=True, exist_ok=True)
    example = torch.randn(1, 3, 224, 224, generator=torch.Generator().manual_seed(SEED))
    outputs = ["logits", "features"]
    paths = {
        "fp32": folder / "model.fp32.onnx",
        "fp16": folder / "model.fp16.onnx",
        "int8": folder / "model.int8.onnx",
    }
    wrapper = ImageForExport(model).eval()
    export_onnx(wrapper, example, paths["fp32"], ["image"], outputs)
    export_onnx(HalfImageForExport(model).eval(), example, paths["fp16"], ["image"], outputs)

    frame = d.load_photos()
    sample = validation_sample(frame, index, 5000)
    calibration = sample.sample(200, random_state=SEED)
    transform = d.eval_transform()
    batches = [
        transform(d.Image.open(DATA_DIR / path).convert("RGB"))[None].numpy()
        for path in calibration["file_path"]
    ]
    to_int8(paths["fp32"], paths["int8"], "image", batches)

    probe = torch.as_tensor(np.concatenate(batches[:8]))
    with torch.inference_mode():
        expected = wrapper(probe)[0].numpy()
    report: dict = {"architecture": config["architecture"], "variants": {}}
    baseline_top1, baseline_macro = onnx_accuracy(paths["fp32"], sample)
    for variant, path in paths.items():
        top1, macro = (
            (baseline_top1, baseline_macro) if variant == "fp32" else onnx_accuracy(path, sample)
        )
        difference = max_absolute_difference(path, {"image": probe.numpy()}, expected)
        report["variants"][variant] = {
            "megabytes": path.stat().st_size / 1e6,
            "val_sample_top1": top1,
            "val_sample_macro": macro,
            "max_logit_difference_vs_pytorch": difference,
            "accuracy_loss_top1": baseline_top1 - top1,
            "within_one_point": baseline_top1 - top1 <= MAX_ACCURACY_LOSS,
        }
        print(variant, report["variants"][variant], flush=True)
    report["sample_size"] = len(sample)
    for variant, limit in PARITY_TOLERANCE.items():
        assert report["variants"][variant]["max_logit_difference_vs_pytorch"] < limit, variant
    update_report("image", report)

    weights = model.get_classifier().weight.detach().numpy().astype(np.float32)
    weights_path = folder / "classifier_weights.bin"
    weights.tofile(weights_path)
    with torch.inference_mode():
        channels = wrapper(example)[1].shape

    metrics = json.loads((ROOT / "docs" / "metrics" / "image.json").read_text())
    species = pd.read_csv(MANIFEST_DIR / "species.csv").set_index("species")
    trust = metrics["trust"]
    fusion = metrics["random"]["fusion"]
    wasm = paths["int8"] if report["variants"]["int8"]["within_one_point"] else paths["fp16"]
    metadata = {
        "version": VERSION,
        "classes": names,
        "genera": [species.loc[n, "genus"] for n in names],
        "families": [
            species.loc[n, "family"] if isinstance(species.loc[n, "family"], str) else "Unknown"
            for n in names
        ],
        "temperature": metrics["random"]["temperature"],
        "fusedTemperature": fusion["fused_temperature"],
        "noLocationThresholds": trust["thresholds_no_location"],
        "thresholds": trust["thresholds_with_location"],
        "priorWeight": fusion["weight"],
        "priorFloor": fusion["floor"],
        "energyThreshold": trust["ood"]["energy_threshold"],
        "featureChannels": int(channels[1]),
        "featureSize": int(channels[2]),
        "weightsFile": file_entry(weights_path, folder),
        "files": {
            "webgpu": file_entry(paths["fp16"], folder),
            "wasm": file_entry(wasm, folder),
        },
    }
    (folder / "metadata.json").write_text(json.dumps(metadata))


def export_prior() -> None:
    run = MODELS_DIR / "runs" / "prior_random"
    config = json.loads((run / "config.json").read_text())
    names = config["classes"]
    model = SpeciesPrior(len(names))
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True))
    wrapper = PriorForExport(model).eval()
    folder = OUT / "prior"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "model.onnx"
    example = torch.tensor([[-33.9, 151.2, 100.0]])
    export_onnx(wrapper, example, path, ["place"], ["presence"])
    probe = torch.tensor([[-12.4, 130.8, 10.0], [-42.9, 147.3, 200.0], [-31.9, 115.9, 365.0]])
    with torch.inference_mode():
        expected = wrapper(probe).numpy()
    difference = max_absolute_difference(path, {"place": probe.numpy()}, expected)
    assert difference < 1e-4, difference
    update_report(
        "prior",
        {"megabytes": path.stat().st_size / 1e6, "max_difference_vs_pytorch": difference},
    )
    entry = file_entry(path, folder)
    (folder / "metadata.json").write_text(
        json.dumps(
            {"version": VERSION, "classes": names, "files": {"webgpu": entry, "wasm": entry}}
        )
    )


class AudioForExport(nn.Module):
    """The spectrogram stays float32, as half precision cannot hold its tiny log-floor values."""

    def __init__(self, model: nn.Module, half: bool) -> None:
        super().__init__()
        self.extractor = LogMel()
        self.half = half
        self.model = model

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        image = self.extractor(waveform)
        if self.half:
            return self.model(image.half()).float()
        return self.model(image)


def export_audio() -> None:
    run = MODELS_DIR / "runs" / "audio_random"
    config = json.loads((run / "config.json").read_text())
    names = config["classes"]
    model = build_for_export(config["architecture"], len(names))
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True))
    model.eval()
    folder = OUT / "audio"
    folder.mkdir(parents=True, exist_ok=True)
    example = torch.randn(1, CLIP_SAMPLES, generator=torch.Generator().manual_seed(SEED)) * 0.1
    fp32 = folder / "model.fp32.onnx"
    fp16 = folder / "model.fp16.onnx"
    full = AudioForExport(model, half=False).eval()
    export_onnx(full, example, fp32, ["waveform"], ["logits"])
    half_model = AudioForExport(build_for_export(config["architecture"], len(names)), True).eval()
    half_model.model.load_state_dict(model.state_dict())
    half_model.model.half()
    export_onnx(half_model, example, fp16, ["waveform"], ["logits"])
    with torch.inference_mode():
        expected = full(example).numpy()
    report = {
        "fp32_megabytes": fp32.stat().st_size / 1e6,
        "fp16_megabytes": fp16.stat().st_size / 1e6,
        "fp32_difference": max_absolute_difference(fp32, {"waveform": example.numpy()}, expected),
        "fp16_difference": max_absolute_difference(fp16, {"waveform": example.numpy()}, expected),
    }
    assert report["fp32_difference"] < 1e-2 and report["fp16_difference"] < 0.2, report
    update_report("audio", report)
    predictions = np.load(run / "predictions.npz")
    temperature = m.fit_temperature(
        predictions["logits_val"], predictions["labels_val"].astype(int)
    )
    metadata = {
        "version": VERSION,
        "classes": names,
        "temperature": temperature,
        "clips": 3,
        "files": {"webgpu": file_entry(fp16, folder), "wasm": file_entry(fp16, folder)},
    }
    (folder / "metadata.json").write_text(json.dumps(metadata))


def export_fewshot() -> None:
    from training.embed_dinov2 import ARCHITECTURE  # noqa: PLC0415

    model = timm.create_model(ARCHITECTURE, pretrained=True, num_classes=0, img_size=224).eval()

    class Embed(nn.Module):
        def __init__(self, backbone: nn.Module, half: bool) -> None:
            super().__init__()
            self.backbone = backbone
            self.half = half

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            source = image.half() if self.half else image
            output = self.backbone(source).float()
            return output / output.norm(dim=-1, keepdim=True)

    folder = OUT / "fewshot"
    folder.mkdir(parents=True, exist_ok=True)
    example = torch.randn(1, 3, 224, 224, generator=torch.Generator().manual_seed(SEED))
    fp32, int8 = folder / "model.fp32.onnx", folder / "model.int8.onnx"
    export_onnx(Embed(model, False).eval(), example, fp32, ["image"], ["embedding"])
    to_int8(fp32, int8, "image", [torch.randn(1, 3, 224, 224).numpy() for _ in range(8)])
    with torch.inference_mode():
        expected = Embed(model, False)(example).numpy()
    report = {
        "fp32_megabytes": fp32.stat().st_size / 1e6,
        "int8_megabytes": int8.stat().st_size / 1e6,
        "fp32_difference": max_absolute_difference(fp32, {"image": example.numpy()}, expected),
    }
    assert report["fp32_difference"] < 1e-3, report
    update_report("fewshot", report)

    settings = json.loads((ROOT / "docs" / "metrics" / "fewshot.json").read_text())["settings"]
    prototypes = np.load(MODELS_DIR / "fewshot" / "prototypes.npy")
    quantised = np.clip(np.round(prototypes * 127), -127, 127).astype(np.int8)
    prototypes_path = folder / "prototypes.bin"
    quantised.tofile(prototypes_path)
    names = d.class_names()
    entry = file_entry(fp32, folder)
    metadata = {
        "version": VERSION,
        "uncertainBelow": settings["uncertain_below"],
        "weight": settings["weight"],
        "scale": settings["scale"],
        "dimension": prototypes.shape[1],
        "classes": names,
        "prototypesFile": file_entry(prototypes_path, folder),
        "files": {"webgpu": entry, "wasm": file_entry(int8, folder)},
    }
    (folder / "metadata.json").write_text(json.dumps(metadata))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", choices=["image", "prior", "audio", "fewshot"])
    args = parser.parse_args()
    actions = {
        "image": export_image,
        "prior": export_prior,
        "audio": export_audio,
        "fewshot": export_fewshot,
    }
    for name in args.models:
        print(f"== exporting {name}", flush=True)
        actions[name]()


if __name__ == "__main__":
    main()
