"""Save a trained classifier's logits for every evaluation set, so later analysis needs no GPU.

uv run python evaluation/predict_image.py --run image_random --split split_random
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from wildlens import imagedata as d
from wildlens.config import MANIFEST_DIR, MODELS_DIR


def predict(model, frame: pd.DataFrame, device, labels=None) -> np.ndarray:
    if frame.empty:
        return np.zeros((0, model.num_classes), dtype=np.float32)
    targets = labels if labels is not None else np.zeros(len(frame), dtype=int)
    dataset = d.PhotoDataset(frame["file_path"].tolist(), targets, d.eval_transform())
    loader = DataLoader(dataset, batch_size=96, num_workers=6)
    outputs = []
    with torch.inference_mode():
        for images, _ in tqdm(loader, leave=False):
            outputs.append(model(images.to(device)).cpu().numpy())
    return np.concatenate(outputs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", default="split_random")
    args = parser.parse_args()

    run = MODELS_DIR / "runs" / args.run
    config = json.loads((run / "config.json").read_text())
    names = config["classes"]
    index = {name: i for i, name in enumerate(names)}
    device = d.device()
    model = timm.create_model(config["architecture"], pretrained=False, num_classes=len(names))
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True))
    model = model.to(device).eval()

    frame = d.load_photos()
    results: dict[str, np.ndarray] = {}
    for split in ("val", "test"):
        chosen = frame[(frame["role"] == "classifier") & (frame[args.split] == split)]
        results[f"logits_{split}"] = predict(model, chosen, device).astype(np.float32)
        results[f"labels_{split}"] = chosen["species"].map(index).to_numpy()
        results[f"photo_{split}"] = chosen["photo_id"].to_numpy()
        print(split, len(chosen), flush=True)
    for split in ("val", "test"):
        chosen = frame[(frame["role"] == "ood") & (frame["ood_split"] == split)]
        results[f"logits_ood_{split}"] = predict(model, chosen, device).astype(np.float32)
    far = pd.read_parquet(MANIFEST_DIR / "photos_far_ood.parquet")
    results["logits_far"] = predict(model, far, device).astype(np.float32)
    np.savez_compressed(run / "predictions.npz", **results)
    print("saved predictions", flush=True)


if __name__ == "__main__":
    main()
