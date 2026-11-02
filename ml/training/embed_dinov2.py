"""Compute frozen DINOv2 ViT-S/14 embeddings for every photo.

DINOv2 is the baseline because its self-supervised features need no labels from this task, showing
how far a strong generic representation goes before any fine-tuning. It is also used for the
few-shot prototypes. Code and weights are Apache 2.0.
"""

from __future__ import annotations

import numpy as np
import timm
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from wildlens import imagedata as d
from wildlens.config import MODELS_DIR

ARCHITECTURE = "vit_small_patch14_dinov2.lvd142m"


def build():
    return timm.create_model(ARCHITECTURE, pretrained=True, num_classes=0, img_size=d.INPUT_SIZE)


def main() -> None:
    frame = d.load_photos()
    device = d.device()
    model = build().to(device).eval()
    dataset = d.PhotoDataset(
        frame["file_path"].tolist(), np.zeros(len(frame), dtype=int), d.eval_transform()
    )
    loader = DataLoader(dataset, batch_size=64, num_workers=6)
    features = []
    with torch.inference_mode():
        for images, _ in tqdm(loader, desc="embedding"):
            features.append(model(images.to(device)).cpu().numpy().astype(np.float32))
    out = MODELS_DIR / "embeddings"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "dinov2.npz", features=np.concatenate(features), photo_id=frame["photo_id"].to_numpy()
    )
    print("saved", len(frame))


if __name__ == "__main__":
    main()
