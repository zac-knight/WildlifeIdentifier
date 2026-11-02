"""Baseline: a linear classifier on frozen DINOv2 embeddings.

    uv run python training/linear_probe.py

Linear probing has no fine-tuning, so its accuracy measures how much the pretrained features already
know. Any gain from the fine-tuned model must beat this to justify its extra training cost.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from wildlens import imagedata as d
from wildlens.config import MODELS_DIR, ROOT, SEED
from wildlens.metrics import macro_accuracy, summary


def fit(x_train, y_train, x_val, y_val, classes, device, l2=1e-4, epochs=60):
    torch.manual_seed(SEED)
    model = torch.nn.Linear(x_train.shape[1], classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=l2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    train_x = torch.as_tensor(x_train, device=device)
    train_y = torch.as_tensor(y_train, device=device)
    val_x = torch.as_tensor(x_val, device=device)
    # Class weights stop larger species dominating, matching balanced training of the main model.
    counts = torch.bincount(train_y, minlength=classes).clamp(min=1).float()
    weights = counts.mean() / counts
    best, best_state = -1.0, None
    for _ in range(epochs):
        order = torch.randperm(len(train_x), device=device)
        for start in range(0, len(order), 1024):
            batch = order[start : start + 1024]
            loss = torch.nn.functional.cross_entropy(
                model(train_x[batch]), train_y[batch], weight=weights
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        with torch.inference_mode():
            score = macro_accuracy(model(val_x).cpu().numpy(), y_val)
        if score > best:
            best = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def main() -> None:
    frame = d.load_photos()
    data = np.load(MODELS_DIR / "embeddings" / "dinov2.npz")
    assert np.array_equal(data["photo_id"], frame["photo_id"].to_numpy()), "Embeddings are stale."
    features = data["features"]
    # Unit length is standard for DINOv2 linear evaluation, and prototypes use the same scale.
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    names = d.class_names()
    index = {name: i for i, name in enumerate(names)}
    device = d.device()
    labels = frame["species"].map(index).to_numpy()
    classifier = (frame["role"] == "classifier").to_numpy()

    output = {}
    runs = {
        "random": ("split_random", None),
        "geo": ("split_geo", None),
        "random_tail10": ("split_random", 10),
    }
    for name, (column, shots) in runs.items():
        train_rows = d.select_training(frame, column, shots)
        train = train_rows.index.to_numpy()
        val = np.where(classifier & (frame[column] == "val").to_numpy())[0]
        test = np.where(classifier & (frame[column] == "test").to_numpy())[0]
        y_train = labels[train].astype(int)
        model = fit(
            features[train], y_train, features[val], labels[val].astype(int), len(names), device
        )
        with torch.inference_mode():
            logits = model(torch.as_tensor(features[test], device=device)).cpu().numpy()
        output[name] = summary(logits, labels[test].astype(int))
        print(name, output[name], flush=True)
    path = ROOT / "docs" / "metrics" / "dinov2_probe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=1))


if __name__ == "__main__":
    main()
