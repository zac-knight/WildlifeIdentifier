"""Fine-tune an image backbone on log mel spectrograms of bird recordings.

    uv run python training/train_audio.py --name audio_random

Spectrograms have local time-frequency patterns much like image textures, so ImageNet features are
a strong starting point even though the input is not a photograph. EfficientNetV2-B0 keeps the
download small, because audio data is scarcer and a larger model would overfit it more easily.
Labels apply to a whole recording, so loudest clips are used and predictions are averaged across
clips at evaluation.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from wildlens import imagedata as d
from wildlens.audio import CLIP_SAMPLES, LogMel, spec_augment
from wildlens.config import DATA_DIR, MANIFEST_DIR, MODELS_DIR, SEED
from wildlens.metrics import macro_accuracy

ARCHITECTURE = "tf_efficientnetv2_b0.in1k"


class ClipDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, labels: np.ndarray, train: bool) -> None:
        self.paths = frame["clip_path"].tolist()
        self.labels = labels
        self.train = train
        self.rng = np.random.default_rng(SEED)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        clips = np.load(DATA_DIR / self.paths[index], mmap_mode="r")
        choice = int(np.random.randint(len(clips))) if self.train else 0
        wave = np.asarray(clips[choice], dtype=np.float32) / 32768.0
        if self.train:
            # Time shifting and gain changes mimic differences between recordists and microphones.
            wave = np.roll(wave, int(np.random.randint(CLIP_SAMPLES)))
            wave = wave * float(10 ** np.random.uniform(-0.5, 0.5))
            # Background noise mixed in from another recording of any species simulates field noise.
            other = np.load(
                DATA_DIR / self.paths[np.random.randint(len(self.paths))], mmap_mode="r"
            )
            noise = np.asarray(other[0], dtype=np.float32) / 32768.0
            wave = wave + noise * float(np.random.uniform(0, 0.25))
        return wave, int(self.labels[index])


def predict_recordings(extractor, model, frame, device) -> np.ndarray:
    """Averages logits over each recording's stored clips, as the browser does."""
    model.eval()
    outputs = []
    with torch.inference_mode():
        for path in frame["clip_path"]:
            clips = np.load(DATA_DIR / path).astype(np.float32) / 32768.0
            logits = model(extractor(torch.as_tensor(clips, device=device)))
            outputs.append(logits.mean(dim=0).cpu().numpy())
    return np.stack(outputs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="audio_random")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = d.device()
    out = d.run_dir(MODELS_DIR, args.name)
    frame = pd.read_parquet(MANIFEST_DIR / "sounds_ready.parquet")
    names = d.class_names()
    index = {name: i for i, name in enumerate(names)}
    train = frame[frame["split_random"] == "train"]
    val = frame[frame["split_random"] == "val"]
    test = frame[frame["split_random"] == "test"]
    y_train = train["species"].map(index).to_numpy()
    y_val = val["species"].map(index).to_numpy()
    y_test = test["species"].map(index).to_numpy()

    weights = torch.as_tensor(d.balanced_weights(y_train, len(names)), dtype=torch.double)
    loader = DataLoader(
        ClipDataset(train, y_train, train=True),
        batch_size=args.batch_size,
        sampler=WeightedRandomSampler(weights, len(train) * 2, replacement=True),
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        drop_last=True,
    )
    extractor = LogMel().to(device)
    model = timm.create_model(ARCHITECTURE, pretrained=True, num_classes=len(names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr / 3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    start, best, stale = 0, -1.0, 0
    last = out / "last.pt"
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start, best, stale = state["epoch"] + 1, state["best"], state["stale"]

    config = {
        **vars(args),
        "architecture": ARCHITECTURE,
        "classes": names,
        "recordings_train": len(train),
        "recordings_val": len(val),
        "recordings_test": len(test),
    }
    (out / "config.json").write_text(json.dumps(config, indent=1))

    with (out / "metrics.jsonl").open("a") as log:
        for epoch in range(start, args.epochs):
            if stale >= args.patience:
                print("early stop", flush=True)
                break
            model.train()
            began, total, seen = time.time(), 0.0, 0
            for waves, targets in loader:
                waves, targets = waves.to(device), targets.to(device)
                with torch.no_grad():
                    images = spec_augment(extractor(waves))
                loss = criterion(model(images), targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(waves)
                seen += len(waves)
            scheduler.step()
            logits = predict_recordings(extractor, model, val, device)
            score = macro_accuracy(logits, y_val)
            row = {
                "epoch": epoch,
                "train_loss": total / seen,
                "val_macro": score,
                "seconds": time.time() - began,
            }
            print(row, flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
            stale = 0 if score > best else stale + 1
            if score > best:
                best = score
                torch.save(model.state_dict(), out / "best.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best": best,
                    "stale": stale,
                },
                last,
            )

    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True))
    np.savez_compressed(
        out / "predictions.npz",
        logits_val=predict_recordings(extractor, model, val, device),
        labels_val=y_val,
        logits_test=predict_recordings(extractor, model, test, device),
        labels_test=y_test,
        observation_test=test["observation_id"].to_numpy(),
    )
    print(f"best validation macro accuracy: {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
