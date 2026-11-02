"""Fine-tune the image classifier. Interrupted runs resume automatically from the last epoch.

    uv run python training/train_image.py --name image_random --split split_random
    uv run python training/train_image.py --name image_geo --split split_geo
    uv run python training/train_image.py --name image_tail10 --split split_random --tail-shots 10

Why EfficientNetV2-B2: 9M parameters is about 18 MB at fp16, which suits a browser download.
Its global pool feeds a single linear layer, so a Class Activation Map is mathematically exact.
ConvNeXt normalises after pooling, which would make its map an approximation.
"""

from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import timm
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from wildlens import imagedata as d
from wildlens.config import MODELS_DIR, SEED
from wildlens.metrics import macro_accuracy

ARCHITECTURE = "tf_efficientnetv2_b2.in1k"


def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    with torch.inference_mode():
        for images, targets in loader:
            logits.append(model(images.to(device)).cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", default="split_random")
    parser.add_argument("--tail-shots", type=int)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, help="Use only this many rows, for smoke tests")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = d.device()
    out = d.run_dir(MODELS_DIR, args.name)

    frame = d.load_photos()
    names = d.class_names()
    index = {name: i for i, name in enumerate(names)}
    train = d.select_training(frame, args.split, args.tail_shots)
    val = d.select_training(frame, args.split, None, "val")
    if args.limit:
        train = train.sample(min(args.limit, len(train)), random_state=SEED)
        val = val.sample(min(args.limit // 4, len(val)), random_state=SEED)
    y_train = train["species"].map(index).to_numpy()
    y_val = val["species"].map(index).to_numpy()

    loader_options = {
        "num_workers": args.workers,
        "persistent_workers": args.workers > 0,
        "pin_memory": device.type == "cuda",
    }
    # One epoch draws as many images as the set holds, with species balanced per draw.
    weights = torch.as_tensor(d.balanced_weights(y_train, len(names)), dtype=torch.double)
    train_loader = DataLoader(
        d.PhotoDataset(train["file_path"].tolist(), y_train, d.train_transform()),
        batch_size=args.batch_size,
        sampler=WeightedRandomSampler(weights, len(train), replacement=True),
        drop_last=True,
        **loader_options,
    )
    val_loader = DataLoader(
        d.PhotoDataset(val["file_path"].tolist(), y_val, d.eval_transform()),
        batch_size=args.batch_size * 2,
        **loader_options,
    )

    model = timm.create_model(ARCHITECTURE, pretrained=True, num_classes=len(names)).to(device)
    # Pretrained layers get a lower rate. They need refining, while the new head starts from random.
    head = list(model.get_classifier().parameters())
    head_ids = {id(parameter) for parameter in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": args.lr / 5}, {"params": head, "lr": args.lr}],
        weight_decay=0.05,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    # Label smoothing discourages overconfidence on hard pairs. Calibration is fitted afterwards.
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    # Mixed precision speeds up CUDA. Autocast on MPS is less reliable, so it stays off there.
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start, best, stale = 0, -1.0, 0
    last_path, best_path = out / "last.pt", out / "best.pt"
    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start, best, stale = state["epoch"] + 1, state["best"], state["stale"]
        torch.set_rng_state(state["rng"])
        print(f"resumed at epoch {start}", flush=True)

    config = {**vars(args), "architecture": ARCHITECTURE, "classes": names, "train": len(train)}
    (out / "config.json").write_text(json.dumps(config, indent=1))

    with (out / "metrics.jsonl").open("a") as log:
        for epoch in range(start, args.epochs):
            if stale >= args.patience:
                print("early stop", flush=True)
                break
            model.train()
            began, total, seen = time.time(), 0.0, 0
            for step, (images, targets) in enumerate(train_loader):
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, enabled=use_amp):
                    loss = criterion(model(images), targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total += loss.item() * len(images)
                seen += len(images)
                if step % 100 == 0:
                    print(f"epoch {epoch} step {step}/{len(train_loader)}", flush=True)
            scheduler.step()
            logits, labels = evaluate(model, val_loader, device)
            score = macro_accuracy(logits, labels)
            row = {
                "epoch": epoch,
                "train_loss": total / seen,
                "val_macro": score,
                "seconds": time.time() - began,
            }
            print(row, flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
            # Macro accuracy decides stopping, so common species cannot mask a falling tail.
            stale = 0 if score > best else stale + 1
            if score > best:
                best = score
                torch.save(model.state_dict(), best_path)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best": best,
                    "stale": stale,
                    "rng": torch.get_rng_state(),
                },
                last_path,
            )
    (out / "done.json").write_text(json.dumps({"best_val_macro": best}))
    print(f"best validation macro accuracy: {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
