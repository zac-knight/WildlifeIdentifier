"""Train the location and date prior, with leak-free training data for each evaluation split.

    uv run python training/train_prior.py --name prior_random --exclude split_random
    uv run python training/train_prior.py --name prior_geo --exclude split_geo

Two leaks are closed here. GBIF republishes iNaturalist records, so a test photo's own coordinates
could enter training and inflate the ablation. Those records are removed by observation ID. For the
geographic split, every record inside a held-out block is also removed, whatever its source.

Threatened species have their coordinates coarsened before training. Otherwise anyone could query
this public model across a grid and rebuild a precise map of where they occur.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from wildlens import imagedata as d
from wildlens.config import GEO_BLOCK_DEGREES, MANIFEST_DIR, MODELS_DIR, SEED
from wildlens.prior import SpeciesPrior, assume_negative_loss, coarsen, day_of_year, encode
from wildlens.splits import block_id

THREATENED_GRID_DEGREES = 0.5
AU_BOX = {"lat": (-44.5, -9.0), "lon": (112.0, 155.0)}


def load_records(exclude: str) -> pd.DataFrame:
    records = pd.read_parquet(MANIFEST_DIR / "occurrences.parquet")
    photos = d.load_photos()
    photos = photos[photos["role"] == "classifier"]
    records = records[records["country"] == "AU"].dropna(subset=["month", "day"])
    records = records[(records["month"].between(1, 12)) & (records["day"].between(1, 31))]

    held_out = photos[photos[exclude].isin(["val", "test"])]
    ids = set(held_out["observation_id"].astype(str))
    before = len(records)
    records = records[~records["inat_observation_id"].isin(ids)]
    print(f"removed {before - len(records)} records matching held-out photos")

    if exclude == "split_geo":
        located = held_out.dropna(subset=["latitude", "longitude"])
        blocks = set(block_id(located["latitude"], located["longitude"], GEO_BLOCK_DEGREES))
        inside = block_id(records["latitude"], records["longitude"], GEO_BLOCK_DEGREES).isin(blocks)
        print(f"removed {int(inside.sum())} records inside held-out blocks")
        records = records[~inside]
    return records.copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--exclude", default="split_random")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--positive-weight", type=float, default=32.0)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    generator = np.random.default_rng(SEED)
    device = torch.device("cpu")  # Small MLP: the CPU is as fast as MPS once transfers are counted.
    names = d.class_names()
    index = {name: i for i, name in enumerate(names)}
    records = load_records(args.exclude)
    records = records[records["species"].isin(index)]

    species = pd.read_csv(MANIFEST_DIR / "species.csv")
    threatened = set(species[species["threatened"]]["species"])
    mask = records["species"].isin(threatened).to_numpy()
    latitude = records["latitude"].to_numpy(np.float32)
    longitude = records["longitude"].to_numpy(np.float32)
    latitude[mask] = coarsen(latitude[mask], THREATENED_GRID_DEGREES)
    longitude[mask] = coarsen(longitude[mask], THREATENED_GRID_DEGREES)
    days = day_of_year(records["month"].to_numpy(int), records["day"].to_numpy(int))
    labels = records["species"].map(index).to_numpy()

    tensors = {
        "lat": torch.as_tensor(latitude),
        "lon": torch.as_tensor(longitude),
        "day": torch.as_tensor(days, dtype=torch.float32),
        "y": torch.as_tensor(labels),
    }
    # Random locations are drawn from real observation sites, so "random" background stays on land.
    size = len(labels)
    print(f"training records: {size}, species: {len(np.unique(labels))}")

    model = SpeciesPrior(len(names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    for epoch in range(args.epochs):
        order = torch.randperm(size)
        total = 0.0
        model.train()
        for start in range(0, size - args.batch_size + 1, args.batch_size):
            batch = order[start : start + args.batch_size]
            observed = encode(tensors["lat"][batch], tensors["lon"][batch], tensors["day"][batch])
            sites = torch.randint(0, size, (len(batch),))
            dates = torch.as_tensor(generator.uniform(1, 365, len(batch)), dtype=torch.float32)
            background = encode(tensors["lat"][sites], tensors["lon"][sites], dates)
            loss = assume_negative_loss(
                model(observed),
                tensors["y"][batch],
                model(background),
                args.positive_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        scheduler.step()
        print(f"epoch {epoch}: loss {total / max(1, size // args.batch_size):.4f}", flush=True)

    out = d.run_dir(MODELS_DIR, args.name)
    torch.save(model.state_dict(), out / "best.pt")
    (out / "config.json").write_text(
        json.dumps({**vars(args), "classes": names, "records": size}, indent=1)
    )


if __name__ == "__main__":
    main()
