"""Image datasets and transforms shared by training, evaluation and export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from wildlens.config import DATA_DIR, MANIFEST_DIR

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 224


def train_transform() -> T.Compose:
    """Augmentation keeps colour changes mild, because plumage colour often separates species.

    Random crops vary framing and scale, and flips are safe since birds have no preferred facing.
    Random erasing simulates perches and foliage hiding part of the bird.
    """
    return T.Compose(
        [
            T.RandomResizedCrop(INPUT_SIZE, scale=(0.5, 1.0), ratio=(0.8, 1.25), antialias=True),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
            T.RandomErasing(p=0.15, scale=(0.02, 0.15)),
        ]
    )


def eval_transform() -> T.Compose:
    """The browser implements this exact pipeline, and a test checks they agree."""
    return T.Compose(
        [
            T.Resize(INPUT_SIZE, antialias=True),
            T.CenterCrop(INPUT_SIZE),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ]
    )


class PhotoDataset(Dataset):
    def __init__(self, paths: list[str], labels: np.ndarray, transform: T.Compose) -> None:
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(DATA_DIR / self.paths[index]) as image:
            return self.transform(image.convert("RGB")), int(self.labels[index])


def load_photos() -> pd.DataFrame:
    return pd.read_parquet(MANIFEST_DIR / "photos.parquet")


def class_names() -> list[str]:
    """Label order is fixed by sorted species name and shipped with each model."""
    species = pd.read_csv(MANIFEST_DIR / "species.csv")
    return sorted(species[species["role"] == "classifier"]["species"])


def select_training(
    frame: pd.DataFrame, split_column: str, tail_shots: int | None, split: str = "train"
) -> pd.DataFrame:
    """Rows for a split. With `tail_shots`, tail species keep only their k-shot training photos."""
    chosen = frame[(frame["role"] == "classifier") & (frame[split_column] == split)]
    if tail_shots is not None and split == "train":
        keep = ~chosen["tail"].astype(bool) | (chosen["tail_rank"] < tail_shots).fillna(False)
        chosen = chosen[keep.astype(bool)]
    return chosen


def balanced_weights(labels: np.ndarray, classes: int) -> np.ndarray:
    """Sampling weights that make each species equally likely per draw.

    This gives the classifier an approximately flat class prior, which is what makes it valid to
    multiply by the location prior later. Geographic frequency belongs in that prior, not here.
    """
    counts = np.bincount(labels, minlength=classes).astype(float)
    return 1.0 / np.maximum(counts[labels], 1.0)


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_dir(root: Path, name: str) -> Path:
    path = root / "runs" / name
    path.mkdir(parents=True, exist_ok=True)
    return path
