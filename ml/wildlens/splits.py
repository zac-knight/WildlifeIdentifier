"""Dataset splits, designed so that test scores measure generalisation and not leakage.

Random split: stratified by species and grouped by observer. A photographer's camera, habits and
local sites are shared across their photos, so they must not span train and test for a species.

Geographic split: whole spatial blocks are held out. Nearby observations share backgrounds and
habitat, so a random split cannot show whether the model has memorised them.

Tail split: 40 well-photographed species get artificially few training images. This is better than
using truly rare species because plenty of test images remain, making the few-shot result reliable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def assign_observers(
    counts: pd.Series, fractions: dict[str, float], rng: np.random.Generator
) -> dict[int, str]:
    """Greedily give each observer to the split furthest below its target size."""
    target = {name: fraction * counts.sum() for name, fraction in fractions.items()}
    filled = dict.fromkeys(fractions, 0)
    order = counts.sample(frac=1, random_state=int(rng.integers(2**31))).sort_values(
        ascending=False, kind="stable"
    )
    assignment: dict[int, str] = {}
    for observer, count in order.items():
        split = max(target, key=lambda name: target[name] - filled[name])
        assignment[observer] = split
        filled[split] += count
    return assignment


def random_split(
    frame: pd.DataFrame, seed: int, fractions: dict[str, float] | None = None
) -> pd.Series:
    fractions = fractions or {"train": 0.8, "val": 0.1, "test": 0.1}
    rng = np.random.default_rng(seed)
    result = pd.Series(index=frame.index, dtype="object")
    for _, group in frame.groupby("species", sort=True):
        counts = group.groupby("observer_id").size()
        assignment = assign_observers(counts, fractions, rng)
        result.loc[group.index] = group["observer_id"].map(assignment)
    return result


def block_id(latitude: pd.Series, longitude: pd.Series, degrees: float) -> pd.Series:
    row = np.floor(latitude / degrees).astype("Int64").astype("string")
    column = np.floor(longitude / degrees).astype("Int64").astype("string")
    return row + "_" + column


def geographic_split(
    frame: pd.DataFrame, seed: int, degrees: float, val: float = 0.1, test: float = 0.2
) -> pd.Series:
    """Hold out whole blocks. Records without a coordinate cannot join this split."""
    result = pd.Series(pd.NA, index=frame.index, dtype="object")
    located = frame.dropna(subset=["latitude", "longitude"])
    blocks = block_id(located["latitude"], located["longitude"], degrees)
    unique = np.array(sorted(blocks.unique()))
    np.random.default_rng(seed).shuffle(unique)
    test_count = max(1, round(len(unique) * test))
    val_count = max(1, round(len(unique) * val))
    assignment = dict.fromkeys(unique[:test_count], "test")
    assignment |= dict.fromkeys(unique[test_count : test_count + val_count], "val")
    result.loc[located.index] = blocks.map(lambda block: assignment.get(block, "train")).astype(
        "object"
    )
    return result


def choose_tail_species(ranked: list[str], tail_count: int, seed: int) -> list[str]:
    """Spread tail species evenly across rank, so they are not only the commonest or rarest."""
    rng = np.random.default_rng(seed)
    bins = np.array_split(np.array(ranked), tail_count)
    return sorted(str(rng.choice(group)) for group in bins)


def tail_shot_rank(frame: pd.DataFrame, tail: list[str], seed: int) -> pd.Series:
    """Rank training photos within each tail species.

    `rank < k` gives the k-shot set, and because ranks are fixed, the 5-shot set is nested inside
    the 10-shot set. Larger k is therefore a clean comparison and not a different random sample.
    """
    rng = np.random.default_rng(seed)
    ranks = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    candidates = frame[frame["species"].isin(tail) & (frame["split_random"] == "train")]
    for _, group in candidates.groupby("species", sort=True):
        order = rng.permutation(len(group))
        ranks.loc[group.index] = order
    return ranks
