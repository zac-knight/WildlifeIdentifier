import numpy as np
import pandas as pd

from wildlens.dedupe import duplicate_mask
from wildlens.splits import (
    choose_tail_species,
    geographic_split,
    random_split,
    tail_shot_rank,
)


def sample_frame(rows: int = 3000, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "species": rng.choice([f"s{i}" for i in range(10)], rows),
            "observer_id": rng.integers(0, 300, rows),
            "latitude": rng.uniform(-40, -10, rows),
            "longitude": rng.uniform(115, 150, rows),
        }
    )


def test_observer_never_spans_splits_within_a_species() -> None:
    frame = sample_frame()
    frame["split"] = random_split(frame, seed=3)
    assert frame.groupby(["species", "observer_id"])["split"].nunique().max() == 1


def test_random_split_hits_target_fractions() -> None:
    frame = sample_frame()
    shares = random_split(frame, seed=3).value_counts(normalize=True)
    assert abs(shares["train"] - 0.8) < 0.02
    assert abs(shares["test"] - 0.1) < 0.02


def test_random_split_is_deterministic() -> None:
    frame = sample_frame()
    assert random_split(frame, seed=3).equals(random_split(frame, seed=3))


def test_geographic_blocks_never_span_splits() -> None:
    frame = sample_frame()
    frame["split"] = geographic_split(frame, seed=3, degrees=2.0)
    blocks = (frame["latitude"] // 2).astype(str) + "_" + (frame["longitude"] // 2).astype(str)
    assert frame.groupby(blocks)["split"].nunique().max() == 1
    assert set(frame["split"]) == {"train", "val", "test"}


def test_records_without_coordinates_leave_the_geographic_split() -> None:
    frame = sample_frame()
    frame.loc[frame.index[:5], "latitude"] = np.nan
    assert geographic_split(frame, seed=3, degrees=2.0).iloc[:5].isna().all()


def test_tail_species_are_spread_across_ranks() -> None:
    ranked = [f"s{i}" for i in range(200)]
    chosen = choose_tail_species(ranked, 40, seed=3)
    positions = sorted(ranked.index(name) for name in chosen)
    assert len(set(chosen)) == 40
    assert positions[0] < 5
    assert positions[-1] >= 195


def test_tail_shot_sets_are_nested() -> None:
    frame = sample_frame()
    frame["split_random"] = random_split(frame, seed=3)
    ranks = tail_shot_rank(frame, ["s1"], seed=3)
    assert ranks[frame["species"] != "s1"].isna().all()
    assert ranks[frame["split_random"] != "train"].isna().all()
    five = set(ranks[ranks < 5].index)
    ten = set(ranks[ranks < 10].index)
    assert len(five) == 5 and len(ten) == 10 and five < ten


def test_near_duplicates_are_marked_and_distinct_kept() -> None:
    base = 0x0F0F_0F0F_0F0F_0F0F
    near = base ^ 0b111
    different = base ^ 0xFFFF_FFFF_FFFF_FFFF
    assert duplicate_mask([base, near, different]).tolist() == [False, True, False]
