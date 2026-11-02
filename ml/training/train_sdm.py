"""Fit a species distribution model per species and project it onto future climates.

    uv run python training/train_sdm.py

Presences are GBIF occurrences within the Australasian window, thinned to one per climate cell.
Background points are other birds' records. Observers favour roads and cities, so using the same
biased sampling for background cancels much of that bias, which random points would not (the
"target-group" method). Random background is also run on a sample, to report whether this mattered.
"""

from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from wildlens import config as c
from wildlens.sdm import (
    BIOCLIM_NAMES,
    auc,
    boyce_index,
    change_agreement,
    climate_cell,
    novel_climate_mask,
    spatial_folds,
)

SCENARIOS = ("ssp245_2050", "ssp245_2070", "ssp585_2050", "ssp585_2070")
CLIMATE = c.DATA_DIR / "climate"
THREATENED_BLOCK_CELLS = 4  # 4 x 0.25 degrees: about 100 km at this latitude.
PARAMETERS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 10,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "num_threads": 8,
    "seed": c.SEED,
}
ROUNDS = 200


def load_grid(name: str, stat: str = "median") -> np.ndarray:
    return np.load(CLIMATE / f"{name}_{stat}.npz")["grid"]


class Lookup:
    """Reads climate values at coordinates, using the grid's affine transform."""

    def __init__(self, grid: np.ndarray) -> None:
        self.grid = grid
        a, _, left, _, e, top = np.load(CLIMATE / "transform.npy")
        self.a, self.left, self.e, self.top = a, left, e, top

    def __call__(self, latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
        row = np.floor((latitude - self.top) / self.e).astype(int)
        column = np.floor((longitude - self.left) / self.a).astype(int)
        rows, columns = self.grid.shape[:2]
        inside = (row >= 0) & (row < rows) & (column >= 0) & (column < columns)
        values = np.full((len(latitude), self.grid.shape[2]), np.nan, dtype=np.float32)
        values[inside] = self.grid[row[inside], column[inside]]
        return values


def fit(x: np.ndarray, y: np.ndarray) -> lgb.Booster:
    # Both classes get equal total weight, so results do not depend on the background sample size.
    weights = np.where(y == 1, 1.0, y.sum() / max(1, (y == 0).sum()))
    return lgb.train(PARAMETERS, lgb.Dataset(x, y, weight=weights), num_boost_round=ROUNDS)


def cross_validate(presence, background) -> dict | None:
    x = np.vstack([presence["x"], background["x"]])
    y = np.r_[np.ones(len(presence["x"])), np.zeros(len(background["x"]))]
    latitude = np.r_[presence["lat"], background["lat"]]
    longitude = np.r_[presence["lon"], background["lon"]]
    aucs, boyces = [], []
    for train, test in spatial_folds(latitude, longitude, 5.0, 5):
        if y[test].sum() < 5 or y[train].sum() < 10 or (y[test] == 0).sum() < 5:
            continue
        predicted = fit(x[train], y[train]).predict(x[test])
        aucs.append(auc(predicted[y[test] == 1], predicted[y[test] == 0]))
        value = boyce_index(predicted[y[test] == 1], predicted[y[test] == 0])
        if not np.isnan(value):
            boyces.append(value)
    if not aucs:
        return None
    return {
        "auc": float(np.mean(aucs)),
        "boyce": float(np.mean(boyces)) if boyces else None,
        "folds": len(aucs),
    }


def au_window(grid: np.ndarray) -> np.ndarray:
    a, _, left, _, e, top = np.load(CLIMATE / "transform.npy")
    row0 = int((c.AU_LAT[1] - top) / e)
    row1 = int((c.AU_LAT[0] - top) / e)
    column0 = int((c.AU_LON[0] - left) / a)
    column1 = int((c.AU_LON[1] - left) / a)
    return grid[row0:row1, column0:column1]


def predict_grid(model: lgb.Booster, grid: np.ndarray) -> np.ndarray:
    land = ~np.isnan(grid).any(axis=-1)
    output = np.full(grid.shape[:2], np.nan, dtype=np.float32)
    output[land] = model.predict(grid[land])
    return output


def coarsen_grid(values: np.ndarray, cells: int) -> np.ndarray:
    """Replaces each block with its mean, so threatened species are never mapped finely."""
    output = values.copy()
    for row in range(0, values.shape[0], cells):
        for column in range(0, values.shape[1], cells):
            block = values[row : row + cells, column : column + cells]
            if not np.isnan(block).all():
                output[row : row + cells, column : column + cells] = np.where(
                    np.isnan(block), np.nan, np.nanmean(block)
                )
    return output


def save_layer(path, suitability, robust, novel) -> None:
    """Red is suitability, green marks agreement across climate percentiles, blue marks land.

    Novel climates get a suitability value but lose the agreement flag and are stored in the alpha
    channel as 128 so the map can hatch them. Alpha is otherwise 255.
    """
    land = ~np.isnan(suitability)
    image = np.zeros((*suitability.shape, 4), dtype=np.uint8)
    image[..., 0] = np.nan_to_num(np.clip(suitability, 0, 1) * 255).astype(np.uint8)
    image[..., 1] = (robust & land).astype(np.uint8) * 255
    image[..., 2] = land.astype(np.uint8) * 255
    image[..., 3] = 255
    image[novel & land, 3] = 128
    Image.fromarray(image, "RGBA").save(path, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--compare-random", type=int, default=30)
    args = parser.parse_args()

    occurrences = pd.read_parquet(c.MANIFEST_DIR / "occurrences.parquet")
    occurrences = occurrences[
        occurrences["latitude"].between(*c.REGION_LAT)
        & occurrences["longitude"].between(*c.REGION_LON)
    ]
    species = pd.read_csv(c.MANIFEST_DIR / "species.csv")
    species = species[species["role"] == "classifier"]
    if args.limit:
        species = species.head(args.limit)

    baseline_grid = load_grid("present")
    lookup = Lookup(baseline_grid)
    all_lat = occurrences["latitude"].to_numpy()
    all_lon = occurrences["longitude"].to_numpy()
    all_x = lookup(all_lat, all_lon)
    ok = ~np.isnan(all_x).any(axis=1)
    occurrences, all_lat, all_lon, all_x = occurrences[ok], all_lat[ok], all_lon[ok], all_x[ok]
    print(f"records on land inside the climate grid: {len(all_x)}", flush=True)
    rng = np.random.default_rng(c.SEED)

    grids = {
        scenario: {stat: au_window(load_grid(scenario, stat)) for stat in ("median", "p10", "p90")}
        for scenario in SCENARIOS
    }
    baseline_au = au_window(baseline_grid)
    land_rows, land_columns = np.where(~np.isnan(baseline_grid).any(axis=-1))
    a, _, left, _, e, top = np.load(CLIMATE / "transform.npy")

    out = c.DATA_DIR / "sdm"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for position, item in enumerate(tqdm(list(species.itertuples()), desc="species")):
        mine = (occurrences["species"] == item.species).to_numpy()
        cells = climate_cell(all_lat[mine], all_lon[mine])
        _, first = np.unique(cells, return_index=True)
        presence_x = all_x[mine][first]
        if len(presence_x) < 20:
            rows.append({"species": item.species, "status": "too_few_records"})
            continue
        presence = {
            "x": presence_x,
            "lat": all_lat[mine][first],
            "lon": all_lon[mine][first],
        }
        others = np.where(~mine)[0]
        sample = rng.choice(others, min(5000, len(others)), replace=False)
        background = {"x": all_x[sample], "lat": all_lat[sample], "lon": all_lon[sample]}
        scores = cross_validate(presence, background)

        random_scores = None
        if position < args.compare_random:
            picks = rng.integers(0, len(land_rows), 5000)
            pick_rows, pick_columns = land_rows[picks], land_columns[picks]
            random_background = {
                "x": baseline_grid[pick_rows, pick_columns],
                "lat": top + (pick_rows + 0.5) * e,
                "lon": left + (pick_columns + 0.5) * a,
            }
            random_scores = cross_validate(presence, random_background)

        x = np.vstack([presence["x"], background["x"]])
        y = np.r_[np.ones(len(presence["x"])), np.zeros(len(background["x"]))]
        model = fit(x, y)
        present = predict_grid(model, baseline_au)
        # The threshold is the suitability that keeps 90% of known presences, a standard choice
        # that tolerates a few erroneous or unusual records.
        threshold = float(np.quantile(model.predict(presence["x"]), 0.1))

        threatened = bool(item.threatened)
        slug = item.species.lower().replace(" ", "-")
        folder = out / slug
        folder.mkdir(exist_ok=True)
        no_flag = np.zeros(present.shape, dtype=bool)
        display = coarsen_grid(present, THREATENED_BLOCK_CELLS) if threatened else present
        save_layer(folder / "present.png", display, ~np.isnan(present), no_flag)

        changes = {}
        base_area = int(np.nansum(present >= threshold))
        for scenario, members in grids.items():
            median = predict_grid(model, members["median"])
            low = predict_grid(model, members["p10"])
            high = predict_grid(model, members["p90"])
            robust = change_agreement(present, [median, low, high], threshold)
            novel = novel_climate_mask(members["median"], x)
            shown = coarsen_grid(median, THREATENED_BLOCK_CELLS) if threatened else median
            save_layer(folder / f"{scenario}.png", shown, robust, novel)
            future_area = int(np.nansum(median >= threshold))
            gained = np.nansum((median >= threshold) & (present < threshold))
            lost = np.nansum((median < threshold) & (present >= threshold))
            land = ~np.isnan(present)
            changes[scenario] = {
                "area_change_percent": (
                    100 * (future_area - base_area) / base_area if base_area else None
                ),
                "gained_cells": int(gained),
                "lost_cells": int(lost),
                "robust_share": float(robust[land].mean()),
                "novel_share": float(novel[land].mean()),
            }
        rows.append(
            {
                "species": item.species,
                "status": "ok",
                "presence_cells": len(presence_x),
                "threshold": threshold,
                "baseline_cells": base_area,
                "threatened": threatened,
                "coarsened": threatened,
                "cv": scores,
                "cv_random_background": random_scores,
                "changes": changes,
            }
        )
    (out / "summary.json").write_text(json.dumps(rows, indent=1))

    fitted = [row for row in rows if row.get("cv")]
    aucs = np.array([row["cv"]["auc"] for row in fitted])
    boyces = np.array([row["cv"]["boyce"] for row in fitted if row["cv"]["boyce"] is not None])
    pairs = [
        (row["cv"]["auc"], row["cv_random_background"]["auc"])
        for row in fitted
        if row.get("cv_random_background")
    ]
    result = {
        "species_attempted": len(rows),
        "species_with_cv": len(fitted),
        "species_too_few_records": sum(row["status"] == "too_few_records" for row in rows),
        "auc_median": float(np.median(aucs)),
        "auc_p10": float(np.quantile(aucs, 0.1)),
        "auc_p90": float(np.quantile(aucs, 0.9)),
        "boyce_median": float(np.median(boyces)),
        "boyce_p10": float(np.quantile(boyces, 0.1)),
        "boyce_p90": float(np.quantile(boyces, 0.9)),
        "target_group_vs_random_auc": {
            "species": len(pairs),
            "target_group_median": float(np.median([p[0] for p in pairs])) if pairs else None,
            "random_median": float(np.median([p[1] for p in pairs])) if pairs else None,
        },
        "variables": list(BIOCLIM_NAMES),
        "spatial_block_degrees": 5,
    }
    path = c.ROOT / "docs" / "metrics" / "sdm.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
