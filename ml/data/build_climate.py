"""Download bioclimatic inputs from NASA NEX-GDDP-CMIP6 (CC0, no account required).

Reads only the required window of each monthly Cloud-Optimised GeoTIFF over HTTP, so the full global
files are never downloaded. The source already provides ensemble medians and 10th and 90th
percentiles across its climate models, which serve here as the projection uncertainty range.

The baseline is the models' own 1995-2014 historical run. Comparing it with their future runs means
that the change reflects the models' climate shift rather than any gap between models and weather
stations.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from tqdm import tqdm

from wildlens.config import DATA_DIR, REGION_LAT, REGION_LON
from wildlens.sdm import bioclim

BASE = "https://nex-gddp-cmip6-cog.s3.us-west-2.amazonaws.com/monthly"
VARIABLES = ("tas", "tasmax", "tasmin", "pr")
SCENARIOS = {
    "present": ("historical", 1995, 2014),
    "ssp245_2050": ("ssp245", 2041, 2060),
    "ssp245_2070": ("ssp245", 2061, 2080),
    "ssp585_2050": ("ssp585", 2041, 2060),
    "ssp585_2070": ("ssp585", 2061, 2080),
}
OUTPUT = DATA_DIR / "climate"


def read_month(stat: str, variable: str, scenario: str, year: int, month: int) -> np.ndarray:
    name = f"{variable}_month_ensemble-{stat}_{scenario}_{year}{month:02d}.tif"
    with rasterio.open(f"/vsicurl/{BASE}/CMIP6_ensemble_{stat}/{variable}/{name}") as source:
        window = from_bounds(
            REGION_LON[0], REGION_LAT[0], REGION_LON[1], REGION_LAT[1], source.transform
        )
        return source.read(1, window=window)


def climatology(stat: str, scenario: str, start: int, end: int) -> dict[str, np.ndarray]:
    """Average each calendar month across the 20 years, for each variable."""
    jobs = [
        (variable, year, month)
        for variable in VARIABLES
        for year in range(start, end + 1)
        for month in range(1, 13)
    ]

    def run(job: tuple[str, int, int]) -> np.ndarray:
        return read_month(stat, job[0], scenario, job[1], job[2])

    with ThreadPoolExecutor(16) as pool:
        arrays = list(tqdm(pool.map(run, jobs), total=len(jobs), desc=f"{stat} {scenario} {start}"))
    results: dict[str, list[list[np.ndarray]]] = {v: [[] for _ in range(12)] for v in VARIABLES}
    for (variable, _, month), array in zip(jobs, arrays, strict=True):
        results[variable][month - 1].append(array)
    return {
        variable: np.stack([np.mean(stack, axis=0) for stack in months]).astype(np.float32)
        for variable, months in results.items()
    }


def build(stat: str, name: str, scenario: str, start: int, end: int) -> Path:
    path = OUTPUT / f"{name}_{stat}.npz"
    if path.exists():
        return path
    monthly = climatology(stat, scenario, start, end)
    grid = bioclim(monthly["tas"], monthly["tasmax"], monthly["tasmin"], monthly["pr"])
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, grid=grid, annual_tas=monthly["tas"].mean(axis=0) - 273.15)
    temporary.replace(path)
    return path


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        f"/vsicurl/{BASE}/CMIP6_ensemble_median/tas/tas_month_ensemble-median_historical_200001.tif"
    ) as source:
        window = from_bounds(
            REGION_LON[0], REGION_LAT[0], REGION_LON[1], REGION_LAT[1], source.transform
        )
        transform = source.window_transform(window)
    np.save(OUTPUT / "transform.npy", np.array(transform)[:6])
    for name, (scenario, start, end) in SCENARIOS.items():
        for stat in ("median", "p10", "p90"):
            build(stat, name, scenario, start, end)
    print("climate grids ready")


if __name__ == "__main__":
    main()
