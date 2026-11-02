"""Copy the small data files the website needs from local artefacts into web/.

    uv run python export/publish_web_data.py

Precise occurrence coordinates are never published. Only species facts, credits and aggregate
evaluation results are written here, along with the coarse climate grid and SDM layers.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd

from wildlens import config as c

WEB = c.ROOT / "web"
OUT = WEB / "data"
METRICS = c.ROOT / "docs" / "metrics"


def slug(name: str) -> str:
    return name.lower().replace(" ", "-")


def clean(value):
    return None if pd.isna(value) else value


def publish_species() -> None:
    species = pd.read_csv(c.MANIFEST_DIR / "species.csv")
    species = species[species["role"] == "classifier"].sort_values("species")
    sdm_path = c.DATA_DIR / "sdm" / "summary.json"
    sdm = {}
    if sdm_path.exists():
        sdm = {row["species"]: row for row in json.loads(sdm_path.read_text())}
    photos_path = c.MANIFEST_DIR / "sounds_ready.parquet"
    sound_counts = {}
    if photos_path.exists():
        sound_counts = pd.read_parquet(photos_path).groupby("species").size().to_dict()
    rows = []
    for item in species.itertuples():
        summary = sdm.get(item.species, {})
        rows.append(
            {
                "name": item.species,
                "slug": slug(item.species),
                "commonName": clean(item.common_name),
                "genus": clean(item.genus),
                "family": clean(item.family),
                "order": clean(item.order),
                "observations": int(item.observation_count),
                "iucn": clean(item.iucn_code),
                "threatened": bool(item.threatened),
                "photoUrl": clean(item.photo_url),
                "photoAttribution": clean(item.photo_attribution),
                "recordings": int(sound_counts.get(item.species, 0)),
                "sdm": None
                if summary.get("status") != "ok"
                else {
                    "threshold": summary["threshold"],
                    "auc": (summary.get("cv") or {}).get("auc"),
                    "boyce": (summary.get("cv") or {}).get("boyce"),
                    "coarsened": summary["coarsened"],
                    "changes": summary["changes"],
                },
            }
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "species.json").write_text(json.dumps(rows, separators=(",", ":")))
    print(f"species: {len(rows)}")


def credit_name(attribution: str | None) -> str | None:
    """Reduce iNaturalist's "(c) Name, some rights reserved (CC BY-NC)" to the person's name."""
    if not attribution:
        return None
    text = attribution.split(",")[0]
    for prefix in ("(c) ", "©"):
        text = text.removeprefix(prefix)
    return text.strip() or None


def publish_credits() -> None:
    """Everyone whose work trained the models is credited on the about page."""
    frames = []
    for name, kind in (("photos.parquet", "photos"), ("sounds_ready.parquet", "recordings")):
        path = c.MANIFEST_DIR / name
        if path.exists():
            frame = pd.read_parquet(path, columns=["attribution", "licence"])
            frame["kind"] = kind
            frames.append(frame)
    combined = pd.concat(frames)
    combined["person"] = combined["attribution"].map(credit_name)
    combined = combined.dropna(subset=["person"])
    people = (
        combined.groupby(["person", "kind"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .sort_values("person", key=lambda names: names.str.casefold())
    )
    records = [
        {
            "n": row["person"],
            "p": int(row.get("photos", 0)),
            "r": int(row.get("recordings", 0)),
        }
        for _, row in people.iterrows()
    ]
    licences = combined.groupby(["kind", "licence"]).size().unstack(fill_value=0)
    payload = {
        "people": records,
        "licences": {
            kind: {licence: int(count) for licence, count in row.items() if count}
            for kind, row in licences.iterrows()
        },
    }
    (OUT / "credits.json").write_text(json.dumps(payload, separators=(",", ":")))
    print(f"credited people: {len(records)}")


def publish_metrics() -> None:
    combined = {}
    for path in sorted(METRICS.glob("*.json")):
        combined[path.stem] = json.loads(path.read_text())
    (OUT / "metrics.json").write_text(json.dumps(combined, indent=1))
    print(f"metric groups: {sorted(combined)}")


def publish_climate() -> None:
    """Coarse grid of projected annual temperature, for the click-a-point chart on species pages."""
    folder = c.DATA_DIR / "climate"
    if not (folder / "present_median.npz").exists():
        return
    a, _, left, _, e, top = np.load(folder / "transform.npy")
    row0, row1 = int((c.AU_LAT[1] - top) / e), int((c.AU_LAT[0] - top) / e)
    column0, column1 = int((c.AU_LON[0] - left) / a), int((c.AU_LON[1] - left) / a)
    block = 2  # Half-degree cells keep the file small. Temperature varies smoothly at this scale.

    def grid(name: str, stat: str) -> list:
        values = np.load(folder / f"{name}_{stat}.npz")["annual_tas"][row0:row1, column0:column1]
        rows, columns = values.shape[0] // block, values.shape[1] // block
        trimmed = values[: rows * block, : columns * block]
        reduced = trimmed.reshape(rows, block, columns, block)
        with np.errstate(invalid="ignore"):
            means = np.nanmean(reduced, axis=(1, 3))
        return [
            [None if np.isnan(value) else round(float(value), 1) for value in row] for row in means
        ]

    scenarios = ("ssp245_2050", "ssp245_2070", "ssp585_2050", "ssp585_2070")
    payload = {
        "top": float(top + row0 * e),
        "left": float(left + column0 * a),
        "cell": float(a * block),
        "present": grid("present", "median"),
        "scenarios": {
            name: {stat: grid(name, stat) for stat in ("median", "p10", "p90")}
            for name in scenarios
        },
    }
    (WEB / "public" / "data").mkdir(parents=True, exist_ok=True)
    (WEB / "public" / "data" / "climate.json").write_text(
        json.dumps(payload, separators=(",", ":"))
    )
    print("climate grid written")


def publish_sdm_layers() -> None:
    source = c.DATA_DIR / "sdm"
    if not source.exists():
        return
    target = WEB / "public" / "data" / "sdm"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    count = 0
    for folder in source.iterdir():
        if folder.is_dir():
            shutil.copytree(folder, target / folder.name)
            count += 1
    size = sum(path.stat().st_size for path in target.rglob("*.png"))
    print(f"SDM layers: {count} species, {size / 1e6:.1f} MB")


def main() -> None:
    publish_species()
    publish_credits()
    publish_metrics()
    publish_climate()
    publish_sdm_layers()


if __name__ == "__main__":
    main()
