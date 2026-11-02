"""Regenerate the WildLens dataset. Every step is resumable and safe to rerun.

uv run python data/build_dataset.py                 # everything
uv run python data/build_dataset.py --steps species observations
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from wildlens import config as c
from wildlens.dedupe import duplicate_mask
from wildlens.http import PoliteClient, Throttle
from wildlens.records import occurrence_row, photo_row, sound_row
from wildlens.splits import choose_tail_species, geographic_split, random_split, tail_shot_rank

STEPS = (
    "species",
    "observations",
    "photos",
    "dedupe",
    "occurrences",
    "splits",
    "far_ood",
    "sounds",
)
THREATENED = {"VU", "EN", "CR", "EW", "EX"}
# iNaturalist adopts new names sooner than the GBIF backbone. These map each label to the name GBIF
# uses for occurrence lookups only. The label shown to users stays the iNaturalist name.
GBIF_NAME_OVERRIDES = {
    "Trichoglossus concinnus": "Glossopsitta concinna",
    "Thinornis melanops": "Elseyornis melanops",
    "Anarhynchus ruficapillus": "Charadrius ruficapillus",
    "Tachyspiza fasciata": "Accipiter fasciatus",
    "Tachyspiza cirrocephala": "Accipiter cirrocephalus",
    "Butorides atricapilla": "Butorides striata",
    "Ardea coromanda": "Bubulcus coromandus",
    "Cinnyris frenatus": "Cinnyris jugularis",
}
FAR_OOD_TAXA = ("Mammalia", "Reptilia", "Amphibia", "Insecta", "Plantae", "Fungi")


def inat() -> PoliteClient:
    # Staff guidance is about 1 request per second. 1.1 s keeps a small margin.
    return PoliteClient(c.CACHE_DIR / "inat", interval=1.1)


def gbif() -> PoliteClient:
    return PoliteClient(c.CACHE_DIR / "gbif", interval=0.2)


def step_species() -> None:
    """Rank species by licensed research-grade observations, then attach GBIF taxonomy."""
    wanted = c.TARGET_SPECIES + c.OOD_SPECIES
    results: list[dict] = []
    client = inat()
    page = 1
    while len(results) < wanted:
        data = client.get_json(
            f"{c.INAT_API}/observations/species_counts",
            {
                "place_id": c.INAT_AUSTRALIA_PLACE_ID,
                "iconic_taxa": c.INAT_BIRDS_TAXON,
                "quality_grade": "research",
                "photo_license": ",".join(c.PHOTO_LICENCES),
                "per_page": 200,
                "page": page,
            },
        )
        if not data["results"]:
            break
        # Hybrids are not species, so they cannot be classes or have a distribution of their own.
        results += [r for r in data["results"] if r["taxon"].get("rank") == "species"]
        page += 1

    backbone = gbif()
    results = results[:wanted]
    # Family and genus come from iNaturalist, so taxonomy always agrees with the labels.
    ancestry: dict[int, dict[str, str]] = {}
    for start in range(0, len(results), 30):
        ids = ",".join(str(r["taxon"]["id"]) for r in results[start : start + 30])
        for taxon in client.get_json(f"{c.INAT_API}/taxa/{ids}", {"per_page": 30})["results"]:
            ancestry[taxon["id"]] = {a["rank"]: a["name"] for a in taxon.get("ancestors", [])}
    rows = []
    for item in tqdm(results, desc="taxonomy"):
        taxon = item["taxon"]
        lookup = GBIF_NAME_OVERRIDES.get(taxon["name"], taxon["name"])
        match = backbone.get_json(
            f"{c.GBIF_API}/species/match",
            {"name": lookup, "class": "Aves", "strict": "true"},
        )
        key = match.get("speciesKey") or match.get("usageKey")
        red_list: dict = {}
        if key:
            try:
                red_list = backbone.get_json(f"{c.GBIF_API}/species/{key}/iucnRedListCategory")
            except httpx.HTTPStatusError as error:
                if error.response.status_code not in (204, 404):
                    raise
            except ValueError:  # An empty body means GBIF holds no assessment for this taxon.
                red_list = {}
        rows.append(
            {
                "species": taxon["name"],
                "common_name": taxon.get("preferred_common_name"),
                "inat_taxon_id": taxon["id"],
                "observation_count": item["count"],
                "gbif_key": key,
                "genus": ancestry.get(taxon["id"], {}).get("genus") or taxon["name"].split()[0],
                "family": ancestry.get(taxon["id"], {}).get("family"),
                "order": ancestry.get(taxon["id"], {}).get("order"),
                "gbif_match": match.get("matchType"),
                "iucn_code": red_list.get("code"),
                "photo_url": (taxon.get("default_photo") or {}).get("medium_url"),
                "photo_attribution": (taxon.get("default_photo") or {}).get("attribution"),
            }
        )
    frame = pd.DataFrame(rows).sort_values("observation_count", ascending=False, kind="stable")
    frame["threatened"] = frame["iucn_code"].isin(THREATENED)
    frame["role"] = ["classifier"] * c.TARGET_SPECIES + ["ood"] * (len(frame) - c.TARGET_SPECIES)
    head = frame[frame["role"] == "classifier"]["species"].tolist()
    tail = set(choose_tail_species(head, c.TAIL_SPECIES, c.SEED))
    frame["tail"] = frame["species"].isin(tail)
    # Near-OOD species are split, so the threshold is never tuned on the species used to report it.
    ood = frame[frame["role"] == "ood"]["species"].tolist()
    ood_val = set(np.random.default_rng(c.SEED).permutation(ood)[: len(ood) // 2].tolist())
    frame["ood_split"] = [
        ("val" if s in ood_val else "test") if s in ood else None for s in frame["species"]
    ]
    c.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(c.MANIFEST_DIR / "species.csv", index=False)
    print(f"species: {len(frame)} ({len(head)} classifier, {frame['threatened'].sum()} threatened)")


def observation_pages(
    client: PoliteClient, species_row: pd.Series, extra: dict, wanted: int
) -> Iterable[dict]:
    """Page by descending ID, so results are stable across reruns and the cache stays valid.

    Taking the newest records alone would bias towards recent years and heavy recent users, so
    `order_by=random` is not used (results would change every call and defeat the cache). Instead,
    pages are spread evenly across the species' whole ID-ordered list.
    """
    params = {
        "place_id": c.INAT_AUSTRALIA_PLACE_ID,
        "taxon_id": int(species_row["inat_taxon_id"]),
        "quality_grade": "research",
        "per_page": 200,
        "order_by": "id",
        "order": "desc",
        **extra,
    }
    first = client.get_json(f"{c.INAT_API}/observations", {**params, "page": 1})
    total = first["total_results"]
    pages_needed = int(np.ceil(wanted * 1.3 / 200))
    # The API rejects paging past 10,000 results.
    last_page = max(1, min(int(np.ceil(total / 200)), 50))
    yield from first["results"]
    for page in np.unique(np.linspace(1, last_page, pages_needed + 1).round().astype(int))[1:]:
        yield from client.get_json(f"{c.INAT_API}/observations", {**params, "page": int(page)})[
            "results"
        ]


def step_observations() -> None:
    species = pd.read_csv(c.MANIFEST_DIR / "species.csv")
    client = inat()
    rng = np.random.default_rng(c.SEED)
    rows: list[dict] = []
    for _, row in tqdm(species.iterrows(), total=len(species), desc="observations"):
        cap = (
            c.MAX_PHOTOS_PER_SPECIES
            if row["role"] == "classifier"
            else c.MAX_OOD_PHOTOS_PER_SPECIES
        )
        extra = {"photo_license": ",".join(c.PHOTO_LICENCES), "photos": "true"}
        found: dict[int, dict] = {}
        for observation in observation_pages(client, row, extra, cap):
            parsed = photo_row(observation, c.PHOTO_LICENCES)
            if parsed:
                found[parsed["observation_id"]] = parsed
        items = list(found.values())
        rng.shuffle(items)
        rows += [{**item, "species": row["species"], "role": row["role"]} for item in items[:cap]]
    frame = pd.DataFrame(rows)
    frame["file_path"] = [
        f"photos/{row.species.replace(' ', '_')}/{row.photo_id}.jpg" for row in frame.itertuples()
    ]
    frame.to_parquet(c.MANIFEST_DIR / "photos_raw.parquet", index=False)
    print(f"photo records: {len(frame)}")


def download_file(client: httpx.Client, url: str, target: Path) -> bool:
    if target.exists():
        return True
    for attempt in range(5):
        try:
            response = client.get(url)
            if response.status_code == 404:
                return False
            response.raise_for_status()
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_bytes(response.content)
            temporary.replace(target)
            return True
        except (httpx.TransportError, httpx.HTTPStatusError):
            time.sleep(2**attempt)
    return False


def step_photos() -> None:
    """Files come from the open data bucket, built for bulk access, so iNaturalist's own servers
    carry no load. Eight connections is a modest level for S3."""
    frame = pd.read_parquet(c.MANIFEST_DIR / "photos_raw.parquet")
    client = httpx.Client(headers={"User-Agent": c.USER_AGENT}, timeout=60, follow_redirects=True)

    def fetch(row: tuple[str, str]) -> bool:
        return download_file(client, row[0], c.DATA_DIR / row[1])

    with ThreadPoolExecutor(8) as pool:
        pairs = list(zip(frame["source_url"], frame["file_path"], strict=True))
        ok = list(tqdm(pool.map(fetch, pairs), total=len(pairs), desc="photos"))
    print(f"downloaded {sum(ok)} of {len(ok)}")


def photo_hash(path: Path) -> int | None:
    try:
        with Image.open(path) as image:
            return int(str(imagehash.phash(image.convert("RGB"))), 16)
    except (OSError, ValueError):
        return None  # Missing or corrupt files are dropped.


def step_dedupe() -> None:
    frame = pd.read_parquet(c.MANIFEST_DIR / "photos_raw.parquet")
    with ThreadPoolExecutor(8) as pool:
        paths = [c.DATA_DIR / path for path in frame["file_path"]]
        hashes = list(tqdm(pool.map(photo_hash, paths), total=len(paths), desc="hashing"))
    frame["phash"] = [format(h, "016x") if h is not None else None for h in hashes]
    readable = frame[frame["phash"].notna()].copy()
    # Hashing across all species also catches one image uploaded under two names.
    mask = duplicate_mask([int(h, 16) for h in readable["phash"]])
    kept = readable[~mask]
    kept.to_parquet(c.MANIFEST_DIR / "photos_deduped.parquet", index=False)
    print(
        f"unreadable: {len(frame) - len(readable)}, duplicates: {int(mask.sum())}, "
        f"kept: {len(kept)}"
    )


def occurrence_offsets(count: int, pages: int) -> list[int]:
    """Spread pages evenly through the reachable records instead of taking only the first ones."""
    reachable = min(count, c.GBIF_MAX_OFFSET) - c.OCCURRENCE_PAGE_SIZE
    if reachable <= 0:
        return [0]
    return sorted({int(x) for x in np.linspace(0, reachable, pages)})


def fetch_occurrences(client: PoliteClient, key: int, base: dict, pages: int) -> list[dict]:
    url = f"{c.GBIF_API}/occurrence/search"
    params = {
        "taxonKey": key,
        "hasCoordinate": "true",
        "hasGeospatialIssue": "false",
        "occurrenceStatus": "PRESENT",
        "year": "1990,2026",
        **base,
    }
    count = client.get_json(url, {**params, "limit": 0})["count"]
    rows = []
    for offset in occurrence_offsets(count, pages):
        data = client.get_json(url, {**params, "limit": c.OCCURRENCE_PAGE_SIZE, "offset": offset})
        rows += [occurrence_row(record) for record in data["results"]]
    return [row for row in rows if row]


def step_occurrences() -> None:
    species = pd.read_csv(c.MANIFEST_DIR / "species.csv")
    species = species[species["role"] == "classifier"].dropna(subset=["gbif_key"])
    client = gbif()
    rows = []
    window = {
        "decimalLatitude": f"{c.REGION_LAT[0]},{c.REGION_LAT[1]}",
        "decimalLongitude": f"{c.REGION_LON[0]},{c.REGION_LON[1]}",
    }
    for _, row in tqdm(species.iterrows(), total=len(species), desc="occurrences"):
        key = int(row["gbif_key"])
        records = fetch_occurrences(client, key, {"country": "AU"}, c.OCCURRENCE_PAGES_AU)
        records += fetch_occurrences(client, key, window, c.OCCURRENCE_PAGES_REGION)
        rows += [{**record, "species": row["species"]} for record in records]
    frame = pd.DataFrame(rows).drop_duplicates(subset=["gbif_id", "species"])
    frame["inat_observation_id"] = frame["inat_observation_id"].astype("string")
    frame["licence"] = frame["licence"].astype("string")
    frame.to_parquet(c.MANIFEST_DIR / "occurrences.parquet", index=False)
    print(f"occurrences: {len(frame)}, AU: {(frame['country'] == 'AU').sum()}")


def step_splits() -> None:
    frame = pd.read_parquet(c.MANIFEST_DIR / "photos_deduped.parquet")
    species = pd.read_csv(c.MANIFEST_DIR / "species.csv")
    meta = species[["species", "gbif_key", "genus", "family", "threatened", "tail", "ood_split"]]
    frame = frame.merge(meta, on="species", how="left")
    classifier = frame["role"] == "classifier"
    frame["split_random"] = pd.Series(pd.NA, index=frame.index, dtype="object")
    frame["split_geo"] = pd.Series(pd.NA, index=frame.index, dtype="object")
    frame.loc[classifier, "split_random"] = random_split(frame[classifier], c.SEED)
    frame.loc[classifier, "split_geo"] = geographic_split(
        frame[classifier], c.SEED, c.GEO_BLOCK_DEGREES
    )
    tail = species[species["tail"]]["species"].tolist()
    frame["tail_rank"] = tail_shot_rank(frame, tail, c.SEED)
    frame.to_parquet(c.MANIFEST_DIR / "photos.parquet", index=False)
    print(frame[classifier].groupby("split_random").size().to_dict())
    print(frame[classifier].groupby("split_geo").size().to_dict())


def step_far_ood() -> None:
    """Non-bird photos for the easier half of the unknown-input evaluation."""
    client = inat()
    rows = []
    for name in FAR_OOD_TAXA:
        data = client.get_json(
            f"{c.INAT_API}/observations",
            {
                "place_id": c.INAT_AUSTRALIA_PLACE_ID,
                "iconic_taxa": name,
                "quality_grade": "research",
                "photo_license": ",".join(c.PHOTO_LICENCES),
                "photos": "true",
                "per_page": 200,
                "order_by": "id",
                "order": "desc",
            },
        )
        parsed = [photo_row(o, c.PHOTO_LICENCES) for o in data["results"]]
        rows += [{**row, "taxon_group": name} for row in parsed if row]
    frame = pd.DataFrame(rows)
    frame["file_path"] = [f"photos/_far_ood/{row.photo_id}.jpg" for row in frame.itertuples()]
    http = httpx.Client(headers={"User-Agent": c.USER_AGENT}, timeout=60, follow_redirects=True)

    def fetch(pair: tuple[str, str]) -> bool:
        return download_file(http, pair[0], c.DATA_DIR / pair[1])

    with ThreadPoolExecutor(8) as pool:
        pairs = list(zip(frame["source_url"], frame["file_path"], strict=True))
        ok = list(tqdm(pool.map(fetch, pairs), total=len(pairs), desc="far OOD"))
    frame = frame[ok]
    frame.to_parquet(c.MANIFEST_DIR / "photos_far_ood.parquet", index=False)
    print(f"far OOD photos: {len(frame)}")


def step_sounds() -> None:
    species = pd.read_csv(c.MANIFEST_DIR / "species.csv")
    species = species[species["role"] == "classifier"]
    client = inat()
    rng = np.random.default_rng(c.SEED)
    rows: list[dict] = []
    for _, row in tqdm(species.iterrows(), total=len(species), desc="sound records"):
        extra = {"sound_license": ",".join(c.SOUND_LICENCES), "sounds": "true"}
        found: dict[int, dict] = {}
        for observation in observation_pages(client, row, extra, c.MAX_SOUNDS_PER_SPECIES):
            parsed = sound_row(observation, c.SOUND_LICENCES)
            if parsed:
                found[parsed["observation_id"]] = parsed
        items = list(found.values())
        rng.shuffle(items)
        rows += [{**item, "species": row["species"]} for item in items[: c.MAX_SOUNDS_PER_SPECIES]]
    frame = pd.DataFrame(rows)
    frame["file_path"] = [
        f"sounds/{row.species.replace(' ', '_')}/{row.sound_id}"
        + f".{row.source_url.split('?')[0].rsplit('.', 1)[-1]}"
        for row in frame.itertuples()
    ]
    frame["split_random"] = random_split(frame, c.SEED)
    frame.to_parquet(c.MANIFEST_DIR / "sounds.parquet", index=False)
    print(f"sound records: {len(frame)} for {frame['species'].nunique()} species")

    # Sound files live on iNaturalist servers, so they download one at a time, at the API rate.
    http = httpx.Client(headers={"User-Agent": c.USER_AGENT}, timeout=120, follow_redirects=True)
    throttle = Throttle(1.1)
    missing = 0
    for row in tqdm(list(frame.itertuples()), desc="sound files"):
        target = c.DATA_DIR / row.file_path
        if target.exists():
            continue
        throttle.wait()
        if not download_file(http, row.source_url, target):
            missing += 1
    print(f"sound files missing: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", nargs="*", choices=STEPS, default=list(STEPS))
    arguments = parser.parse_args()
    actions = globals()
    for step in STEPS:
        if step in arguments.steps:
            print(f"== {step}")
            actions[f"step_{step}"]()


if __name__ == "__main__":
    main()
