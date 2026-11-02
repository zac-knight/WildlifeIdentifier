"""Project-wide constants and paths. Scripts read these so a rerun cannot silently change scope."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("WILDLENS_DATA_DIR", ROOT / "data")).resolve()
MODELS_DIR = Path(os.environ.get("WILDLENS_MODELS_DIR", ROOT / "models")).resolve()
MANIFEST_DIR = DATA_DIR / "manifests"
CACHE_DIR = DATA_DIR / "cache"
PHOTO_DIR = DATA_DIR / "photos"
SOUND_DIR = DATA_DIR / "sounds"

CONTACT = os.environ.get("WILDLENS_CONTACT", "github.com/zac-knight/WildlifeIdentifier")
USER_AGENT = f"WildLens/0.1 (non-commercial portfolio project; {CONTACT})"

INAT_API = "https://api.inaturalist.org/v1"
INAT_PHOTO_HOST = "https://inaturalist-open-data.s3.amazonaws.com/photos/"
GBIF_API = "https://api.gbif.org/v1"

INAT_AUSTRALIA_PLACE_ID = 6744
INAT_BIRDS_TAXON = "Aves"
GBIF_BIRDS_CLASS_KEY = 212

# Photos: the brief permits CC0, CC BY and CC BY-NC.
PHOTO_LICENCES = ("cc0", "cc-by", "cc-by-nc")
# Sounds: the owner also approved share-alike. No-derivatives files are never used.
SOUND_LICENCES = ("cc0", "cc-by", "cc-by-nc", "cc-by-sa", "cc-by-nc-sa")

TARGET_SPECIES = 200
TAIL_SPECIES = 40
OOD_SPECIES = 40
MAX_PHOTOS_PER_SPECIES = 300
MAX_OOD_PHOTOS_PER_SPECIES = 40
MAX_SOUNDS_PER_SPECIES = 100
# Tail species are artificially capped at these training sizes for the few-shot evaluation.
TAIL_SHOTS = (5, 10, 20)
OCCURRENCE_PAGES_AU = 8
OCCURRENCE_PAGES_REGION = 3
OCCURRENCE_PAGE_SIZE = 300
# GBIF search stops paging at this offset. Larger sets must be sampled within it.
GBIF_MAX_OFFSET = 100_000
MAX_POSITIONAL_ACCURACY_M = 5_000
# Occurrence window around Australia. Fitting climate models beyond the border avoids truncating
# the climate niche of species that also live in New Guinea, Indonesia or New Zealand.
REGION_LAT = (-55.0, 10.0)
REGION_LON = (90.0, 180.0)
AU_LAT = (-44.0, -9.0)
AU_LON = (112.0, 154.5)
GEO_BLOCK_DEGREES = 2.0
SEED = 20261103
