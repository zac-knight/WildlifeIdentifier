"""Turn raw API responses into manifest rows. Anything without a permitted licence is rejected."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from wildlens.config import INAT_PHOTO_HOST, MAX_POSITIONAL_ACCURACY_M


def parse_location(observation: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """Return a usable coordinate, or nothing when it is obscured or imprecise.

    iNaturalist randomises coordinates of many threatened species. Treating those as real points
    would corrupt the geographic split, so the photo stays usable but carries no location.
    """
    location = observation.get("location")
    accuracy = observation.get("positional_accuracy")
    if not location or observation.get("obscured"):
        return None, None
    if accuracy is not None and accuracy > MAX_POSITIONAL_ACCURACY_M:
        return None, None
    latitude, longitude = (float(part) for part in location.split(","))
    return latitude, longitude


def medium_photo_url(photo: Mapping[str, Any]) -> str | None:
    """Only files in iNaturalist's open data bucket are used."""
    url = photo.get("url") or ""
    if not url.startswith(INAT_PHOTO_HOST):
        return None
    return url.split("?")[0].replace("/square.", "/medium.")


def _base_row(observation: Mapping[str, Any]) -> dict[str, Any]:
    latitude, longitude = parse_location(observation)
    return {
        "observation_id": observation["id"],
        "observer_id": observation["user"]["id"],
        "inat_taxon_id": observation["taxon"]["id"],
        "species": observation["taxon"]["name"],
        "latitude": latitude,
        "longitude": longitude,
        "date": observation.get("observed_on"),
    }


def photo_row(observation: Mapping[str, Any], allowed: Iterable[str]) -> dict[str, Any] | None:
    """Keep one photo per observation: further photos of the same bird add little and could leak."""
    allowed = set(allowed)
    for photo in observation.get("photos") or []:
        url = medium_photo_url(photo)
        if url and (photo.get("license_code") or "").lower() in allowed:
            return {
                **_base_row(observation),
                "photo_id": photo["id"],
                "licence": photo["license_code"].lower(),
                "attribution": photo.get("attribution"),
                "source_url": url,
                "observation_url": f"https://www.inaturalist.org/observations/{observation['id']}",
            }
    return None


def sound_row(observation: Mapping[str, Any], allowed: Iterable[str]) -> dict[str, Any] | None:
    allowed = set(allowed)
    for sound in observation.get("sounds") or []:
        url = sound.get("file_url")
        if url and not sound.get("hidden") and (sound.get("license_code") or "").lower() in allowed:
            return {
                **_base_row(observation),
                "sound_id": sound["id"],
                "licence": sound["license_code"].lower(),
                "attribution": sound.get("attribution"),
                "source_url": url,
                "content_type": sound.get("file_content_type"),
                "observation_url": f"https://www.inaturalist.org/observations/{observation['id']}",
            }
    return None


def occurrence_row(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """A GBIF occurrence. Records from iNaturalist keep their ID so test data can be excluded."""
    latitude, longitude = record.get("decimalLatitude"), record.get("decimalLongitude")
    if latitude is None or longitude is None:
        return None
    institution = record.get("institutionCode")
    return {
        "gbif_id": record["key"],
        "dataset_key": record.get("datasetKey"),
        "latitude": latitude,
        "longitude": longitude,
        "uncertainty_m": record.get("coordinateUncertaintyInMeters"),
        "event_date": record.get("eventDate"),
        "month": record.get("month"),
        "day": record.get("day"),
        "year": record.get("year"),
        "licence": record.get("license"),
        "country": record.get("countryCode"),
        "inat_observation_id": record.get("catalogNumber")
        if institution == "iNaturalist"
        else None,
    }
