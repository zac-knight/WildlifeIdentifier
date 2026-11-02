from wildlens.config import PHOTO_LICENCES
from wildlens.records import (
    occurrence_row,
    parse_location,
    photo_row,
    prefer_downloaded,
    sound_row,
)

PHOTO_URL = "https://inaturalist-open-data.s3.amazonaws.com/photos/9/square.jpg?123"


def observation(**overrides):
    base = {
        "id": 1,
        "user": {"id": 2},
        "taxon": {"id": 3, "name": "Dacelo novaeguineae"},
        "location": "-33.5,151.25",
        "positional_accuracy": 10,
        "observed_on": "2026-01-02",
        "photos": [
            {"id": 9, "license_code": "cc-by-nc", "attribution": "(c) A", "url": PHOTO_URL},
        ],
    }
    return base | overrides


def test_obscured_coordinates_are_dropped() -> None:
    assert parse_location(observation(obscured=True)) == (None, None)


def test_imprecise_coordinates_are_dropped() -> None:
    assert parse_location(observation(positional_accuracy=50_000)) == (None, None)


def test_unknown_accuracy_is_kept() -> None:
    assert parse_location(observation(positional_accuracy=None)) == (-33.5, 151.25)


def test_photo_row_uses_medium_file_and_keeps_attribution() -> None:
    row = photo_row(observation(), PHOTO_LICENCES)
    assert row["source_url"].endswith("/photos/9/medium.jpg")
    assert row["attribution"] == "(c) A"
    assert row["licence"] == "cc-by-nc"


def test_photo_with_all_rights_reserved_is_rejected() -> None:
    photos = [{"id": 9, "license_code": None, "attribution": "(c) A", "url": PHOTO_URL}]
    assert photo_row(observation(photos=photos), PHOTO_LICENCES) is None


def test_no_derivatives_photo_is_rejected() -> None:
    photos = [{"id": 9, "license_code": "cc-by-nd", "attribution": "(c) A", "url": PHOTO_URL}]
    assert photo_row(observation(photos=photos), PHOTO_LICENCES) is None


def test_photo_outside_open_data_bucket_is_rejected() -> None:
    photos = [
        {
            "id": 9,
            "license_code": "cc-by",
            "attribution": "(c) A",
            "url": "https://static.inaturalist.org/photos/9/square.jpg",
        }
    ]
    assert photo_row(observation(photos=photos), PHOTO_LICENCES) is None


def test_hidden_sound_is_rejected() -> None:
    sounds = [
        {"id": 5, "license_code": "cc-by", "file_url": "https://x/5.m4a", "hidden": True},
    ]
    assert sound_row(observation(sounds=sounds), ("cc-by",)) is None


def test_inaturalist_occurrence_keeps_id_for_leak_exclusion() -> None:
    record = {
        "key": 1,
        "decimalLatitude": -30.0,
        "decimalLongitude": 150.0,
        "institutionCode": "iNaturalist",
        "catalogNumber": "333",
    }
    assert occurrence_row(record)["inat_observation_id"] == "333"


def test_other_occurrence_has_no_inaturalist_id() -> None:
    record = {
        "key": 1,
        "decimalLatitude": -30.0,
        "decimalLongitude": 150.0,
        "institutionCode": "CLO",
        "catalogNumber": "333",
    }
    assert occurrence_row(record)["inat_observation_id"] is None


def test_downloaded_files_come_first_and_order_is_kept() -> None:
    items = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]
    ordered = prefer_downloaded(items, lambda item: item["id"] in {2, 4})
    assert [item["id"] for item in ordered] == [2, 4, 1, 3]
