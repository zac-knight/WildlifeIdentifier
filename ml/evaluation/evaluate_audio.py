"""Report sound model metrics, and photo plus sound fusion where the same observation has both.

    uv run python evaluation/evaluate_audio.py

Paired observations are the only honest test of combining sight and sound. Their test membership
comes from each model's own split, so a pair counts only if both its photo and recording are test
data. If few exist, the result is reported with its size so that it is not over-read.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.special import log_softmax

from wildlens import imagedata as d
from wildlens import metrics as m
from wildlens.config import MANIFEST_DIR, MODELS_DIR, ROOT

MIN_PAIRS_FOR_MACRO = 50


def main() -> None:
    audio = np.load(MODELS_DIR / "runs" / "audio_random" / "predictions.npz")
    labels = audio["labels_test"].astype(int)
    logits = audio["logits_test"]
    temperature = m.fit_temperature(audio["logits_val"], audio["labels_val"].astype(int))
    frame = pd.read_parquet(MANIFEST_DIR / "sounds_ready.parquet")
    per_species = frame.groupby("species").size()
    result: dict = {
        "test": m.summary(logits, labels),
        "temperature": temperature,
        "ece_after": m.expected_calibration_error(
            np.exp(log_softmax(logits / temperature, axis=1)), labels
        ),
        "recordings_total": len(frame),
        "species_with_recordings": int(per_species.size),
        "median_recordings_per_species": float(per_species.median()),
        "confused_pairs": m.confused_pairs(logits, labels, d.class_names(), 10),
    }

    image = np.load(MODELS_DIR / "runs" / "image_random" / "predictions.npz")
    image_temperature = m.fit_temperature(image["logits_val"], image["labels_val"].astype(int))
    photos = d.load_photos().set_index("photo_id").loc[image["photo_test"]]
    photo_row = {obs: i for i, obs in enumerate(photos["observation_id"])}
    pairs = [
        (position, photo_row[obs])
        for position, obs in enumerate(audio["observation_test"])
        if obs in photo_row
    ]
    fusion: dict = {"paired_test_observations": len(pairs)}
    if pairs:
        sound_rows, image_rows = (np.array(values) for values in zip(*pairs, strict=True))
        truth = labels[sound_rows]
        log_sound = log_softmax(logits[sound_rows] / temperature, axis=1)
        log_photo = log_softmax(image["logits_test"][image_rows] / image_temperature, axis=1)
        fusion |= {
            "photo": m.summary(log_photo, truth),
            "sound": m.summary(log_sound, truth),
            "photo_plus_sound": m.summary(log_photo + log_sound, truth),
            "reliable": len(pairs) >= MIN_PAIRS_FOR_MACRO,
        }
    result["fusion"] = fusion
    (ROOT / "docs" / "metrics" / "audio.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "confused_pairs"}, indent=1))


if __name__ == "__main__":
    main()
