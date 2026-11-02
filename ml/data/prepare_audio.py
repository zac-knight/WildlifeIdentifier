"""Decode recordings to 32 kHz mono and cache the loudest clips as compact arrays.

    uv run python data/prepare_audio.py

Decoding in advance means training does not spend most of its time running ffmpeg.
"""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

from wildlens.audio import SAMPLE_RATE, energy_clips
from wildlens.config import DATA_DIR, MANIFEST_DIR

CLIPS_PER_RECORDING = 3
OUTPUT = DATA_DIR / "audio_clips"
MIN_SECONDS = 1.0


def decode(path) -> np.ndarray | None:
    """Sound files came from the web, so ffmpeg only reads them as data, with no shell involved."""
    command = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-t",
        "180",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "f32le",
        "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, timeout=120)
    if result.returncode != 0 or not result.stdout:
        return None
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def process(row) -> dict | None:
    target = OUTPUT / f"{row.sound_id}.npy"
    if target.exists():
        return {"sound_id": row.sound_id, "clip_path": str(target.relative_to(DATA_DIR))}
    source = DATA_DIR / row.file_path
    if not source.exists():
        return None
    waveform = decode(source)
    if waveform is None or len(waveform) < SAMPLE_RATE * MIN_SECONDS:
        return None
    clips = energy_clips(waveform, CLIPS_PER_RECORDING)
    # 16-bit storage is what the source audio could plausibly resolve, and it halves disk use.
    np.save(target, np.clip(clips * 32767, -32768, 32767).astype(np.int16))
    return {"sound_id": row.sound_id, "clip_path": str(target.relative_to(DATA_DIR))}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(MANIFEST_DIR / "sounds.parquet")
    rows = list(frame.itertuples())
    with ThreadPoolExecutor(8) as pool:
        results = list(tqdm(pool.map(process, rows), total=len(rows), desc="decoding"))
    ready = pd.DataFrame([result for result in results if result])
    merged = frame.merge(ready, on="sound_id")
    merged.to_parquet(MANIFEST_DIR / "sounds_ready.parquet", index=False)
    print(
        f"usable recordings: {len(merged)} of {len(frame)}, species: {merged['species'].nunique()}"
    )


if __name__ == "__main__":
    main()
