"""Perceptual deduplication, so one image cannot appear in both train and test."""

from __future__ import annotations

import numpy as np


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def duplicate_mask(hashes: list[int], max_distance: int = 4) -> np.ndarray:
    """Mark each image that is near-identical to an earlier one.

    Comparing all pairs is quadratic, so hashes are bucketed by four 16-bit bands. By the pigeonhole
    principle, two 64-bit hashes differing in at most 3 bits must share a band. At distance 4 a pair
    could differ once per band and be missed, which is an accepted rare miss.
    """
    buckets: dict[tuple[int, int], list[int]] = {}
    duplicates = np.zeros(len(hashes), dtype=bool)
    for index, value in enumerate(hashes):
        bands = [(band, (value >> (16 * band)) & 0xFFFF) for band in range(4)]
        seen: set[int] = set()
        for key in bands:
            seen.update(buckets.get(key, ()))
        if any(hamming(value, hashes[other]) <= max_distance for other in seen):
            duplicates[index] = True
            continue
        for key in bands:
            buckets.setdefault(key, []).append(index)
    return duplicates
