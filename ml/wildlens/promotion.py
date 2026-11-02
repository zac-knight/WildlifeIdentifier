"""Rules for whether a retrained model may replace the current one."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scores:
    top1: float
    macro: float
    geographic_macro: float


@dataclass(frozen=True)
class Decision:
    promote: bool
    reasons: tuple[str, ...]


def should_promote(current: Scores, candidate: Scores, tolerance: float = 0.002) -> Decision:
    """A candidate must not regress on any fixed test metric.

    A small tolerance allows for noise, as sampling alone moves accuracy by a few tenths of a point.
    All three checks must pass, so a gain on common species cannot buy a loss on rarer ones or in
    unseen regions. Feedback data never enters the test sets, so this bar cannot quietly shift.
    """
    reasons = []
    for name in ("top1", "macro", "geographic_macro"):
        before, after = getattr(current, name), getattr(candidate, name)
        if after < before - tolerance:
            reasons.append(f"{name} fell from {before:.4f} to {after:.4f}")
    return Decision(not reasons, tuple(reasons))


def validate_feedback_rows(rows: list[dict], known: set[str]) -> list[dict]:
    """Keep only reviewed, approved labels for known species. User feedback alone is not ground
    truth, so nothing unreviewed can reach training."""
    return [
        row
        for row in rows
        if row.get("review_status") == "approved"
        and (row.get("reviewed_species") or row.get("correct_species")) in known
        and row.get("consented_image_path")
    ]
