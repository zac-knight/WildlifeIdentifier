"""Evaluation metrics and post-processing. All work on NumPy arrays, so they are simple to test."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, logsumexp, softmax
from sklearn.metrics import roc_auc_score


def topk_accuracy(logits: np.ndarray, labels: np.ndarray, k: int = 1) -> float:
    top = np.argpartition(-logits, kth=min(k, logits.shape[1] - 1) - 1, axis=1)[:, :k]
    return float((top == labels[:, None]).any(axis=1).mean())


def per_class_accuracy(logits: np.ndarray, labels: np.ndarray) -> dict[int, float]:
    predicted = logits.argmax(axis=1)
    return {int(c): float((predicted[labels == c] == c).mean()) for c in np.unique(labels)}


def macro_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Each species counts equally, so common species cannot hide failure on rarer ones."""
    return float(np.mean(list(per_class_accuracy(logits, labels).values())))


def summary(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(labels)),
        "species": int(len(np.unique(labels))),
        "top1": topk_accuracy(logits, labels, 1),
        "top5": topk_accuracy(logits, labels, 5),
        "macro": macro_accuracy(logits, labels),
    }


def confused_pairs(
    logits: np.ndarray, labels: np.ndarray, names: list[str], limit: int = 20
) -> list[dict]:
    predicted = logits.argmax(axis=1)
    pairs: dict[tuple[int, int], int] = {}
    for truth, guess in zip(labels, predicted, strict=True):
        if truth != guess:
            key = (min(int(truth), int(guess)), max(int(truth), int(guess)))
            pairs[key] = pairs.get(key, 0) + 1
    ordered = sorted(pairs.items(), key=lambda item: -item[1])[:limit]
    return [{"a": names[a], "b": names[b], "errors": count} for (a, b), count in ordered]


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature scaling has one parameter, so it can correct confidence on a modest validation
    set without changing which species ranks first, unlike richer calibrators."""

    def loss(log_temperature: float) -> float:
        probabilities = log_softmax(logits / np.exp(log_temperature), axis=1)
        return float(-probabilities[np.arange(len(labels)), labels].mean())

    result = minimize_scalar(loss, bounds=(-3, 3), method="bounded")
    return float(np.exp(result.x))


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 15
) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        inside = (confidence > low) & (confidence <= high)
        if inside.any():
            total += inside.mean() * abs(correct[inside].mean() - confidence[inside].mean())
    return float(total)


def reliability_bins(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> list[dict]:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        inside = (confidence > low) & (confidence <= high)
        if inside.any():
            rows.append(
                {
                    "low": float(low),
                    "high": float(high),
                    "n": int(inside.sum()),
                    "confidence": float(confidence[inside].mean()),
                    "accuracy": float(correct[inside].mean()),
                }
            )
    return rows


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Higher means more in-distribution. Unlike the top probability, this uses every logit, so it
    can separate 'all classes weak' from 'one class relatively strongest'."""
    return temperature * logsumexp(logits / temperature, axis=1)


def msp_score(logits: np.ndarray) -> np.ndarray:
    return softmax(logits, axis=1).max(axis=1)


def ood_auroc(in_scores: np.ndarray, out_scores: np.ndarray) -> float:
    labels = np.concatenate([np.ones(len(in_scores)), np.zeros(len(out_scores))])
    return float(roc_auc_score(labels, np.concatenate([in_scores, out_scores])))


def threshold_at_tpr(in_scores: np.ndarray, tpr: float = 0.95) -> float:
    """Accept a fixed share of known-species images. This is chosen on validation data only."""
    return float(np.quantile(in_scores, 1 - tpr))


def fpr_at_threshold(out_scores: np.ndarray, threshold: float) -> float:
    return float((out_scores >= threshold).mean())


def fuse(
    log_image: np.ndarray,
    log_prior: np.ndarray | None = None,
    weight: float = 1.0,
    floor: float = 1e-3,
    log_audio: np.ndarray | None = None,
) -> np.ndarray:
    """score(y) = log p(y|image) + log p(y|audio) + weight * log max(p(y|place, day), floor).

    Balanced training makes the classifier close to a likelihood with a flat class prior, so adding
    a location prior does not count that information twice. The floor stops the prior from ruling
    out a genuine vagrant entirely. Evidence is assumed conditionally independent given species.
    """
    score = log_image.copy()
    if log_audio is not None:
        score = score + log_audio
    if log_prior is not None:
        score = score + weight * np.maximum(log_prior, np.log(floor))
    return score


def hierarchical_decision(
    probabilities: np.ndarray,
    genus_of: list[str],
    family_of: list[str],
    species_threshold: float,
    genus_threshold: float,
    family_threshold: float,
) -> tuple[str, int | str | None, float]:
    """Report the most specific level that is confident enough, and otherwise admit uncertainty."""
    best = int(probabilities.argmax())
    if probabilities[best] >= species_threshold:
        return "species", best, float(probabilities[best])
    for level, groups, threshold in (
        ("genus", genus_of, genus_threshold),
        ("family", family_of, family_threshold),
    ):
        totals: dict[str, float] = {}
        for probability, group in zip(probabilities, groups, strict=True):
            totals[group] = totals.get(group, 0.0) + float(probability)
        group, value = max(totals.items(), key=lambda item: item[1])
        if value >= threshold:
            return level, group, value
    return "unknown", None, float(probabilities[best])
