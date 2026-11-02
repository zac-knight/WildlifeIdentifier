"""Few-shot evaluation: do DINOv2 prototypes help tail species the classifier saw little of?

    uv run python evaluation/evaluate_fewshot.py

Uses the classifier that really was trained with only 10 images per tail species. Evaluating the
full model instead would flatter the classifier, since it has seen hundreds of images of them.
Prototype evidence only applies when the classifier is uncertain, as the product would use it.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.special import softmax

from wildlens import imagedata as d
from wildlens import metrics as m
from wildlens.config import MANIFEST_DIR, MODELS_DIR, ROOT

THRESHOLDS = (0.3, 0.5, 0.7, 0.9, 1.01)
WEIGHTS = (0.25, 0.5, 1.0, 2.0, 4.0)
SCALES = (10.0, 20.0, 40.0)


def build_prototypes(
    features: np.ndarray, labels: np.ndarray, classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """A prototype is the normalised mean of a species' embeddings: the simplest classifier that
    can work from a handful of images, with no parameters to overfit."""
    prototypes = np.zeros((classes, features.shape[1]), dtype=np.float32)
    present = np.zeros(classes, dtype=bool)
    for label in np.unique(labels):
        prototypes[label] = features[labels == label].mean(axis=0)
        present[label] = True
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    return prototypes / np.maximum(norms, 1e-8), present


def blend(
    logits: np.ndarray,
    similarity: np.ndarray,
    temperature: float,
    threshold: float,
    weight: float,
    scale: float,
) -> np.ndarray:
    """Add prototype evidence only for images where the classifier's confidence is low."""
    probabilities = softmax(logits / temperature, axis=1)
    scores = np.log(probabilities + 1e-12)
    uncertain = probabilities.max(axis=1) < threshold
    scores[uncertain] += weight * (similarity[uncertain] * scale)
    return scores


def main() -> None:
    frame = d.load_photos()
    species = pd.read_csv(MANIFEST_DIR / "species.csv")
    names = d.class_names()
    index = {name: i for i, name in enumerate(names)}
    embeddings = np.load(MODELS_DIR / "embeddings" / "dinov2.npz")
    features = embeddings["features"]
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    position = {photo: i for i, photo in enumerate(embeddings["photo_id"])}

    run = np.load(MODELS_DIR / "runs" / "image_tail10" / "predictions.npz")
    y_val, y_test = run["labels_val"].astype(int), run["labels_test"].astype(int)
    val_features = features[[position[p] for p in run["photo_val"]]]
    test_features = features[[position[p] for p in run["photo_test"]]]
    temperature = m.fit_temperature(run["logits_val"], y_val)

    tail_names = set(species[species["tail"]]["species"])
    tail_mask = np.array([name in tail_names for name in names])
    results: dict = {"shots": {}}
    train = d.select_training(frame, "split_random", 10)
    rows = np.array([position[p] for p in train["photo_id"]])
    labels = train["species"].map(index).to_numpy()
    prototypes, _ = build_prototypes(features[rows], labels, len(names))

    best = (-1.0, 1.01, 0.0, 10.0)
    # The tuning score weighs head and tail equally, so tail gains cannot hide head losses.
    val_tail = tail_mask[y_val]
    for threshold in THRESHOLDS:
        for weight in WEIGHTS:
            for scale in SCALES:
                scores = blend(
                    run["logits_val"],
                    val_features @ prototypes.T,
                    temperature,
                    threshold,
                    weight,
                    scale,
                )
                score = m.macro_accuracy(scores, y_val)
                if score > best[0] + 1e-9:
                    best = (score, threshold, weight, scale)
    _, threshold, weight, scale = best
    baseline = run["logits_test"]
    blended = blend(baseline, test_features @ prototypes.T, temperature, threshold, weight, scale)
    prototype_only = test_features @ prototypes.T
    test_tail = tail_mask[y_test]

    def group(scores: np.ndarray) -> dict:
        return {
            "all": m.summary(scores, y_test),
            "tail": m.summary(scores[test_tail], y_test[test_tail]),
            "head": m.summary(scores[~test_tail], y_test[~test_tail]),
        }

    results["settings"] = {
        "uncertain_below": threshold,
        "weight": weight,
        "scale": scale,
        "temperature": temperature,
        "val_tail_share": float(val_tail.mean()),
    }
    results["classifier_alone"] = group(baseline)
    results["classifier_plus_prototypes"] = group(blended)
    results["prototypes_alone"] = group(prototype_only)

    # Pure prototype results at each shot count need no retraining, so all are reported.
    for shots in (5, 10, 20):
        subset = d.select_training(frame, "split_random", shots)
        subset_rows = np.array([position[p] for p in subset["photo_id"]])
        built, _ = build_prototypes(
            features[subset_rows], subset["species"].map(index).to_numpy(), len(names)
        )
        results["shots"][str(shots)] = m.summary(
            (test_features @ built.T)[test_tail], y_test[test_tail]
        )

    # The product keeps all available images, so prototypes are rebuilt from the full set.
    final = d.select_training(frame, "split_random", None)
    final_rows = np.array([position[p] for p in final["photo_id"]])
    final_prototypes, _ = build_prototypes(
        features[final_rows], final["species"].map(index).to_numpy(), len(names)
    )
    out = MODELS_DIR / "fewshot"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "prototypes.npy", final_prototypes)
    (ROOT / "docs" / "metrics" / "fewshot.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({k: results[k] for k in results if k != "shots"}, indent=1))


if __name__ == "__main__":
    main()
