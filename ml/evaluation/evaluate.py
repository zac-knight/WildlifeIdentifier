"""Compute every reported image, prior, trust and few-shot metric from saved predictions.

    uv run python evaluation/evaluate.py

Validation data is split into two folds with different jobs. Fold A fits temperatures, and fold B
tunes the fusion settings and thresholds. Reusing one fold for everything would make the selected
settings look better than they are. The test set is used for reporting only.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from scipy.special import log_softmax, softmax

from wildlens import imagedata as d
from wildlens import metrics as m
from wildlens.config import MANIFEST_DIR, MODELS_DIR, ROOT, SEED
from wildlens.prior import SpeciesPrior, day_of_year, encode

OUT = ROOT / "docs" / "metrics"
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
FLOORS = (1e-4, 1e-3, 1e-2, 3e-2)
SPECIES_PRECISION_TARGET = 0.9
GROUP_PRECISION_TARGET = 0.9


def prior_for_photos(run: str, frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    """Presence score per species at each photo's place and date."""
    model = SpeciesPrior(len(names))
    model.load_state_dict(torch.load(MODELS_DIR / "runs" / run / "best.pt", weights_only=True))
    model.eval()
    dates = pd.to_datetime(frame["date"])
    days = day_of_year(dates.dt.month.to_numpy(), dates.dt.day.to_numpy())
    with torch.inference_mode():
        encoded = encode(
            torch.as_tensor(frame["latitude"].to_numpy(np.float32)),
            torch.as_tensor(frame["longitude"].to_numpy(np.float32)),
            torch.as_tensor(days, dtype=torch.float32),
        )
        return torch.sigmoid(model(encoded)).numpy()


def fold_masks(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split validation in half within each species, so both folds cover all species."""
    rng = np.random.default_rng(SEED)
    fold_a = np.zeros(len(labels), dtype=bool)
    for label in np.unique(labels):
        rows = np.where(labels == label)[0]
        rng.shuffle(rows)
        fold_a[rows[: (len(rows) + 1) // 2]] = True
    return fold_a, ~fold_a


def precision_threshold(probabilities: np.ndarray, correct: np.ndarray, target: float) -> float:
    """The lowest confidence at which accepted answers still reach the target precision."""
    confidence = probabilities.max(axis=1)
    for threshold in np.linspace(0.05, 0.99, 95):
        kept = confidence >= threshold
        if kept.sum() >= 30 and correct[kept].mean() >= target:
            return float(threshold)
    return 0.99


def group_threshold(
    probabilities: np.ndarray, labels: np.ndarray, groups: list[str], target: float
) -> float:
    names = sorted(set(groups))
    lookup = {name: i for i, name in enumerate(names)}
    indices = np.array([lookup[group] for group in groups])
    totals = np.zeros((len(probabilities), len(names)))
    for column, group in enumerate(indices):
        totals[:, group] += probabilities[:, column]
    return precision_threshold(totals, totals.argmax(axis=1) == indices[labels], target)


def hierarchy_outcomes(
    probabilities: np.ndarray,
    labels: np.ndarray,
    genera: list[str],
    families: list[str],
    thresholds: dict[str, float],
) -> dict:
    """How often each answer level is used, and how often that answer is right."""
    outcome: dict[str, list[bool]] = {"species": [], "genus": [], "family": [], "unknown": []}
    for row, truth in zip(probabilities, labels, strict=True):
        level, value, _ = m.hierarchical_decision(
            row, genera, families, thresholds["species"], thresholds["genus"], thresholds["family"]
        )
        if level == "species":
            outcome[level].append(value == truth)
        elif level == "genus":
            outcome[level].append(value == genera[truth])
        elif level == "family":
            outcome[level].append(value == families[truth])
        else:
            outcome[level].append(False)
    total = len(labels)
    return {
        level: {
            "share": len(values) / total,
            "accuracy": float(np.mean(values)) if values and level != "unknown" else None,
        }
        for level, values in outcome.items()
    }


def evaluate_run(run: str, column: str, prior_run: str, names: list[str], photos: pd.DataFrame):
    data = np.load(MODELS_DIR / "runs" / run / "predictions.npz")
    by_id = photos.set_index("photo_id")
    val_rows = by_id.loc[data["photo_val"]]
    test_rows = by_id.loc[data["photo_test"]]
    logits_val, y_val = data["logits_val"], data["labels_val"].astype(int)
    logits_test, y_test = data["logits_test"], data["labels_test"].astype(int)
    fold_a, fold_b = fold_masks(y_val)

    temperature = m.fit_temperature(logits_val[fold_a], y_val[fold_a])
    calibrated_test = softmax(logits_test / temperature, axis=1)
    result: dict = {
        "split": column,
        "test": m.summary(logits_test, y_test),
        "temperature": temperature,
        "calibration": {
            "ece_before": m.expected_calibration_error(softmax(logits_test, axis=1), y_test),
            "ece_after": m.expected_calibration_error(calibrated_test, y_test),
            "reliability_before": m.reliability_bins(softmax(logits_test, axis=1), y_test),
            "reliability_after": m.reliability_bins(calibrated_test, y_test),
        },
        "confused_pairs": m.confused_pairs(logits_test, y_test, names),
    }

    # Fusion can only be measured on photos with usable coordinates and dates.
    located_val = (val_rows["latitude"].notna() & val_rows["date"].notna()).to_numpy()
    located_test = (test_rows["latitude"].notna() & test_rows["date"].notna()).to_numpy()
    prior_val = prior_for_photos(prior_run, val_rows[located_val], names)
    prior_test = prior_for_photos(prior_run, test_rows[located_test], names)
    log_image_val = log_softmax(logits_val / temperature, axis=1)[located_val]
    log_image_test = log_softmax(logits_test / temperature, axis=1)[located_test]
    y_val_located, y_test_located = y_val[located_val], y_test[located_test]
    tune = fold_b[located_val]

    best = (-1.0, 1.0, 1e-3)
    grid = []
    for weight in WEIGHTS:
        for floor in FLOORS:
            fused = m.fuse(log_image_val[tune], np.log(prior_val[tune] + 1e-12), weight, floor)
            score = m.macro_accuracy(fused, y_val_located[tune])
            grid.append({"weight": weight, "floor": floor, "val_macro": score})
            if score > best[0] + 1e-9:
                best = (score, weight, floor)
    _, weight, floor = best
    fused_val = m.fuse(log_image_val, np.log(prior_val + 1e-12), weight, floor)
    fused_test = m.fuse(log_image_test, np.log(prior_test + 1e-12), weight, floor)
    fit_rows = fold_a[located_val]
    fused_temperature = m.fit_temperature(fused_val[fit_rows], y_val_located[fit_rows])
    fused_probabilities = softmax(fused_test / fused_temperature, axis=1)
    result["fusion"] = {
        "weight": weight,
        "floor": floor,
        "fused_temperature": fused_temperature,
        "located_test_photos": int(located_test.sum()),
        "unlocated_test_photos": int((~located_test).sum()),
        "image_only": m.summary(log_image_test, y_test_located),
        "image_plus_prior": m.summary(fused_test, y_test_located),
        "prior_alone": m.summary(np.log(prior_test + 1e-12), y_test_located),
        "ece_after_fusion": m.expected_calibration_error(fused_probabilities, y_test_located),
        "grid": grid,
    }
    return result, {
        "data": data,
        "temperature": temperature,
        "fused_temperature": fused_temperature,
        "weight": weight,
        "floor": floor,
        "fold_a": fold_a,
        "fold_b": fold_b,
        "logits_val": logits_val,
        "y_val": y_val,
        "logits_test": logits_test,
        "y_test": y_test,
        "located_val": located_val,
        "fused_val": fused_val,
        "fused_test": fused_test,
        "y_val_located": y_val_located,
        "y_test_located": y_test_located,
    }


def trust(context: dict, names: list[str], genera: list[str], families: list[str]) -> dict:
    data = context["data"]
    temperature = context["temperature"]
    fold_b = context["fold_b"]
    logits_val, y_val = context["logits_val"], context["y_val"]
    logits_test, y_test = context["logits_test"], context["y_test"]

    def compare(in_logits, out_logits):
        return {
            "energy": m.ood_auroc(
                m.energy_score(in_logits, temperature), m.energy_score(out_logits, temperature)
            ),
            "msp": m.ood_auroc(
                m.msp_score(in_logits / temperature), m.msp_score(out_logits / temperature)
            ),
        }

    energy_threshold = m.threshold_at_tpr(m.energy_score(logits_val[fold_b], temperature), 0.95)
    msp_threshold = m.threshold_at_tpr(m.msp_score(logits_val[fold_b] / temperature), 0.95)
    near, far = data["logits_ood_test"], data["logits_far"]
    ood = {
        "energy_threshold": energy_threshold,
        "kept_known_test": float(
            (m.energy_score(logits_test, temperature) >= energy_threshold).mean()
        ),
        "near": {
            **compare(logits_test, near),
            "photos": len(near),
            "energy_false_accept": m.fpr_at_threshold(
                m.energy_score(near, temperature), energy_threshold
            ),
            "msp_false_accept": m.fpr_at_threshold(m.msp_score(near / temperature), msp_threshold),
        },
        "far": {
            **compare(logits_test, far),
            "photos": len(far),
            "energy_false_accept": m.fpr_at_threshold(
                m.energy_score(far, temperature), energy_threshold
            ),
            "msp_false_accept": m.fpr_at_threshold(m.msp_score(far / temperature), msp_threshold),
        },
    }

    # Thresholds for the fallback are set on fold B, aiming for 90% precision at each level.
    def thresholds_for(probabilities, labels):
        return {
            "species": precision_threshold(
                probabilities, probabilities.argmax(axis=1) == labels, SPECIES_PRECISION_TARGET
            ),
            "genus": group_threshold(probabilities, labels, genera, GROUP_PRECISION_TARGET),
            "family": group_threshold(probabilities, labels, families, GROUP_PRECISION_TARGET),
        }

    no_location = thresholds_for(softmax(logits_val[fold_b] / temperature, axis=1), y_val[fold_b])
    located_fold = fold_b[context["located_val"]]
    with_location = thresholds_for(
        softmax(context["fused_val"][located_fold] / context["fused_temperature"], axis=1),
        context["y_val_located"][located_fold],
    )
    outcomes_image = hierarchy_outcomes(
        softmax(logits_test / temperature, axis=1), y_test, genera, families, no_location
    )
    outcomes_fused = hierarchy_outcomes(
        softmax(context["fused_test"] / context["fused_temperature"], axis=1),
        context["y_test_located"],
        genera,
        families,
        with_location,
    )
    return {
        "ood": ood,
        "thresholds_no_location": no_location,
        "thresholds_with_location": with_location,
        "fallback_image_only": outcomes_image,
        "fallback_with_location": outcomes_fused,
    }


def tail_results(run: str, names: list[str], species: pd.DataFrame) -> dict | None:
    path = MODELS_DIR / "runs" / run / "predictions.npz"
    if not path.exists():
        return None
    data = np.load(path)
    tail_names = set(species[species["tail"]]["species"])
    tail = np.array([name in tail_names for name in names])
    labels = data["labels_test"].astype(int)
    logits = data["logits_test"]
    in_tail = tail[labels]
    return {
        "overall": m.summary(logits, labels),
        "tail_species": m.summary(logits[in_tail], labels[in_tail]),
        "head_species": m.summary(logits[~in_tail], labels[~in_tail]),
    }


def main() -> None:
    photos = d.load_photos()
    species = pd.read_csv(MANIFEST_DIR / "species.csv")
    names = d.class_names()
    meta = species.set_index("species")
    genera = [meta.loc[name, "genus"] for name in names]
    families = [
        meta.loc[name, "family"] if isinstance(meta.loc[name, "family"], str) else "Unknown"
        for name in names
    ]
    random_result, context = evaluate_run(
        "image_random", "split_random", "prior_random", names, photos
    )
    geo_result, _ = evaluate_run("image_geo", "split_geo", "prior_geo", names, photos)

    full = np.load(MODELS_DIR / "runs" / "image_random" / "predictions.npz")
    tail_names = set(species[species["tail"]]["species"])
    tail_mask = np.array([name in tail_names for name in names])
    labels = full["labels_test"].astype(int)
    payload = {
        "random": random_result,
        "geographic": geo_result,
        "trust": trust(context, names, genera, families),
        "tail_with_all_images": {
            "tail_species": m.summary(
                full["logits_test"][tail_mask[labels]], labels[tail_mask[labels]]
            )
        },
        "tail_capped_10": tail_results("image_tail10", names, species),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "image.json").write_text(json.dumps(payload, indent=1))
    print(
        json.dumps(
            {
                "random": random_result["test"],
                "geographic": geo_result["test"],
                "ece": random_result["calibration"]["ece_after"],
                "fusion": {
                    k: random_result["fusion"][k]
                    for k in ("weight", "floor", "image_only", "image_plus_prior")
                },
                "fusion_geo": {
                    k: geo_result["fusion"][k]
                    for k in ("weight", "floor", "image_only", "image_plus_prior")
                },
                "ood": payload["trust"]["ood"],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
