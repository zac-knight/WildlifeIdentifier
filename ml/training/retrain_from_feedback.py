"""Retrain with reviewed feedback, and promote the model only if it does not regress.

    uv run python training/retrain_from_feedback.py --feedback reviewed.jsonl --candidate retrain_1

Steps: read reviewed feedback, add approved consented images to the training set only, retrain, get
scores on the fixed test sets, then compare with the current model. Public image storage is off by
default, so rows need a `consented_image_path` you have supplied. Without such rows, this stops
rather than spending hours training an identical model.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from wildlens import imagedata as d
from wildlens.config import DATA_DIR, MANIFEST_DIR, MODELS_DIR, ROOT
from wildlens.metrics import macro_accuracy, topk_accuracy
from wildlens.promotion import Scores, should_promote, validate_feedback_rows


def scores_for(run: str) -> Scores:
    data = np.load(MODELS_DIR / "runs" / run / "predictions.npz")
    labels = data["labels_test"].astype(int)
    geo = np.load(MODELS_DIR / "runs" / "image_geo" / "predictions.npz")
    return Scores(
        top1=topk_accuracy(data["logits_test"], labels, 1),
        macro=macro_accuracy(data["logits_test"], labels),
        geographic_macro=macro_accuracy(geo["logits_test"], geo["labels_test"].astype(int)),
    )


def merge(photos: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    """New rows join training only. The test sets stay frozen, so comparisons remain fair."""
    additions = []
    for number, row in enumerate(rows):
        species = row.get("reviewed_species") or row["correct_species"]
        additions.append(
            {
                "file_path": row["consented_image_path"],
                "species": species,
                "photo_id": -(number + 1),
                "observer_id": -(number + 1),
                "observation_id": -(number + 1),
                "role": "classifier",
                "split_random": "train",
                "split_geo": pd.NA,
                "tail": False,
                "licence": "user-consent",
                "attribution": "WildLens user, with consent",
                "source_url": "feedback",
            }
        )
    return pd.concat([photos, pd.DataFrame(additions)], ignore_index=True)


def run_step(*arguments: str) -> None:
    subprocess.run([sys.executable, *arguments], check=True, cwd=ROOT / "ml")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--candidate", default="retrain_candidate")
    parser.add_argument("--current", default="image_random")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.feedback.read_text().splitlines() if line.strip()]
    usable = validate_feedback_rows(rows, set(d.class_names()))
    print(f"usable feedback rows: {len(usable)} of {len(rows)}")
    if not usable:
        print("Nothing to train on, so the current model stays.")
        return
    missing = [r for r in usable if not (DATA_DIR / r["consented_image_path"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} consented images are missing from {DATA_DIR}.")

    manifest = MANIFEST_DIR / "photos.parquet"
    backup = MANIFEST_DIR / "photos.before_retrain.parquet"
    shutil.copy2(manifest, backup)
    try:
        merge(pd.read_parquet(manifest), usable).to_parquet(manifest, index=False)
        run_step("training/train_image.py", "--name", args.candidate, "--epochs", str(args.epochs))
        run_step("evaluation/predict_image.py", "--run", args.candidate)
    finally:
        # The manifest is always restored, even when training fails.
        backup.replace(manifest)

    decision = should_promote(scores_for(args.current), scores_for(args.candidate))
    result = {"candidate": args.candidate, "promote": decision.promote, "why": decision.reasons}
    (MODELS_DIR / "runs" / args.candidate / "promotion.json").write_text(json.dumps(result))
    if decision.promote:
        print("PROMOTE: run export/export_models.py with this run's weights to publish it.")
    else:
        print("KEEP CURRENT MODEL:", "; ".join(decision.reasons))


if __name__ == "__main__":
    main()
