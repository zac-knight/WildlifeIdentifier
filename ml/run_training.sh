#!/usr/bin/env bash
# Runs the whole modelling sequence after data/build_dataset.py has finished. Every training step
# resumes after an interruption, and a step that fails stops the sequence so that later results
# can never be built on a missing earlier one.
set -euo pipefail
cd "$(dirname "$0")"

step() {
  echo "=== $(date '+%H:%M:%S') $*"
  uv run python "$@"
}

# A quick smoke test finds code problems in minutes, before hours of real training.
step training/train_image.py --name smoke --limit 400 --epochs 1 --batch-size 16 --workers 2
rm -rf ../models/runs/smoke

step training/embed_dinov2.py
step training/linear_probe.py
step training/train_prior.py --name prior_random --exclude split_random
step training/train_prior.py --name prior_geo --exclude split_geo

step training/train_image.py --name image_random --split split_random
step evaluation/predict_image.py --run image_random --split split_random
step training/train_image.py --name image_geo --split split_geo
step evaluation/predict_image.py --run image_geo --split split_geo
step training/train_image.py --name image_tail10 --split split_random --tail-shots 10
step evaluation/predict_image.py --run image_tail10 --split split_random

step evaluation/evaluate.py
step evaluation/evaluate_fewshot.py
step export/export_models.py image prior fewshot
echo "=== $(date '+%H:%M:%S') image pipeline finished"
