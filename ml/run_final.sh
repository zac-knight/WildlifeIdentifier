#!/usr/bin/env bash
# Runs after the image pipeline: audio training, then every published result and report.
set -euo pipefail
cd "$(dirname "$0")"

step() {
  echo "=== $(date '+%H:%M:%S') $*"
  uv run python "$@"
}

step training/train_audio.py --name audio_random
step evaluation/evaluate_audio.py
step export/export_models.py audio
step data/make_dataset_card.py
step export/publish_web_data.py
step evaluation/write_report.py
mkdir -p ../web/public/models
cp -R ../models/web/. ../web/public/models/
echo "=== $(date '+%H:%M:%S') everything finished"
