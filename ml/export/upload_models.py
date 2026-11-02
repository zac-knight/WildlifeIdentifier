"""Upload the browser models to a public Hugging Face model repository.

    HF_TOKEN=... uv run --with huggingface_hub python export/upload_models.py owner/repository

Needs a free Hugging Face account and a write token. Only finished browser files are uploaded.
"""

from __future__ import annotations

import argparse
import os

from huggingface_hub import HfApi

from wildlens.config import MODELS_DIR, ROOT

CARD = """---
license: cc-by-nc-4.0
tags: [onnx, image-classification, audio-classification, birds, australia]
---

# WildLens models

Browser models for [WildLens](https://github.com/zac-knight/WildlifeIdentifier), which identifies
Australian birds. Training data includes non-commercial Creative Commons licences, and sound models
also used share-alike recordings, so these models are for non-commercial use.

See the repository's `docs/` folder for model cards, the dataset card and evaluation results.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository")
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN to a Hugging Face write token first.")
    folder = MODELS_DIR / "web"
    (folder / "README.md").write_text(CARD)
    api = HfApi(token=token)
    api.create_repo(args.repository, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(
        repo_id=args.repository,
        folder_path=str(folder),
        ignore_patterns=["*.pre.onnx", "*.tmp"],
    )
    print(f"Set NEXT_PUBLIC_MODEL_BASE_URL=https://huggingface.co/{args.repository}/resolve/main")
    print(f"Model cards are in {ROOT / 'docs'}")


if __name__ == "__main__":
    main()
