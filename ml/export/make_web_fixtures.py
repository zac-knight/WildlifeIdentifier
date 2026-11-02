"""Write test vectors computed by the real Python code, for the TypeScript tests to reproduce.

uv run python export/make_web_fixtures.py
"""

from __future__ import annotations

import base64
import json

import numpy as np
from PIL import Image
from scipy.special import log_softmax, softmax

from wildlens.config import ROOT
from wildlens.imagedata import eval_transform
from wildlens.metrics import energy_score, fuse, hierarchical_decision

OUTPUT = ROOT / "web" / "tests" / "fixtures" / "python_vectors.json"


def image_case(width: int, height: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    # Smooth random noise resembles photographic content more closely than per-pixel noise.
    coarse = rng.integers(0, 256, (height // 8 + 1, width // 8 + 1, 3), dtype=np.uint8)
    image = Image.fromarray(coarse).resize((width, height), Image.Resampling.BICUBIC)
    rgb = np.asarray(image)
    rgba = np.dstack([rgb, np.full((height, width), 255, dtype=np.uint8)])
    tensor = eval_transform()(image).numpy()
    return {
        "width": width,
        "height": height,
        "rgba_base64": base64.b64encode(rgba.tobytes()).decode(),
        # Storing the whole output is unnecessary. Every 37th value still detects any misalignment.
        "sample_step": 37,
        "sampled_output": [round(float(v), 5) for v in tensor.flatten()[::37]],
    }


def main() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=8) * 3
    audio = rng.normal(size=8) * 2
    prior = rng.random(8) ** 3
    temperature, audio_temperature, weight, floor = 1.3, 0.9, 0.7, 2e-2
    fused = fuse(
        log_softmax(logits / temperature)[None],
        np.log(prior)[None],
        weight,
        floor,
        log_softmax(audio / audio_temperature)[None],
    )[0]

    genera = ["A", "A", "B", "B", "C", "C", "D", "E"]
    families = ["F", "F", "F", "F", "G", "G", "G", "H"]
    decisions = []
    for probabilities in ([0.9, 0.1], [0.45, 0.45, 0.1], [0.3, 0.1, 0.3, 0.3], [0.2] * 5):
        values = np.zeros(8)
        values[: len(probabilities)] = probabilities
        level, group, confidence = hierarchical_decision(values, genera, families, 0.7, 0.7, 0.7)
        decisions.append(
            {
                "probabilities": values.tolist(),
                "level": level,
                "group": group,
                "confidence": confidence,
            }
        )

    payload = {
        "fusion": {
            "logits": logits.tolist(),
            "audio": audio.tolist(),
            "prior": prior.tolist(),
            "temperature": temperature,
            "audio_temperature": audio_temperature,
            "weight": weight,
            "floor": floor,
            "fused": fused.tolist(),
            "probabilities": softmax(fused).tolist(),
            "energy": float(energy_score(logits[None])[0]),
        },
        "hierarchy": {"genera": genera, "families": families, "cases": decisions},
        "images": [
            image_case(300, 200, 1),
            image_case(180, 260, 2),
            image_case(300, 225, 3),
            image_case(120, 90, 4),
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload))
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
