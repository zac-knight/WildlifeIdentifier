"""Measure how often AI-written species cards state something their sources do not support.

    uv run python evaluation/evaluate_grounding.py export-sample cards.jsonl sample.csv
    # A person fills the `supported` column with yes or no, by reading each claim and its source.
    uv run python evaluation/evaluate_grounding.py score sample.csv

Human judgement is the reference here. Asking another LLM to grade an LLM could share its mistakes.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from wildlens.config import ROOT, SEED


def export_sample(cards: Path, output: Path, count: int) -> None:
    rows = []
    for line in cards.read_text().splitlines():
        card = json.loads(line)
        if card["mode"] != "llm":
            continue
        sources = {source["id"]: source for source in card["sources"]}
        for claim in card["claims"]:
            rows.append(
                {
                    "species": card["species"],
                    "claim": claim["text"],
                    "source": sources.get(claim["sourceId"], {}).get("text", ""),
                    "supported": "",
                }
            )
    random.Random(SEED).shuffle(rows)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["species", "claim", "source", "supported"])
        writer.writeheader()
        writer.writerows(rows[:count])
    print(f"wrote {min(count, len(rows))} claims for review")


def score(sample: Path) -> dict:
    with sample.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    answers = [row["supported"].strip().lower() for row in rows]
    if not all(answer in {"yes", "no"} for answer in answers):
        raise SystemExit("Every row needs yes or no in the supported column.")
    unsupported = answers.count("no")
    return {
        "claims": len(answers),
        "unsupported": unsupported,
        "unsupported_rate": unsupported / len(answers) if answers else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export-sample")
    export.add_argument("cards", type=Path)
    export.add_argument("output", type=Path)
    export.add_argument("--count", type=int, default=100)
    scoring = commands.add_parser("score")
    scoring.add_argument("sample", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "export-sample":
        export_sample(arguments.cards, arguments.output, arguments.count)
    else:
        result = score(arguments.sample)
        (ROOT / "docs" / "metrics" / "grounding.json").write_text(json.dumps(result, indent=1))
        print(result)


if __name__ == "__main__":
    main()
