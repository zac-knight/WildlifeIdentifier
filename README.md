# WildLens

WildLens identifies Australian birds from photos and sound, combines that evidence with where and
when the observation happened, and says so when it is unsure.

This README is a skeleton and will grow as the project is built.

## Run it locally

```bash
# Web app
cd web && npm install && npm run dev

# Machine learning project
cd ml && uv sync --extra train --extra climate && uv run pytest
```

## Project layout

- `web/`: Next.js application
- `ml/`: Python data pipeline, training, evaluation and export
- `docs/`: decision log, dataset card, model cards and evaluation report
- `data/`, `models/`: local artefacts, not committed
