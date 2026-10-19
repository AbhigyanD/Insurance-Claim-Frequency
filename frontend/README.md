# Claims Pricing Dashboard (React)

A static React dashboard for the claim frequency, severity and expected-claims-cost models in this repository.
It reads small JSON files exported from the real model outputs; there is no backend and no credentials.

**Tabs:** Executive overview, Risk analysis, Model performance, Risk drivers, Monitoring.

## What it shows (and does not)

- Everything comes from the **validation split** used for model selection. The held-out test split is not exported or shown.
- Only aggregates are exported (no policy-level rows).
- Historical outcomes (actual claims) and model predictions are labelled separately.
- Groups with under 1,000 exposure-years are flagged as small samples.
- Risk drivers are associations, not causal effects. Expected cost is a pure-premium estimate, not a customer price.
- Monitoring shows PSI between the training and validation splits only. The dataset has no date column, so performance-over-time and
  data-quality-alert views are intentionally absent rather than simulated.

## Run it

```bash
# from the repo root, after running the Python pipeline (see the main README)
python -m scripts.export_frontend_data      # writes frontend/public/data/*.json

cd frontend
npm install
npm run dev        # development server
npm test           # unit and app tests (vitest)
npm run build      # production build in dist/
```

The committed JSON files in `public/data/` are the exported results of the pipeline run documented in the main README, so the app
works straight after `npm install`.
