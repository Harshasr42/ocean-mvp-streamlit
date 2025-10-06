## Local dev & tests

1. Create a fresh venv (do not commit it):
   python -m venv .venv
   source .venv/bin/activate  # or `.\.venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   # optional dev tools
   pip install -r requirements-dev.txt

2. Run tests:
   pytest -q

3. Train a pipeline from CSV:
   python models/train_model.py --data-csv data/train.csv --target species_count --out-dir models/saved_models --version v1.0.0

4. Run API locally:
   export SPECIES_PIPELINE_PATH=models/saved_models/species_abundance_pipeline.pkl
   export MODEL_METADATA_PATH=models/saved_models/metadata.json
   python -m flask --app app.predict_api run --port 5000

## Docker (optional)
Build:
  docker build -t myorg/ocean-mvp:latest .

Run (example):
  docker run -p 8000:8000 -e SPECIES_PIPELINE_PATH=/app/models/saved_models/species_abundance_pipeline.pkl myorg/ocean-mvp:latest

## CI / GitHub
- The repository contains a GitHub Actions workflow `.github/workflows/ci.yml` that runs tests on push and PRs to `main`.
- Add any credentials or tokens to GitHub Secrets (Settings → Secrets → Actions) and never commit them into the repo.