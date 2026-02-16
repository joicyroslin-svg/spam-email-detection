# Spam Email Detection using Machine Learning

A production-style machine learning project that classifies emails as **spam** or **ham** with reusable training, prediction, evaluation, benchmarking, and API workflows.

## Problem Statement
Spam emails create security and productivity risks. This project builds a reliable classifier to detect spam, expose explainability signals, and provide reproducible reporting artifacts.

## Project Architecture (Text Diagram)
```text
CLI (spam_email_detection_project.py)      FastAPI (api/main.py)
                |                                   |
                +-------------------+---------------+
                                    v
                           src/spam_pipeline.py
                                    |
      +-----------------------------+------------------------------+
      |                             |                              |
 data/sample_emails.csv      data/benchmark_emails.csv     models/spam_model.joblib
                                    |
                                    v
 reports/evaluation.txt, benchmark_results.csv, tuning_results.csv,
 reports/confusion_matrix.png, roc_curve.png, pr_curve.png,
 reports/explainability_report.txt, reports/model_metadata.json
```

## Repository Structure
```text
spam-email-detection/
    api/
    data/
    models/
    src/
    reports/
    notebooks/
    screenshots/
    Dockerfile
    docker-compose.yml
    pyproject.toml
    README.md
    requirements.txt
```

## Key Improvements
- Larger benchmark dataset: `data/benchmark_emails.csv`
- Hyperparameter tuning: `RandomizedSearchCV` with saved history in `reports/tuning_results.csv`
- Probability calibration: `--calibrate` (CalibratedClassifierCV)
- Threshold tuning + curves: ROC/PR curves + best threshold in `reports/evaluation.txt`
- Explainability: global and sample term report in `reports/explainability_report.txt`
- Versioning metadata: `reports/model_metadata.json` (dataset hash, model hash, git commit)
- API serving: FastAPI `/predict`
- Quality gates: `ruff` + `mypy` in CI

## Models
- Naive Bayes (`naive_bayes`)
- Logistic Regression (`logistic_regression`)

## Example CLI Usage
1. Train with comparison, CV, tuning, and calibration:
```powershell
py spam_email_detection_project.py train --compare-models --cv-folds 3 --tune-hyperparams --n-iter 10 --calibrate
```

2. Predict one email:
```powershell
py spam_email_detection_project.py predict --text "URGENT: Verify your account now"
```

3. Evaluate and generate reports:
```powershell
py spam_email_detection_project.py evaluate
```

## API Usage
Run API locally:
```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Test endpoints:
```powershell
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text":"Win cash now! Click link"}'
```

## Docker Usage
```powershell
docker compose up --build
```

## Generated Reports
- `reports/evaluation.txt`
- `reports/confusion_matrix.png`
- `reports/roc_curve.png`
- `reports/pr_curve.png`
- `reports/tuning_results.csv`
- `reports/benchmark_results.csv`
- `reports/explainability_report.txt`
- `reports/model_metadata.json`

## Quality Checks
```powershell
ruff check .
mypy spam_email_detection_project.py src api
pytest
```

## Pre-commit Setup
```powershell
pre-commit install
pre-commit run --all-files
```

## Semantic Release
This repository uses **Release Please** to automate semantic versioning and changelog updates on `main`.
- Workflow: `.github/workflows/release.yml`
- Config: `release-please-config.json`
- Manifest: `.release-please-manifest.json`

Use Conventional Commit messages (for example: `feat: ...`, `fix: ...`) for clean release notes.
