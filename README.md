# Spam Email Detection using Machine Learning

A production-style machine learning project that classifies emails as **spam** or **ham** with reusable training, prediction, and evaluation workflows.

## Problem Statement
Spam emails create security and productivity risks. This project builds a reliable classifier that can detect spam automatically and provide evaluation artifacts for model quality tracking.

## Project Architecture (Text Diagram)
```text
CLI (spam_email_detection_project.py)
        |
        v
src/train.py   src/predict.py   src/evaluate.py
        \        |             /
         \       |            /
              src/spam_pipeline.py
                    |
      +-------------+----------------+
      |                              |
 data/sample_emails.csv         models/spam_model.joblib
                                      |
                                      v
                        reports/evaluation.txt
                        reports/confusion_matrix.png
```

## Repository Structure
```text
spam-email-detection/
    data/
    models/
    src/
    reports/
    notebooks/
    screenshots/
    README.md
    requirements.txt
```

## Skills Demonstrated
- Data loading and validation
- Text preprocessing with TF-IDF pipeline
- Model training and optional model comparison
- Metrics reporting (precision, recall, F1-score)
- Confusion matrix visualization
- Error handling and structured logging
- CLI-based reproducible ML workflow

## Models
- Naive Bayes (`naive_bayes`)
- Logistic Regression (`logistic_regression`)

Use `--compare-models` to automatically train both and save the best one by F1-score.
Use `--cv-folds` to run optional cross-validation before final training.
Use `--tune-hyperparams` to run grid-search tuning on the selected model.

## Example CLI Usage
1. Train a single model:
```powershell
py spam_email_detection_project.py train --model-type logistic_regression
```

2. Compare models and save the best:
```powershell
py spam_email_detection_project.py train --compare-models
```

2.1. Compare models with cross-validation:
```powershell
py spam_email_detection_project.py train --compare-models --cv-folds 5
```

2.2. Train with hyperparameter tuning:
```powershell
py spam_email_detection_project.py train --model-type logistic_regression --cv-folds 3 --tune-hyperparams
```

3. Predict one email:
```powershell
py spam_email_detection_project.py predict --text "URGENT: Verify your account now"
```

4. Evaluate and generate reports:
```powershell
py spam_email_detection_project.py evaluate
```

## Generated Evaluation Outputs
- `reports/evaluation.txt`
- `reports/confusion_matrix.png`
- `reports/roc_curve.png`
- `reports/pr_curve.png`
- `reports/model_metadata.json`
- `screenshots/sample_output_real.png`

## Model Evaluation Metrics
Training and evaluation now report the following metrics:
- Accuracy
- Precision
- Recall
- F1-score

These metrics are logged in terminal output and written to `reports/evaluation.txt` after evaluation.

## Future Improvements
- Hyperparameter optimization with grid/random search
- Larger and more diverse dataset
- Experiment tracking (MLflow)
- REST API deployment with FastAPI
- Continuous model monitoring for drift
