# Spam Email Detection

An end-to-end, GitHub-ready AI project for detecting spam emails with a practical, explainable workflow.

## What makes this unique

- Hybrid features: combines classic text TF-IDF features with handcrafted email risk signals.
- Explainability-first: shows the top words/signals influencing each prediction.
- Real-world style split: includes confidence score and easy model comparison.
- Portfolio-ready structure: clean scripts, reproducible steps, and sample dataset.

## Project Structure

```
spam-email-detection/
+- data/
¦  +- sample_emails.csv
+- models/
¦  +- .gitkeep
+- src/
¦  +- train.py
¦  +- predict.py
¦  +- evaluate.py
+- requirements.txt
+- .gitignore
+- README.md
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```powershell
python src/train.py --data data/sample_emails.csv --model-out models/spam_model.joblib
```

This will:
- Train Logistic Regression and Linear SVM pipelines
- Print metrics on validation split
- Save the best model to `models/spam_model.joblib`

## Predict

```powershell
python src/predict.py --model models/spam_model.joblib --text "Congratulations! You won a free iPhone. Click now!"
```

## Evaluate with Explainability

```powershell
python src/evaluate.py --model models/spam_model.joblib --text "Urgent: Your account is locked, verify immediately"
```

## GitHub Upload

```powershell
git init
git add .
git commit -m "Initial commit: unique spam email detection project"
# then connect remote and push
```

## Future Upgrades

- Add FastAPI + web UI
- Use larger public dataset (e.g., SMS Spam Collection)
- Add CI tests and model drift monitoring