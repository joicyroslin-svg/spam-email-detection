# Spam Email Detection

An end-to-end, GitHub-ready AI project for detecting spam emails with a practical, explainable workflow.

## About

This project uses a hybrid AI + rule-based approach to classify emails as spam or ham.
It combines TF-IDF text features, handcrafted spam indicators, and phishing-risk signal analysis to provide:

- Spam/ham prediction
- Confidence score
- Explainable top spam terms
- Risk-score based safety action

## What makes this unique

- Hybrid features: combines classic text TF-IDF features with handcrafted email risk signals.
- Explainability-first: shows the top words/signals influencing each prediction.
- Real-world style split: includes confidence score and easy model comparison.
- Portfolio-ready structure: clean scripts, reproducible steps, and sample dataset.

## Project Structure

```text
spam-email-detection/
+- data/
|  +- sample_emails.csv
+- models/
|  +- .gitkeep
|  +- spam_model.joblib
+- src/
|  +- train.py
|  +- predict.py
|  +- evaluate.py
+- requirements.txt
+- .gitignore
+- README.md
+- spam_email_detection_project.py
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
py -m pip install -r requirements.txt
```

## Train

```powershell
py spam_email_detection_project.py train --data data/sample_emails.csv --model-out models/spam_model.joblib
```

## Predict

```powershell
py spam_email_detection_project.py predict --model models/spam_model.joblib --text "Congratulations! You won a free iPhone. Click now!"
```

## Evaluate with Explainability

```powershell
py spam_email_detection_project.py explain --model models/spam_model.joblib --text "Urgent: Your account is locked, verify immediately"
```

## Analyze with Risk Signals

```powershell
py spam_email_detection_project.py analyze --model models/spam_model.joblib --text "URGENT: Verify OTP now to avoid account suspension."
```

## Future Upgrades

- Add FastAPI + web UI
- Use larger public dataset (e.g., SMS Spam Collection)
- Add CI tests and model drift monitoring
