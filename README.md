# Spam Email Detection using Machine Learning

## Problem Statement
Email spam is a major cybersecurity and productivity issue. The goal of this project is to build a machine learning model that can automatically classify incoming emails as **spam** or **ham** (not spam), reducing manual filtering effort.

## Dataset Description
The project expects a CSV dataset in `dataset/sample_emails.csv` with the following columns:
- `email`: email text content
- `label`: target class (`spam` or `ham`)

A sample dataset file is included for quick testing.

## Technologies Used
- Python
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- nltk

## Model Used
This project supports two machine learning models:
- Naive Bayes (`naive_bayes`)
- Logistic Regression (`logistic_regression`)

The text is transformed using TF-IDF vectorization before classification.

## Project Structure
```text
spam-email-detection/
|- dataset/
|  |- sample_emails.csv
|- notebook/
|  |- .gitkeep
|- src/
|  |- spam_detector.py
|- model/
|  |- .gitkeep
|  |- spam_model.pkl
|- README.md
|- requirements.txt
|- spam_email_detection_project.py
```

## Steps to Run the Project
1. Create and activate a virtual environment.
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies.
```powershell
pip install -r requirements.txt
```

3. Train the model.
```powershell
py spam_email_detection_project.py train --data dataset/sample_emails.csv --model-out model/spam_model.pkl --model-type naive_bayes
```

4. Predict a new email.
```powershell
py spam_email_detection_project.py predict --model model/spam_model.pkl --text "Congratulations! You have won a free gift card. Click now!"
```

## Example Output
```text
Prediction: spam
```

## Future Improvements
- Add advanced NLP preprocessing (lemmatization, stopword tuning, n-gram optimization).
- Add hyperparameter tuning and cross-validation.
- Build a simple web app interface (Flask/FastAPI).
- Add model performance plots using matplotlib/seaborn.
- Expand dataset for better real-world generalization.
