# Sentiment Analysis with Multiple Models (SVM, Naive Bayes, Random Forest)

This project demonstrates how to perform **sentiment analysis** using different machine learning models:
- **Support Vector Machine (SVM)**
- **Naive Bayes (MultinomialNB)**
- **Random Forest Classifier**

The dataset is provided as a CSV file containing:
- `content` → text data to analyze
- `score` → numerical ratings (mapped into sentiment classes)

---

## Project Overview

The workflow includes:
1. **Data Preprocessing**
   - Clean text (lowercasing, removing non-alphabetic characters, trimming whitespace).
   - Map numeric scores into sentiment labels:
     - `score >= 4` → **POSITIVE**
     - `score <= 2` → **NEGATIVE**
     - `score = 3` → **NEUTRAL**

2. **Feature Extraction**
   - Using **TF-IDF Vectorization** with up to 5000 features.

3. **Model Training and Evaluation**
   - Models are trained using an 70/30 **train-test split** with stratification.
   - Evaluation metrics:
     - Accuracy
     - Precision
     - Recall
     - F1-score
     - Confusion Matrix

---

## Requirements

Make sure you have the following installed:

```bash
pip install pandas numpy scikit-learn

Usage

1. Load and preprocess the dataset
import pandas as pd
import re

# Load dataset
df = pd.read_csv("your_dataset.csv")

# Preprocess text
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["content"] = df["content"].apply(preprocess_text)

# Map score to sentiment
def map_score_to_sentiment(score):
    if score >= 4:
        return "POSITIVE"
    elif score <= 2:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

df["label"] = df["score"].apply(map_score_to_sentiment)

2. Train-test split
from sklearn.model_selection import train_test_split

X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

3. Define models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

models = {
    "SVM": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
        ("clf", SVC(kernel="linear", probability=True, random_state=42))
    ]),
    "Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
        ("clf", MultinomialNB())
    ]),
    "Random Forest": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
}

4. Train and evaluate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

for name, model in models.items():
    print(f"\n=== {name} ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
Notes

* You can extend this by adding Logistic Regression or Deep Learning models (Transformers).

* This implementation is built in Jupyter Notebook for experimentation and analysis.

*For production-ready API, you can integrate with FastAPI (already supported in your codebase).