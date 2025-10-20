# train_model.py
"""
SMS Spam Detection - Model Training Script
This script creates:
1. TF-IDF vectorizer
2. Calibrated Linear SVM model
3. IsolationForest for anomaly detection

All models are saved in 'models/' folder.
"""

import os
import random
import re
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# -----------------------------
# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

TFIDF_PATH = os.path.join(MODEL_DIR, "sms_tfidf.joblib")
SVM_PATH   = os.path.join(MODEL_DIR, "sms_svm_calibrated.joblib")
ISO_PATH   = os.path.join(MODEL_DIR, "sms_iso.joblib")
STOP_WORDS = set(stopwords.words('english'))

# -----------------------------
# Text preprocessing
URL_RE = re.compile(r'(hxxp|http|vvv|www)[\S]*', re.IGNORECASE)
def normalize_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = s.replace('hxxp','http').replace('vvv','www')
    s = re.sub(r'[^a-z0-9\s:/\.]', ' ', s)
    s = URL_RE.sub(' URL_TOKEN ', s)
    tokens = [w for w in s.split() if w not in STOP_WORDS]
    return " ".join(tokens)

# -----------------------------
# Generate small demo dataset (replace with real CSV if available)
def generate_sms_dataset(n=4000, spam_ratio=0.12):
    benign = [
        "Hey, are we meeting today?",
        "Your order has been shipped.",
        "See you tomorrow.",
        "Call me when free."
    ]
    spam = [
        "Congratulations! You've won a prize. Click http://free.redeem",
        "URGENT: verify your account at http://secure-bank.example",
        "Buy meds cheap at http://cheappharma.example"
    ]
    rows = []
    for i in range(n):
        if random.random() < spam_ratio:
            msg = random.choice(spam)
            label = 1
        else:
            msg = random.choice(benign)
            label = 0
        rows.append({'message': msg, 'label': label, 'ts': datetime.utcnow().isoformat()})
    return pd.DataFrame(rows)

# -----------------------------
def train_and_save(dataset_csv=None):
    # Load or generate dataset
    if dataset_csv and os.path.exists(dataset_csv):
        df = pd.read_csv(dataset_csv)  # Expect columns: message,label
    else:
        df = generate_sms_dataset(n=4000, spam_ratio=0.12)

    df['clean'] = df['message'].apply(normalize_text)

    # TF-IDF vectorizer
    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1,2))
    X_text = tfidf.fit_transform(df['clean'])
    y = df['label'].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    # IsolationForest for anomaly detection (trained only on ham)
    iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
    iso.fit(X_train[y_train==0])

    # Linear SVM + calibration to get probability
    svc = LinearSVC(max_iter=5000, dual=False)
    clf = CalibratedClassifierCV(svc, cv=3, method='sigmoid')
    clf.fit(X_train, y_train)

    # Save all artifacts
    joblib.dump(tfidf, TFIDF_PATH)
    joblib.dump(clf, SVM_PATH)
    joblib.dump(iso, ISO_PATH)
    print(f"✅ Models saved to {MODEL_DIR}:\n- {TFIDF_PATH}\n- {SVM_PATH}\n- {ISO_PATH}")

# -----------------------------
if __name__ == '__main__':
    train_and_save()
