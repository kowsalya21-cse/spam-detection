# python/ml_service.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import pandas as pd
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

STOP = set(stopwords.words('english'))
URL_RE = re.compile(r'(hxxp|http|vvv|www)[\S]*', re.IGNORECASE)

def normalize_text(s):
    if pd.isna(s): 
        return ""
    s = str(s).lower()
    s = s.replace('hxxp','http').replace('vvv','www')
    s = re.sub(r'[^a-z0-9\s:/\.]', ' ', s)
    s = URL_RE.sub(' URL_TOKEN ', s)
    tokens = [w for w in s.split() if w not in STOP]
    return " ".join(tokens)

# Load models
TFIDF_PATH = "models/sms_tfidf.joblib"
SVM_PATH = "models/sms_svm_calibrated.joblib"
ISO_PATH = "models/sms_iso.joblib"

tfidf = joblib.load(TFIDF_PATH)
clf = joblib.load(SVM_PATH)
iso = joblib.load(ISO_PATH)

app = FastAPI(title="SMS Spam ML Service")

class MessageIn(BaseModel):
    text: str

@app.post("/predict")
def predict(msg: MessageIn):
    text = msg.text
    clean = normalize_text(text)
    X_text = tfidf.transform([clean])

    # SVM prediction probability
    prob = float(clf.predict_proba(X_text)[:,1][0])

    # Isolation Forest anomaly detection
    iso_flag = int((iso.predict(X_text) == -1)[0])

    # Decision rule
    threshold = 0.45
    label = 1 if (prob >= threshold or iso_flag == 1) else 0

    return {
        "label": int(label),
        "prob": prob,
        "anomaly": bool(iso_flag),
        "clean": clean
    }

@app.get("/")
def root():
    return {"status": "ML service running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
