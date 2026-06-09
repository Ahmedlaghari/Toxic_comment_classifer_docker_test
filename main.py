from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model     = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    tfidf = vectorizer.transform([input.text])
    pred  = model.predict(tfidf)[0]
    prob  = model.predict_proba(tfidf)[0][1]
    return {
        "text": input.text,
        "toxic": bool(pred),
        "confidence": round(float(prob), 2)
    }

@app.get("/")
def root():
    return {"status": "running"}