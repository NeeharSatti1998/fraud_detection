from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

class InvoiceFeatures(BaseModel):
    amount: float 
    tax: float
    total: float
    invoice_year: int
    invoice_month: int
    invoice_day: int
    invoice_weekday: int

app = FastAPI(title="Invoice Fraud Detection API")

MODEL_PATH = os.path.join("model", "production_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Promoted model not found at: {MODEL_PATH}. Run evaluate_model.py first.")

model = joblib.load(MODEL_PATH)


@app.get("/")
def root():
    return {"message": "Invoice fraud detection API is live."}

@app.post("/predict_invoice_fraud", response_model=dict)
def predict_fraud(data: InvoiceFeatures):
    features = np.array([[data.amount, data.tax, data.total,
                          data.invoice_year, data.invoice_month,
                          data.invoice_day, data.invoice_weekday]])
    
    prediction = model.predict(features)[0]
    probability = float(model.predict_proba(features)[0][1])
    
    return {
        "prediction": int(prediction),
        "fraud_probability": round(probability, 4)
    }