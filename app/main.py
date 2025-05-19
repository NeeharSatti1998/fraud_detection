from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
import time
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

# Request model
class InvoiceFeatures(BaseModel):
    amount: float 
    tax: float
    total: float
    invoice_year: int
    invoice_month: int
    invoice_day: int
    invoice_weekday: int

# Response model
class PredictionResponse(BaseModel):
    prediction: int
    fraud_probability: float

# FastAPI app setup
app = FastAPI(title="Invoice Fraud Detection API")

# Load the model
MODEL_PATH = os.path.join("model", "production_model.pkl")
model = joblib.load(MODEL_PATH)

# Prometheus metrics
fraud_prediction_counter = Counter(
    name="invoice_fraud_prediction_total",
    documentation="Total number of fraud predictions by class",
    labelnames=["prediction"]
)

prediction_latency = Histogram(
    name="invoice_prediction_latency_seconds",
    documentation="Time taken to generate invoice fraud prediction"
)

# Instrumentator setup
instrumentator = Instrumentator()
instrumentator.instrument(app)

@app.on_event("startup")
async def on_startup():
    instrumentator.expose(app)

@app.get("/")
def root():
    return {"message": "Invoice fraud detection API is live."}

@app.post("/predict_invoice_fraud", response_model=dict)
def predict_fraud(data: InvoiceFeatures):
    try:
        features = np.array([[data.amount, data.tax, data.total,
                              data.invoice_year, data.invoice_month,
                              data.invoice_day, data.invoice_weekday]])
        
        # Measure latency
        start_time = time.time()
        prediction = int(model.predict(features)[0])
        probability = round(float(model.predict_proba(features)[0][1]), 4)
        elapsed = time.time() - start_time

        # Record metrics
        prediction_latency.observe(elapsed)
        fraud_prediction_counter.labels(prediction=str(prediction)).inc()

        return {"prediction": prediction, "fraud_probability": probability}
    
    except Exception as e:
        import traceback
        print("ERROR during prediction:", e)
        traceback.print_exc()
        return {"error": str(e)}
