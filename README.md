# Invoice Fraud Detection – MLOps Pipeline with Monitoring and Visualization

This project is a full-stack MLOps pipeline for detecting invoice fraud using machine learning. It includes a FastAPI backend for predictions, a synthetic data sender to simulate production traffic, Prometheus for metrics scraping, Grafana for monitoring dashboards, and Streamlit for displaying results interactively. All components run within Docker containers for modularity and ease of deployment.

---

## Project Architecture

**Components:**

- **FastAPI:** Serves fraud predictions from a trained ML model.
- **Synthetic Data Sender:** Periodically sends invoice data to FastAPI.
- **Prometheus:** Scrapes metrics from FastAPI for monitoring.
- **Grafana:** Visualizes system metrics, predictions, and latencies.
- **Streamlit:** Web dashboard to view model predictions.
- **Docker Compose:** Orchestrates all services.

---

## Machine Learning Model

- **Model:** XGBoostClassifier
- **Input features:**
  - `amount`, `tax`, `total`
  - `invoice_year`, `invoice_month`, `invoice_day`, `invoice_weekday`
- **Output:** Binary prediction (fraud or not fraud)
- **Training:** Performed on synthetic invoice data and saved as `model/production_model.pkl`

---

## Folder Structure

```
fraud_detection/
├── app/
│   └── main.py                   # FastAPI app
├── streamlit_app.py              # Streamlit frontend
├── send_synthetic_data.py        # Data sender script
├── Dockerfile                    # For FastAPI + model
├── Dockerfile.sender             # For synthetic sender
├── docker-compose.yml            # Docker Compose config
├── monitoring/
│   └── prometheus.yml            # Prometheus scrape config
├── model/
│   └── production_model.pkl      # Trained ML model
├── requirements.txt              # Python dependencies
```

---

## How It Works

### 1. FastAPI Prediction Endpoint

- **URL:** `http://localhost:8000/predict_invoice_fraud`
- **Accepts:** POST request with invoice data
- **Returns:**

```json
{
  "prediction": 1,
  "fraud_probability": 0.92
}
```

---

### 2. Synthetic Data Sender

- Sends random invoice data every 30 seconds using:

```python
requests.post(URL, json=invoice)
```

- Keeps the API active and continuously feeds data to Prometheus.

---

### 3. Prometheus Monitoring

- Scrapes `/metrics` from FastAPI.
- Custom metrics exposed:
  - `invoice_fraud_prediction_total`
  - `invoice_prediction_latency_seconds`

---

### 4. Grafana Dashboards

- Visualizations include:
  - Prediction counts over time
  - Prediction latency heatmap
  - Fraud predictions automatically trigger email notifications
---

### 5. Streamlit Frontend

- Displays real-time prediction results fetched from FastAPI.

---

## Dockerized Setup

### Build and Run All Services

```bash
docker-compose up --build
```

### Containers Created

- `fraud_api` – FastAPI server
- `synthetic_data_sender` – Sends synthetic invoices
- `prometheus` – Scrapes `/metrics`
- `grafana` – Visual dashboard
- `streamlit_ui` – Displays predictions

---

## Environment Variables and Secrets

- API keys (e.g., SendGrid for alerts) should be stored securely.
- Use a `.env` file and make sure to add it to `.gitignore`.

Example:

```env
GF_SMTP_PASSWORD=your_key_here
```

---

## Monitoring Setup

- **Prometheus Config:** `monitoring/prometheus.yml`
- **Scrapes from:** FastAPI on port `8000`

Example query in Grafana:

```promql
rate(invoice_prediction_latency_seconds_bucket[1m])
```

---

## Grafana Visualization Highlights

- **Prediction Heatmap:** Shows latency visually in real time.
- **Request Count Panel:** Number of predictions over time.
- **Latency Summary Panel:** Response time of the model.
- **Customizations:**
  - Color gradient (`YlOrRd`)
  - Panel titles, units, and legends properly configured

---

## Future Improvements

- Add alerting rules to Grafana for fraud spikes
- Store predictions in a database like MySQL or PostgreSQL
- Replace synthetic data with real-world invoice datasets
- Deploy it on cloud
