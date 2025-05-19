import requests
import random
import time
from datetime import datetime

URL = "http://fraud_api:8000/predict_invoice_fraud"

def generate_synthetic_invoice():
    amount = round(random.uniform(100, 10000), 2)
    tax = round(amount * 0.1, 2)
    total = round(amount + tax, 2)
    year = 2025
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    weekday = random.randint(0, 6)

    return {
        "amount": amount,
        "tax": tax,
        "total": total,
        "invoice_year": year,
        "invoice_month": month,
        "invoice_day": day,
        "invoice_weekday": weekday
    }

def send_request(invoice):
    try:
        response = requests.post(URL, json=invoice)
        print(f"[{datetime.now()}] Sent: {invoice}")
        print(f"[{datetime.now()}] Received: {response.json()}\n")
    except Exception as e:
        print(f"[{datetime.now()}] Error: {e}\n")

if __name__ == "__main__":
    print("Sending synthetic invoice data every 30 seconds...\n")
    while True:
        invoice = generate_synthetic_invoice()
        send_request(invoice)
        time.sleep(10)
