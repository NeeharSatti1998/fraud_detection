import streamlit as st
import requests

st.set_page_config(page_title="Invoice Fraud Detector")

st.markdown("<h1 style='color:#4A90E2;'> Invoice Fraud Detection App</h1>", unsafe_allow_html=True)
st.markdown("Use this tool to predict whether an invoice is potentially fraudulent based on financial metadata.")


col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Amount", min_value=0.0, value=1000.0)
    tax = st.number_input("Tax", value=180.0)
    total = st.number_input("Total", value=1180.0)


with col2:
    invoice_year = st.number_input("Year", min_value=1900, max_value=2100, value=2024)
    invoice_month = st.slider("Month", 1, 12, 5)
    invoice_day = st.slider("Day", 1, 31, 15)
    invoice_weekday = st.selectbox("Weekday (0=Mon, 6=Sun)", list(range(7)))

if st.button("Predict Fraud"):
    payload = {
        "amount": amount,
        "tax": tax,
        "total": total,
        "invoice_year": invoice_year,
        "invoice_month": invoice_month,
        "invoice_day": invoice_day,
        "invoice_weekday": invoice_weekday
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict_invoice_fraud", json=payload)
        result = response.json()

        st.markdown("---")
        st.subheader("Prediction Result")
        if result["prediction"] == 1:
            st.error(f"**Fraud Detected!**\n\nFraud Probability: **{result['fraud_probability']*100:.2f}%**")
        else:
            st.success(f"**Invoice is Clean**\n\nFraud Probability: **{result['fraud_probability']*100:.2f}%**")
        
        st.progress(min(result["fraud_probability"], 1.0))

    except Exception as e:
        st.error(f"API request failed: {e}")


