# Use official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all necessary files
COPY ./app ./app
COPY ./model ./model
COPY ./streamlit_app.py .
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose both ports: FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Run both apps in parallel
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
