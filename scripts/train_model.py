import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,roc_auc_score
from xgboost import XGBClassifier
import joblib
import os

df = pd.read_csv("data/cleaned_invoice_data.csv")
df['date'] = pd.to_datetime(df['date'],errors='coerce')
df['invoice_year'] = df['date'].dt.year
df['invoice_month'] = df['date'].dt.month
df['invoice_day'] = df['date'].dt.day
df['invoice_weekday'] = df['date'].dt.weekday


df_model = df.drop(columns=['invoice_id', 'date', 'vendor', 'item', 'filename'],axis = 1).dropna()
X= df_model.drop(['is_fraud'],axis = 1)
y = df_model['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X ,y ,test_size=0.2 ,stratify=y ,random_state=42)

params = {
    "n_estimators": 150,
    "max_depth": 3,
    "learning_rate": 0.2,
    "subsample": 0.6,
    "colsample_bytree": 1.0,
    "gamma": 0,
    "use_label_encoder": False,
    "eval_metric": "logloss"
}

mlflow.set_experiment("Invoice_fraud_detection")


with mlflow.start_run():
    
    mlflow.log_params(params)

    model = XGBClassifier(**params)
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test,y_prob)

    mlflow.log_metric('accuracy', report['accuracy'])
    mlflow.log_metric('f1_fraud', report ['1']['f1-score'])
    mlflow.log_metric("recall_fraud", report["1"]["recall"])
    mlflow.log_metric("roc_auc", auc)


    os.makedirs("model", exist_ok=True)
    model_path = "model/tuned_xgboost_invoice_fraud_model_mlops.pkl"
    joblib.dump(model, model_path)

    mlflow.log_artifact(model_path)

    print("Model training complete and logged to MLflow.")





