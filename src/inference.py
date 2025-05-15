import os
import json
import joblib
import numpy as np
import pandas as pd

TRAIN_COLUMNS = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'tenure_norm',
    'gender_Female', 'gender_Male', 'Partner_No', 'Partner_Yes', 'Dependents_No',
    'Dependents_Yes', 'PhoneService_No', 'PhoneService_Yes', 'MultipleLines_No',
    'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_DSL',
    'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes', 'DeviceProtection_No',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No',
    'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No',
    'StreamingTV_No internet service', 'StreamingTV_Yes', 'StreamingMovies_No',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'Contract_Month-to-month',
    'Contract_One year', 'Contract_Two year', 'PaperlessBilling_No',
    'PaperlessBilling_Yes', 'PaymentMethod_Bank transfer (automatic)',
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check'
]

def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.joblib")
    return joblib.load(model_path)

def preprocess(df):
    # Drop ID column if present
    df = df.drop(columns=["customerID"], errors="ignore")

    # Convert numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    # Normalize tenure
    df["tenure_norm"] = (df["tenure"] / 72.0).astype("float32")

    # One-hot encode categorical features
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols)

    # Add missing columns
    for col in TRAIN_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    # Ensure column order
    return df[TRAIN_COLUMNS]

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        parsed = json.loads(request_body)
        data = parsed["data"] if "data" in parsed else [parsed]
        df = pd.DataFrame(data)
        print("🧪 Columns received at inference:", df.columns.tolist())
        return preprocess(df)
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, accept="application/json"):
    if accept == "application/json":
        return json.dumps(prediction.tolist()), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
