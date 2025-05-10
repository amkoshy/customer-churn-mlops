# preprocessing.py
import pandas as pd
import os

input_path = "/opt/ml/processing/input/WA_Fn-UseC_-Telco-Customer-Churn.csv"

output_path = "/opt/ml/processing/output/processed.csv"

print("Loading data...")
df = pd.read_csv(input_path)

# Example preprocessing
df = df.dropna()                                 # Drop missing rows
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})  # Encode categorical
df['tenure_norm'] = df['tenure'] / df['tenure'].max()      # Normalize

# Save result
print("Saving processed data...")
df.to_csv(output_path, index=False)
