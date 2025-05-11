# preprocessing.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split


input_path = "/opt/ml/processing/input/WA_Fn-UseC_-Telco-Customer-Churn.csv"



print("Loading data...")
df = pd.read_csv(input_path)

# Example preprocessing
df = df.dropna()                                 # Drop missing rows
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})  # Encode categorical
df['tenure_norm'] = df['tenure'] / df['tenure'].max()      # Normalize

# Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Output directories
train_output_path = "/opt/ml/processing/train/train.csv"
test_output_path = "/opt/ml/processing/test/test.csv"

# Save result
print("Saving processed data...")
train_df.to_csv(train_output_path, index=False)
test_df.to_csv(test_output_path, index=False)
