import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Load raw input
input_path = "/opt/ml/processing/input/churn/Fn-UseC_-Telco-Customer-Churn.csv"
print("Loading data...")
df = pd.read_csv(input_path)

# Convert TotalCharges to numeric, coerce invalids to NaN
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Drop missing rows
df = df.dropna()

# Encode target column
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df = df.dropna(subset=["Churn"])  # Drop if any target is still missing

# Drop ID column
if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)

# Normalize 'tenure' (float32 to save space)
if "tenure" in df.columns:
    df["tenure_norm"] = (df["tenure"] / df["tenure"].max()).astype("float32")

# One-hot encode only categorical columns (exclude numeric columns)
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
df = pd.get_dummies(df, columns=categorical_cols)

# 3-way split: 64% train, 16% val, 20% test
train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)  # 20% of 80% = 16%

# Save splits
print("Saving processed splits...")
train_output_path = "/opt/ml/processing/train/train.csv"
val_output_path = "/opt/ml/processing/val/val.csv"
test_output_path = "/opt/ml/processing/test/test.csv"

os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
os.makedirs(os.path.dirname(val_output_path), exist_ok=True)
os.makedirs(os.path.dirname(test_output_path), exist_ok=True)

train_df.to_csv(train_output_path, index=False)
val_df.to_csv(val_output_path, index=False)
test_df.to_csv(test_output_path, index=False)

print("Done.")
