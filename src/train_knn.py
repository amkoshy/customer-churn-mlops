# train_knn.py

import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# Load preprocessed training data
input_path = "/opt/ml/input/data/train/train.csv"
df = pd.read_csv(input_path)

# Drop ID column if exists
if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)

# Ensure 'Churn' column exists and is binary
if "Churn" in df.columns:
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Drop rows where target is missing
df = df.dropna(subset=["Churn"])

# Convert categorical columns to numeric (after target handling)
df = pd.get_dummies(df)

# Prepare features and labels
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN classifier
algorithm_name = "knn"  # Specify the algorithm name
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)
print("Classification Report:")
print(classification_report(y_val, y_pred))

# Save model
output_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
os.makedirs(output_dir, exist_ok=True)
model_filename = f"model_{algorithm_name}.joblib"  # Include algorithm name in the file name
joblib.dump(model, os.path.join(output_dir, model_filename))
