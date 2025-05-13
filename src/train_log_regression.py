# train_logistic.py

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import os

# Load preprocessed training and validation data
train_path = "/opt/ml/input/data/train/train.csv"
val_path = "/opt/ml/input/data/val/val.csv"

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)

# Split features and labels
X_train = train_df.drop("Churn", axis=1)
y_train = train_df["Churn"]
X_val = val_df.drop("Churn", axis=1)
y_val = val_df["Churn"]

# Train Logistic Regression model
algorithm_name = "logistic"
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate on validation set
y_pred = model.predict(X_val)
print("Classification Report:")
print(classification_report(y_val, y_pred))

# Save model
output_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
os.makedirs(output_dir, exist_ok=True)
model_filename = f"model_{algorithm_name}.joblib"
joblib.dump(model, os.path.join(output_dir, model_filename))
print(f"Model saved to {os.path.join(output_dir, model_filename)}")
print("Training and evaluation complete.")
