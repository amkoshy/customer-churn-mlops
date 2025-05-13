import argparse
import os
import pandas as pd
import joblib
import tarfile
from sklearn.metrics import classification_report
import json
import re

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, required=True, help="Path to model.tar.gz")
parser.add_argument("--test-data", type=str, required=True, help="Path to test.csv")
parser.add_argument("--model-s3-uri", type=str, required=True, help="Original S3 path to model.tar.gz")

args = parser.parse_args()
print("evaluate.py script's ARGS:", vars(args))
print("Loading test data...")
test_df = pd.read_csv(args.test_data)

print("Extracting model tarball...")
extracted_dir = "/opt/ml/processing/extracted/"
os.makedirs(extracted_dir, exist_ok=True)

with tarfile.open(args.model_path, "r:gz") as tar:
    tar.extractall(path=extracted_dir)
    print("Extracted files:", tar.getnames())

# Identify the model file
model_files = [f for f in os.listdir(extracted_dir) if f.endswith(".joblib")]
if not model_files:
    raise FileNotFoundError("No .joblib model file found in tarball.")
model_filename = model_files[0]
model_path = os.path.join(extracted_dir, model_filename)

print(f"Loading model: {model_filename}...")
model = joblib.load(model_path)

# Prepare data
print("Preparing test data...")
if "Churn" not in test_df.columns:
    raise ValueError("Target column 'Churn' missing from test data.")
y_true = test_df["Churn"]
X_test = test_df.drop("Churn", axis=1)

# Predict
print("Running prediction...")
y_pred = model.predict(X_test)

# Classification report
print("Generating classification report...")
report_dict = classification_report(y_true, y_pred, output_dict=True)
report_text = classification_report(y_true, y_pred)

# Extract algorithm name
algo_match = re.search(r'train-([a-zA-Z0-9]+)-', args.model_s3_uri)

algorithm_name = algo_match.group(1) if algo_match else "unknown"
report_dict["algorithm"] = algorithm_name
report_text = f"Algorithm: {algorithm_name}\n\n" + report_text

# Save outputs
output_dir = "/opt/ml/processing/output"
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "classification_report_" + algorithm_name +"..json"), "w") as f:
    json.dump(report_dict, f, indent=4)

with open(os.path.join(output_dir, "classification_report_" + algorithm_name +".txt"), "w") as f:
    f.write(report_text)
print(output_dir)

print("Evaluation completed.")
