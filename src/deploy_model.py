import argparse
import os
import json
import tarfile
import tempfile
import time
import boto3
from urllib.parse import urlparse
from sagemaker import Model, Session
from sagemaker.image_uris import retrieve

# ----------------- Parse Arguments -----------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--algorithm",
    type=str,
    required=True,
    choices=["knn", "log_regression", "random_forest"],
    help="Specify the algorithm (e.g., 'knn', 'log_regression', 'random_forest')."
)
parser.add_argument(
    "--endpoint-name",
    type=str,
    required=False,
    help="Optional custom endpoint name (default: auto-generated)."
)
args = parser.parse_args()

# ----------------- Load S3 URI from models.json -----------------
models_file_path = os.path.join(os.path.dirname(__file__), "../models.json")
with open(models_file_path, "r") as f:
    models = json.load(f)

if args.algorithm not in models:
    raise ValueError(f"Algorithm '{args.algorithm}' not found in models.json. Available: {list(models.keys())}")

model_s3_uri = models[args.algorithm]
print(f"📦 Deploying model from S3: {model_s3_uri}")

# ----------------- Inspect Tarball -----------------
def inspect_model_tarball(s3_uri):
    s3 = boto3.client("s3")
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    print("🔍 Checking model.tar.gz contents...")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as f:
        s3.download_fileobj(bucket, key, f)
        f.flush()
        with tarfile.open(f.name) as tar:
            contents = tar.getnames()
            print("📂 Contents of model.tar.gz:", contents)
            if not any(name.endswith(".joblib") for name in contents):
                raise ValueError("❌ No .joblib file found in model.tar.gz — deployment will fail.")
            if any("/" in name for name in contents):
                print("⚠️ Warning: model file may be nested inside a folder. It should be at root level.")

inspect_model_tarball(model_s3_uri)

# ----------------- Generate Endpoint Name -----------------
timestamp = time.strftime("%Y%m%d-%H%M%S")
endpoint_name = args.endpoint_name or f"churn-{args.algorithm}-endpoint-{timestamp}"

# ----------------- Deploy Model -----------------
role = os.environ["SAGEMAKER_ROLE_ARN"]
sagemaker_session = Session()
region = sagemaker_session.boto_region_name

image_uri = retrieve("sklearn", region=region, version="1.2-1")  # Match your training version

# 🔧 Updated part: Added entry_point for inference script
model = Model(
    image_uri=image_uri,
    model_data=model_s3_uri,
    role=role,
    sagemaker_session=sagemaker_session,
    entry_point="inference.py"  # Required for SKLearn inference
)

print(f"🚀 Deploying to endpoint: {endpoint_name}")
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name=endpoint_name
)

print("✅ Model deployed successfully.")
print(f"🔗 Endpoint: {endpoint_name}")
