import argparse
import os
import boto3
import sagemaker
from dotenv import load_dotenv
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from datetime import datetime

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--script", type=str, required=True, help="Evaluation script (e.g., evaluate.py)")
parser.add_argument("--model_s3_uri", type=str, required=True, help="S3 path to model.tar.gz")
args = parser.parse_args()

# AWS config
region = "ap-south-1"
bucket_name = "learnsagemakeramk"

# Sessions
boto_session = boto3.Session(region_name=region)
sagemaker_session = sagemaker.Session(boto_session=boto_session, default_bucket=bucket_name)

# Load role
load_dotenv()
role = os.getenv("SAGEMAKER_ROLE_ARN")
if role is None:
    raise ValueError("SAGEMAKER_ROLE_ARN not found in .env file.")

# Script processor
script_processor = SKLearnProcessor(
    framework_version="1.0-1",
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    sagemaker_session=sagemaker_session
)

# Paths
test_data_s3_path = f"s3://{bucket_name}/churn/processed/test/test.csv"
output_s3_path = f"s3://{bucket_name}/churn/evaluation/{datetime.now().strftime('%Y%m%d%H%M%S')}"

# Run job
script_processor.run(
    code=args.script,
    arguments=[
        "--model-path", "/opt/ml/processing/model/model.tar.gz",
        "--test-data", "/opt/ml/processing/test/test.csv",
        "--model-s3-uri", args.model_s3_uri 
    ],
    inputs=[
        ProcessingInput(source=args.model_s3_uri, destination="/opt/ml/processing/model"),
        ProcessingInput(source=test_data_s3_path, destination="/opt/ml/processing/test")
    ],
    outputs=[
        ProcessingOutput(source="/opt/ml/processing/output", destination=output_s3_path)
    ]
)
