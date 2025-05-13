# run_training.py

import sagemaker
from sagemaker.sklearn.estimator import SKLearn
import boto3
import os
from dotenv import load_dotenv
import argparse
import time

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run SageMaker training job with a specified training script.")
parser.add_argument(
    "--script", 
    type=str, 
    required=True, 
    help="Specify the training script (e.g., 'train_knn.py' or 'train_random_forest.py')"
)
args = parser.parse_args()

# AWS region & S3 bucket setup
region = "ap-south-1"
bucket_name = "learnsagemakeramk"

# Create boto3 & SageMaker session using your preferred bucket
boto_session = boto3.Session(region_name=region)
sagemaker_session = sagemaker.Session(
    boto_session=boto_session,
    default_bucket=bucket_name  # This forces SageMaker to use YOUR bucket
)

# Load environment variables from .env file
load_dotenv()
role = os.getenv("SAGEMAKER_ROLE_ARN")  # Ensure this is set in your .env file

if role is None:
    raise ValueError("SAGEMAKER_ROLE_ARN not found. Check your .env file.")

# Define the S3 input path for training data
prefix = "churn"
train_input_s3 = f"s3://{bucket_name}/{prefix}/processed/train/train.csv"
train_input = sagemaker.inputs.TrainingInput(train_input_s3, content_type="text/csv")
#add val_input_s3
val_input_s3 = f"s3://{bucket_name}/{prefix}/processed/val/val.csv"
val_input = sagemaker.inputs.TrainingInput(val_input_s3, content_type="text/csv")

# Use the script provided as an argument
entry_point_script = args.script

# Generate a valid base job name
base_job_name = os.path.splitext(entry_point_script)[0]  # Remove the file extension
base_job_name = base_job_name.replace("_", "-")  # Replace underscores with hyphens
timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())  # Shorter timestamp format
max_length = 63 - len(timestamp) - 1  # Reserve space for the timestamp and hyphen
base_job_name = f"{base_job_name[:max_length]}-{timestamp}"  # Truncate and append timestamp

# Configure estimator
estimator = SKLearn(
    entry_point=entry_point_script,
    role=role,
    instance_type="ml.m5.large",
    framework_version="1.2-1",
    py_version="py3",
    sagemaker_session=sagemaker_session,
    base_job_name=base_job_name,  # Use the sanitized and shortened base job name
    hyperparameters={},
)

# Launch training job
estimator.fit({
    "train": train_input,
    "val": val_input
})
#passing a "train" channel, and a "val" (or "validation") channel
