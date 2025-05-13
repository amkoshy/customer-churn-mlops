import boto3
import sagemaker
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from dotenv import load_dotenv
import os


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

# Create the SKLearnProcessor object
sklearn_processor = SKLearnProcessor(
    framework_version="1.0-1",  # Must be a supported version
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    sagemaker_session=sagemaker_session
)

# Input and output locations (within your own bucket)
input_s3_uri = f"s3://{bucket_name}/WA_Fn-UseC_-Telco-Customer-Churn.csv"
train_output_s3_uri = f"s3://{bucket_name}/churn/processed/train"
val_output_s3_uri = f"s3://{bucket_name}/churn/processed/val"
test_output_s3_uri = f"s3://{bucket_name}/churn/processed/test"

# Run the preprocessing job
sklearn_processor.run(
    code="preprocessing.py",  # Your local script that does pandas-based processing
    inputs=[
        ProcessingInput(
            source=input_s3_uri,
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/train",
            destination=train_output_s3_uri
        ),
        ProcessingOutput(
            source="/opt/ml/processing/val",
            destination=val_output_s3_uri
        ),
        ProcessingOutput(
            source="/opt/ml/processing/test",
            destination=test_output_s3_uri
        )
    ]
)

print("✅ Preprocessing job started using your own S3 bucket.")
