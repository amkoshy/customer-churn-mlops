import argparse
import boto3
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Invoke SageMaker endpoint for inference")
    parser.add_argument('--endpoint', required=True, help='Name of the deployed SageMaker endpoint')
    parser.add_argument('--input', required=True, help='Path to input JSON file')
    return parser.parse_args()

def main():
    args = parse_args()

    # Read input data from JSON file
    with open(args.input, 'r') as f:
        input_data = json.load(f)

    # Convert input to JSON string
    payload = json.dumps(input_data)

    # Set up SageMaker runtime client
    client = boto3.client('sagemaker-runtime')

    # Invoke the endpoint
    response = client.invoke_endpoint(
        EndpointName=args.endpoint,
        ContentType='application/json',
        Body=payload
    )

    # Read and print prediction
    result = response['Body'].read().decode('utf-8')
    print("🧠 Prediction:", result)

if __name__ == '__main__':
    main()
