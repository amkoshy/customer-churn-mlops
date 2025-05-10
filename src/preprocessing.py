import os
import shutil
import kagglehub

#
# Check if the data directory is empty
data_dir = "data/"
if not os.path.exists(data_dir) or not os.listdir(data_dir):
    # Download the dataset to the default location
    path = kagglehub.dataset_download("blastchar/telco-customer-churn")
    print("Path to dataset files:", path)


    # Move the downloaded files to the data directory
    for file in os.listdir(path):
        if file.endswith(".csv"):
            shutil.move(os.path.join(path, file), os.path.join(data_dir, file))
            print(f"Moved {file} to {data_dir}")
    # Remove the downloaded files                                   
    shutil.rmtree(path)
else:       
    print("Data directory is not empty. Skipping download.")
    # List the files in the data directory
    print("Files in data directory:")
    for file in os.listdir(data_dir):
        print(file)