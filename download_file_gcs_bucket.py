# from google.cloud import storage

# # Authenticate with GCS
# storage_client = storage.Client()

# # Specify the bucket name
# BUCKET_NAME = "document_ocr_sr"  # Replace with your actual bucket name
# OUTPUT_FILE = "blobs.txt"  # Specify the output file path

# def list_blobs_to_file(bucket_name, output_file):
#     """List all blobs in the specified GCS bucket and store their paths in a text file."""
#     bucket = storage_client.bucket(bucket_name)
#     blobs = bucket.list_blobs()

#     with open(output_file, 'w') as f:
#         for blob in blobs:
#             f.write(f"{blob.name}\n")  # Write each blob name to the file
#             print(f"Blob path: {blob.name} stored in {output_file}")

# if __name__ == "__main__":
#     list_blobs_to_file(BUCKET_NAME, OUTPUT_FILE)
#     print(f"All blob paths have been stored in {OUTPUT_FILE}")



# import pandas as pd

# # Read the processed files CSV and remove the '.npz' extension
# df1 = pd.read_csv("/root/data/ProcessedResults/processed_files.csv", header=None)
# df1[0] = df1[0].str.replace('.npz', '', regex=False)

# # Rename the column to 'filenames' to use in the merge
# df1.columns = ['filenames']
# files_to_download = df1['filenames']

# # Read the blobs.txt with the correct delimiter (adjust it if necessary)
# df2 = pd.read_csv("blobs.txt", header=None, delimiter='################')

# # Extract the filenames from the file paths
# df2[1] = df2[0].str.split('/').str[-1]

# # Rename the extracted filenames column to 'filenames' to match df1
# df2.columns = ['filepaths', 'filenames']

# # Perform the right join on 'filenames'
# result = pd.merge(df1, df2, on='filenames', how='inner')

# # Print the shape of df2 (before or after the merge depending on your requirement)
# print(result.shape)

# # Optionally, you can print the result of the merge
# result.to_csv("/root/data/ProcessedResults/Download_info.csv", index=False)

import os
import pandas as pd
from google.cloud import storage
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Initialize Google Cloud Storage client
client = storage.Client()

# Define your GCS bucket name
bucket_name = 'document_ocr_sr'
bucket = client.bucket(bucket_name)

# Assuming the DataFrame is already loaded with 'filenames' and 'filepaths' columns
df = pd.read_csv("/root/data/ProcessedResults/Download_info.csv")

# Define the local folder to download the files
local_download_folder = '/root/data/ProcessedResults/resume'

# Ensure the local folder exists
os.makedirs(local_download_folder, exist_ok=True)

# Function to download a file from GCS
def download_file_from_gcs(blob_path, local_filename):
    blob = bucket.blob(blob_path)
    local_file_path = os.path.join(local_download_folder, local_filename)
    
    # Download the file
    blob.download_to_filename(local_file_path)
    print(f"Downloaded {blob_path} to {local_file_path}")

# Use ThreadPoolExecutor for parallel processing
def download_files(df):
    with ThreadPoolExecutor() as executor:
        # Use tqdm to show progress
        list(tqdm(executor.map(
            lambda row: download_file_from_gcs(row[1]['filepaths'], row[1]['filenames']),
            df.iterrows()
        ), total=len(df)))

# Call the function to download files
download_files(df)
