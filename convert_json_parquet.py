import os
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm
import re

# Specify the folder paths
json_folder_path = '/root/data/ProcessedResults/json'  # Replace with your JSON folder path
parquet_folder_path = '/root/data/ProcessedResults/parquet'  # Replace with your Parquet folder path

# Create the parquet folder if it doesn't exist
Path(parquet_folder_path).mkdir(parents=True, exist_ok=True)

def clean_html_text(text):
    """
    Clean the input HTML text by removing unwanted special characters, HTML tags, 
    and extra whitespace. Returns the cleaned text.
    """
    # Remove HTML tags and unwanted characters
    cleaned_text = re.sub(r'<[^>]+>', '', text)
    cleaned_text = re.sub(r'\s*([a-zA-Z-]+:\s*[^;]+;?)\s*', '', cleaned_text)
    cleaned_text = re.sub(r'url\(\s*.*?\)', '', cleaned_text)
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s.,;:?!\'"()\-]', '', cleaned_text)  # Keep alphanumeric and some punctuation
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # Replace multiple spaces with one
    return cleaned_text

# Define the consistent schema with default values
consistent_columns = ['key','content','hash_id']

def read_resumes_from_json(file_path):
    """Extracts relevant data from the JSON file, ensuring all columns are consistent."""
    extracted_data = []
    with open(file_path, 'r') as file:
        data = json.load(file)
        for key, value in data.items():
            if ".html" not in key:
                # Ensure all required fields are extracted, with default values for missing fields
                #clean_text = clean_html_text(value.get('content', ''))
                extracted_data.append({
                    'key': key,
                    'content': value.get('content', ''),  # Clean text if it's HTML
                    #'is_resume': value.get('is_resume', False),  # Default to False if not present
                    #'created_time': value.get('created_time', None),  # Default to None if not present
                    'hash_id': value.get('hash_id', ''),  # Default to empty string if not present
                    #'emails': value.get('contact_info', {}).get('emails', [])  # Default to empty list if not present
                })
    return extracted_data

def process_json_to_parquet(file_name):
    """Converts a JSON file to a Parquet file, ensuring consistent schema."""
    try:
        file_path = os.path.join(json_folder_path, file_name)
        # Extract the data
        resume_data = read_resumes_from_json(file_path)
        
        # Convert the extracted data to a DataFrame
        df = pd.DataFrame(resume_data, columns=consistent_columns)  # Use consistent schema

        # Specify the parquet file path
        parquet_file_path = os.path.join(parquet_folder_path, f"{os.path.splitext(file_name)[0]}.parquet")
        
        # Write DataFrame to Parquet with consistent schema and compression
        df.to_parquet(parquet_file_path, index=False, compression='snappy')

    except Exception as e:
        print(f"Error processing {file_name}: {e}")

def process_files_in_parallel():
    """Processes all JSON files in parallel and converts them to Parquet."""
    json_files = [f for f in os.listdir(json_folder_path) if f.endswith('.json')]
    
    # Use ThreadPoolExecutor to process files in parallel
    with ThreadPoolExecutor() as executor:
        # Wrap the executor with tqdm for progress tracking
        list(tqdm(executor.map(process_json_to_parquet, json_files), total=len(json_files), desc="Processing files", unit="file"))

if __name__ == "__main__":
    process_files_in_parallel()
