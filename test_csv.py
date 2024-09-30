import os
import pandas as pd

def get_csv_row_counts(directory_path):
    """
    Counts the number of rows in each CSV file in the given directory and the total number of files.

    Parameters:
    directory_path (str): Path to the directory containing CSV files.

    Returns:
    tuple: A dictionary with file names as keys and row counts as values, and the total number of files.
    """
    row_counts = {}
    total_files = 0

    # Check if the provided path exists
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return row_counts, total_files

    # Iterate through the directory to count rows in each CSV file
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(directory_path, file_name)
            try:
                # Read the CSV file and get the number of rows
                df = pd.read_csv(file_path)
                row_count = len(df)
                row_counts[file_name] = row_count
                total_files += 1
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

    return row_counts, total_files

# Directory path to look for CSV files
directory_path = "/root/data/ProcessedResults/csv"

# Get the row counts and total file count
csv_row_counts, csv_file_count = get_csv_row_counts(directory_path)

# Display results
print(f"Total number of CSV files: {csv_file_count}")
print("Row counts per file:")
for file, count in csv_row_counts.items():
    print(f"{file}: {count} rows")

# Calculate and display the total number of rows across all files
total_rows = sum(csv_row_counts.values())
print(f"Total number of rows across all CSV files: {total_rows}")
