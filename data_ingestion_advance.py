import os
import re
import json
import csv
import hashlib
from datetime import datetime
from tqdm import tqdm
import multiprocessing
from functools import partial
from itertools import islice
from collections import defaultdict

from doc_extractors import extract_text_from_doc
from pdf_extractors import extract_text_from_pdf
from bs4 import BeautifulSoup

def file_path_generator(input_directory):
    """Generator function to yield file paths."""
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith(('.pdf', '.doc', '.docx', '.html')):
                yield os.path.join(root, file)

def extract_cleaned_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        html_content = file.read()
    soup = BeautifulSoup(html_content, 'lxml')
    text_content = soup.get_text(separator="\n")
    return "\n".join(line.strip() for line in text_content.splitlines() if line.strip())

class DocumentProcessor:
    def __init__(self, input_directory, output_directory, batch_size=1000, num_processes=None):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.batch_size = batch_size
        self.num_processes = num_processes or max(1, multiprocessing.cpu_count() - 1)

        self.json_output_dir = os.path.join(output_directory, 'json')
        self.csv_output_dir = os.path.join(output_directory, 'csv')
        self.run_dir = os.path.join(output_directory, 'run')
        for dir_path in [self.json_output_dir, self.csv_output_dir, self.run_dir]:
            os.makedirs(dir_path, exist_ok=True)

        self.existing_hashes = self.load_existing_hashes()

    def load_existing_hashes(self):
        existing_hashes = set()
        for filename in os.listdir(self.csv_output_dir):
            if filename.endswith('.csv'):
                with open(os.path.join(self.csv_output_dir, filename), 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    next(reader)
                    for row in reader:
                        if len(row) > 1:
                            existing_hashes.add(row[1])
        return existing_hashes

    @staticmethod
    def generate_hash(content):
        return hashlib.md5(content.encode()).hexdigest()

    @staticmethod
    def get_creation_time(file_path):
        return datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()

    def is_resume(self, text):
        text_lower = text.lower()
        keywords = ["resume", "curriculum vitae", "cv", "professional summary", "objective",
                    "skills", "experience", "education", "certifications", "projects"]
        sections = ["work experience", "professional experience", "employment history",
                    "education", "academic background", "skills", "certifications"]

        keyword_matches = any(keyword in text_lower for keyword in keywords)
        section_matches = any(section in text_lower for section in sections)
        date_pattern = re.compile(r'\b\d{4}[-/]\d{2}[-/]\d{2}\b|\b\d{4}\b')
        job_title_pattern = re.compile(r'\b(software|engineer|developer|manager|analyst|consultant)\b', re.IGNORECASE)

        return (keyword_matches or section_matches) and (bool(date_pattern.search(text)) or bool(job_title_pattern.search(text)))

    @staticmethod
    def clean_text(text):
        text = re.sub(r'\\u\w{4}', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def extract_contact_info(text):
        patterns = {
            'emails': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
            'phone_numbers': r'\b(?:\+?(\d{1,3}))?[-.\s]?(\(?\d{3}\)?)[-.\s]?(\d{3})[-.\s]?(\d{4})\b',
            'linkedin': r'https?://(?:www\.)?linkedin\.com/in/[\w-]+',
            'github': r'https?://(?:www\.)?github\.com/[\w-]+',
            'urls': r'https?://(?:www\.)?\S+\.\S+'
        }
        return {key: re.findall(pattern, text) for key, pattern in patterns.items()}

    def process_document(self, file_path):
        filename = os.path.basename(file_path)
        try:
            if filename.lower().endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
            elif filename.lower().endswith(('.doc', '.docx')):
                text = extract_text_from_doc(file_path)
            elif filename.lower().endswith('.html'):
                text = extract_cleaned_text(file_path)
            else:
                return filename, None, True, False

            hash_id = self.generate_hash(text)
            if hash_id in self.existing_hashes:
                return filename, None, True, False

            text = self.clean_text(text)
            contact_info = self.extract_contact_info(text)
            creation_time = self.get_creation_time(file_path)
            is_resume = self.is_resume(text)

            return filename, {
                "content": text,
                "is_resume": is_resume,
                "contact_info": contact_info,
                "created_time": creation_time,
                "hash_id": hash_id,
            }, False, False
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            return filename, None, False, True

    def save_results(self, results, index):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_file = os.path.join(self.json_output_dir, f'document_processing_results_{timestamp}_{index}.json')
        csv_file = os.path.join(self.csv_output_dir, f'document_hash_ids_{timestamp}_{index}.csv')

        with open(json_file, 'w') as file_handle:
            json.dump(results, file_handle, indent=4)

        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Filename', 'Hash ID', 'Is Resume'])
            for filename, metadata in results.items():
                writer.writerow([filename, metadata['hash_id'], metadata['is_resume']])
                self.existing_hashes.add(metadata['hash_id'])

    def save_summary(self, start_time, file_count, duplicate_files, resume_count, unprocessed_files):
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        summary = {
            "run_start_time": start_time.isoformat(),
            "run_end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "total_files_processed": file_count,
            "total_duplicates_count": len(duplicate_files),
            "total_resumes_detected": resume_count,
            "unprocessed_documents_count": len(unprocessed_files)
        }

        timestamp = end_time.strftime('%Y%m%d_%H%M%S')
        summary_file = os.path.join(self.run_dir, f'document_processing_summary_{timestamp}.json')

        with open(summary_file, 'w') as file_handle:
            json.dump(summary, file_handle, indent=4)

    def process_documents_parallel(self):
        start_time = datetime.now()
        file_generator = file_path_generator(self.input_directory)

        total_processed = 0
        total_duplicates = []
        total_resumes = 0
        total_unprocessed = []

        # Count total files for tqdm
        total_files = sum(1 for _ in file_path_generator(self.input_directory))

        with multiprocessing.Pool(processes=self.num_processes) as pool:
            with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
                for batch_index, batch in enumerate(iter(lambda: list(islice(file_generator, self.batch_size)), []), 1):
                    results = pool.map(self.process_document, batch)

                    processed_files = {}
                    batch_duplicates = []
                    batch_resumes = 0
                    batch_unprocessed = []

                    for filename, content, duplicate_flag, unprocessed_flag in results:
                        if content is not None:
                            processed_files[filename] = content
                            if content['is_resume']:
                                batch_resumes += 1
                        if duplicate_flag:
                            batch_duplicates.append(filename)
                        if unprocessed_flag:
                            batch_unprocessed.append(filename)

                    self.save_results(processed_files, batch_index)

                    total_processed += len(batch)
                    total_duplicates.extend(batch_duplicates)
                    total_resumes += batch_resumes
                    total_unprocessed.extend(batch_unprocessed)

                    # Update progress bar
                    pbar.update(len(batch))

                    # Calculate and display estimated time remaining
                    elapsed_time = (datetime.now() - start_time).total_seconds()
                    files_per_second = total_processed / elapsed_time
                    remaining_files = total_files - total_processed
                    estimated_time_remaining = remaining_files / files_per_second

                    pbar.set_postfix({
                        'Batch': batch_index,
                        'Processed': total_processed,
                        'Resumes': total_resumes,
                        'ETA': f"{estimated_time_remaining:.2f}s"
                    })

        self.save_summary(start_time, total_processed, total_duplicates, total_resumes, total_unprocessed)

if __name__ == '__main__':
    input_directory = "/root/data/Resume_22Sep2024_10GB"
    output_directory = "/root/data/ProcessedResults/"
    batch_size = 1000
    num_processes = 6
    
    processor = DocumentProcessor(input_directory, output_directory, batch_size, num_processes)
    processor.process_documents_parallel()