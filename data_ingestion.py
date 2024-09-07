import os
import re
import io
import json
import csv
import hashlib
from datetime import datetime
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import docx2txt
import subprocess
import tempfile
import multiprocessing
import time
from tqdm import tqdm

from doc_extractors import extract_text_from_doc
from pdf_extractors import extract_text_from_pdf

class DocumentProcessor:
    def __init__(self, input_directory, output_directory, num_processes=8, save_interval=100):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.num_processes = num_processes
        self.save_interval = save_interval

        # Ensure output directories exist
        self.json_output_dir = os.path.join(output_directory, 'json')
        self.csv_output_dir = os.path.join(output_directory, 'csv')
        self.run_dir = os.path.join(output_directory, 'run')
        os.makedirs(self.json_output_dir, exist_ok=True)
        os.makedirs(self.csv_output_dir, exist_ok=True)
        os.makedirs(self.run_dir, exist_ok=True)

        self.existing_hashes = self.load_existing_hashes()

    def load_existing_hashes(self):
        """
        Load existing hash IDs from all CSV files in the output directory to avoid processing duplicates.
        """
        existing_hashes = set()
        
        for filename in os.listdir(self.csv_output_dir):
            if filename.endswith('.csv'):
                csv_file = os.path.join(self.csv_output_dir, filename)
                with open(csv_file, 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    next(reader)  # Skip header
                    for row in reader:
                        if len(row) > 1:  # Ensure there's a second column
                            existing_hashes.add(row[1])  # Hash ID is in the second column
        
        return existing_hashes

    # @staticmethod
    # def extract_text_from_pdf(pdf_path):
    #     """
    #     Extract text from a PDF file using pdfminer.
    #     """
    #     resource_manager = PDFResourceManager()
    #     fake_file_handle = io.StringIO()
    #     converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    #     page_interpreter = PDFPageInterpreter(resource_manager, converter)
        
    #     with open(pdf_path, 'rb') as file_handle:
    #         for page in PDFPage.get_pages(file_handle, caching=True, check_extractable=True):
    #             page_interpreter.process_page(page)
            
    #         text = fake_file_handle.getvalue()
        
    #     converter.close()
    #     fake_file_handle.close()
        
    #     return text

    @staticmethod
    def extract_text_from_doc_old(doc_path):
        """
        Extract text from a DOC or DOCX file.
        """
        try:
            text = docx2txt.process(doc_path)
        except Exception:
            try:
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_filename = temp_file.name
                
                subprocess.run(['antiword', doc_path], stdout=open(temp_filename, 'w'))
                
                with open(temp_filename, 'r') as file_handle:
                    text = file_handle.read()
                
                os.unlink(temp_filename)
            except Exception as e:
                raise Exception(f"Failed to process DOC file: {str(e)}")
        
        return text

    @staticmethod
    def generate_hash(content):
        """
        Generate a unique hash for the given content.
        """
        return hashlib.md5(content.encode()).hexdigest()

    @staticmethod
    def get_creation_time(file_path):
        """
        Get the creation time of a file.
        """
        timestamp = os.path.getctime(file_path)
        return datetime.fromtimestamp(timestamp).isoformat()

    def is_resume(self, text):
        """
        Determine if the text is likely a resume by analyzing its content.
        """
        text_lower = text.lower()
        
        keywords = [
            "resume", "curriculum vitae", "cv", "professional summary", "objective", 
            "skills", "experience", "education", "certifications", "projects", 
            "languages", "references", "awards", "honors"
        ]
        
        sections = [
            "work experience", "professional experience", "employment history", 
            "education", "academic background", "skills", "certifications", 
            "projects", "summary", "languages", "references", "awards", "honors"
        ]
        
        date_pattern = re.compile(r'\b\d{4}[-/]\d{2}[-/]\d{2}\b|\b\d{4}\b')
        job_title_pattern = re.compile(r'\b(software|engineer|developer|manager|analyst|consultant|specialist)\b', re.IGNORECASE)
        company_name_pattern = re.compile(r'\b(inc|llc|corp|ltd|company|co|limited|group)\b', re.IGNORECASE)

        keyword_matches = any(keyword in text_lower for keyword in keywords)
        section_matches = any(section in text_lower for section in sections)
        date_matches = bool(date_pattern.search(text))
        job_title_matches = bool(job_title_pattern.search(text))
        company_name_matches = bool(company_name_pattern.search(text))
        
        resume_likelihood = (keyword_matches or section_matches) and (date_matches or job_title_matches or company_name_matches)
        
        return resume_likelihood

    @staticmethod
    def clean_text(text):
        text = re.sub(r'\n+', ' ', text)
        # Remove special unicode characters like \u00a0 (non-breaking spaces)
        text = re.sub(r'\\u\w{4}', ' ', text)
        # Remove multiple tabs and excess spaces
        text = re.sub(r'\t+', ' ', text)
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        # Trim leading and trailing spaces
        text = text.strip()
        
        return text

    @staticmethod
    def extract_contact_info(text):
        # Define regex patterns for email addresses, phone numbers, LinkedIn URLs, GitHub URLs, and other URLs
        patterns = {
            'emails': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
            'phone_numbers': r'\b(?:\+?(\d{1,3}))?[-.\s]?(\(?\d{3}\)?)[-.\s]?(\d{3})[-.\s]?(\d{4})\b',
            'linkedin': r'https?://(?:www\.)?linkedin\.com/in/[\w-]+',
            'github': r'https?://(?:www\.)?github\.com/[\w-]+',
            'urls': r'https?://(?:www\.)?\S+\.\S+'
        }
        
        # Extract information using regex patterns
        extracted_info = {key: re.findall(pattern, text) for key, pattern in patterns.items()}
        
        return extracted_info
    def process_document(self, file_path):
        """
        Process a document, extracting its content, hash, and creation time.
        """
        filename = os.path.basename(file_path)
        
        try:
            if filename.lower().endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
            elif filename.lower().endswith(('.doc', '.docx')):
                text = extract_text_from_doc(file_path)
            else:
                raise ValueError(f"Unsupported file type: {filename}")
            
            hash_id = self.generate_hash(text)
            if hash_id in self.existing_hashes:
                return filename, None
            
            text = DocumentProcessor.clean_text(text)
            contact_info = DocumentProcessor.extract_contact_info(text)
            
            creation_time = self.get_creation_time(file_path)
            is_resume = self.is_resume(text)
            
            return filename, {"content": text,  
                              "is_resume": is_resume, 
                              "contact_info":contact_info,
                              "created_time": creation_time, 
                              "hash_id": hash_id,}
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            return filename, None

    def save_results(self, results, index):
        """
        Save the processed results to JSON and CSV files.
        """
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

    def save_summary(self, start_time, file_count, duplicate_count, resume_count, unprocessed_files):
        """
        Save a summary of the processing run to a JSON file.
        """
        end_time = time.time()
        duration = end_time - start_time
        
        summary = {
            "run_start_time": datetime.fromtimestamp(start_time).isoformat(),
            "run_end_time": datetime.fromtimestamp(end_time).isoformat(),
            "duration_seconds": duration,
            "total_files_processed": file_count,
            "total_duplicates_detected": duplicate_count,
            "total_resumes_detected": resume_count,
            "unprocessed_documents": unprocessed_files  # Add unprocessed documents to the summary
        }

        print(summary)
        
        # Format the datetime for the filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_filename = f'document_processing_summary_{timestamp}.json'
        summary_file = os.path.join(self.run_dir, summary_filename)
        
        with open(summary_file, 'w') as file_handle:
            json.dump(summary, file_handle, indent=4)

    def process_documents_parallel(self):
        """
        Process documents in parallel and save results at intervals.
        """
        start_time = time.time()
        file_paths = [os.path.join(self.input_directory, f) for f in os.listdir(self.input_directory) 
                      if f.lower().endswith(('.pdf', '.doc', '.docx'))]
        
        processed_files = {}
        duplicate_count = 0
        resume_count = 0
        unprocessed_files = []

        with multiprocessing.Pool(processes=self.num_processes) as pool:
            for i, result in enumerate(tqdm(pool.imap_unordered(self.process_document, file_paths), 
                                           total=len(file_paths), 
                                           desc="Processing Documents"), 1):
                # Unpack result properly: result is a tuple (filename, content)
                filename, content = result
                if content is not None:
                    processed_files[filename] = content
                    if content['is_resume']:
                        resume_count += 1
                else:
                    unprocessed_files.append(filename)  # Add filename to unprocessed list
                    duplicate_count += 1
                
                if i % self.save_interval == 0:
                    self.save_results(processed_files, i)
                    processed_files.clear()  # Reset the results to save memory

        if processed_files:
            self.save_results(processed_files, 'final')

        # Save the summary after processing
        self.save_summary(start_time, len(file_paths), duplicate_count, resume_count, unprocessed_files)


    def process_specific_files(self, file_list):
        """
        Process a specific list of files, extracting their content, hash, and creation time.
        
        Parameters:
        - file_list: list of str, filenames to process.
        """
        start_time = time.time()
        
        # Build the file paths based on the provided list
        file_paths = [os.path.join(self.input_directory, f) for f in file_list
                      if os.path.isfile(os.path.join(self.input_directory, f))]
        
        processed_files = {}
        duplicate_count = 0
        resume_count = 0
        unprocessed_files = []

        # Process the files
        with multiprocessing.Pool(processes=self.num_processes) as pool:
            for i, result in enumerate(tqdm(pool.imap_unordered(self.process_document, file_paths), 
                                           total=len(file_paths), 
                                           desc="Processing Specific Documents"), 1):
                filename, content = result
                if content is not None:
                    processed_files[filename] = content
                    if content['is_resume']:
                        resume_count += 1
                else:
                    unprocessed_files.append(filename)
                    duplicate_count += 1
                
                if i % self.save_interval == 0:
                    self.save_results(processed_files, i)
                    processed_files.clear()

        if processed_files:
            self.save_results(processed_files, 'final')

        # Save the summary
        self.save_summary(start_time, len(file_paths), duplicate_count, resume_count, unprocessed_files)
        
if __name__ == '__main__':
    
    input_directory = "/root/Batch1"
    output_directory = "/root/ProcessedResults"    
    
    num_processes = 8
    save_interval = 100

    processor = DocumentProcessor(input_directory, output_directory, num_processes, save_interval)
    #processor.process_documents_parallel()

    specific_files = [
        "ZHQI0pwKtJtVsP10Gn6e29vrKRSBOqo2M5EWYd25.doc",
        "Zuoke_Okoro_resume.pdf"
    ]
    # Process only the files in the list
    processor.process_specific_files(specific_files)

    