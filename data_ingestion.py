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

class DocumentProcessor:
    def __init__(self, input_directory, output_directory, num_processes=4, save_interval=100):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.num_processes = num_processes
        self.save_interval = save_interval

        # Ensure output directories exist
        self.json_output_dir = os.path.join(output_directory, 'json')
        self.csv_output_dir = os.path.join(output_directory, 'csv')
        os.makedirs(self.json_output_dir, exist_ok=True)
        os.makedirs(self.csv_output_dir, exist_ok=True)

        self.existing_hashes = self.load_existing_hashes()

    def load_existing_hashes(self):
        """
        Load existing hash IDs from the CSV file to avoid processing duplicates.
        """
        existing_hashes = set()
        csv_file = os.path.join(self.csv_output_dir, 'document_hash_ids_final.csv')
        
        if os.path.exists(csv_file):
            with open(csv_file, 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip header
                for row in reader:
                    existing_hashes.add(row[1])  # Hash ID is in the second column
        
        return existing_hashes

    @staticmethod
    def extract_text_from_pdf(pdf_path):
        """
        Extract text from a PDF file using pdfminer.
        """
        resource_manager = PDFResourceManager()
        fake_file_handle = io.StringIO()
        converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
        page_interpreter = PDFPageInterpreter(resource_manager, converter)
        
        with open(pdf_path, 'rb') as file_handle:
            for page in PDFPage.get_pages(file_handle, caching=True, check_extractable=True):
                page_interpreter.process_page(page)
            
            text = fake_file_handle.getvalue()
        
        converter.close()
        fake_file_handle.close()
        
        return text

    @staticmethod
    def extract_text_from_doc(doc_path):
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

    def process_document(self, file_path):
        """
        Process a document, extracting its content, hash, and creation time.
        """
        filename = os.path.basename(file_path)
        
        try:
            if filename.lower().endswith('.pdf'):
                text = self.extract_text_from_pdf(file_path)
            elif filename.lower().endswith(('.doc', '.docx')):
                text = self.extract_text_from_doc(file_path)
            else:
                raise ValueError(f"Unsupported file type: {filename}")
            
            hash_id = self.generate_hash(text)
            if hash_id in self.existing_hashes:
                return filename, None
            
            creation_time = self.get_creation_time(file_path)
            
            is_resume = self.is_resume(text)
            
            return filename, {"content": text, "created_time": creation_time, "hash_id": hash_id, "is_resume": is_resume}
        except Exception as e:
            return filename, None

    def save_results(self, results, index):
        """
        Save the processed results to JSON and CSV files.
        """
        json_file = os.path.join(self.json_output_dir, f'document_processing_results_{index}.json')
        csv_file = os.path.join(self.csv_output_dir, f'document_hash_ids_{index}.csv')

        with open(json_file, 'w') as file_handle:
            json.dump(results, file_handle, indent=4)
        
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Filename', 'Hash ID', 'Is Resume'])
            for filename, metadata in results.items():
                writer.writerow([filename, metadata['hash_id'], metadata['is_resume']])
                self.existing_hashes.add(metadata['hash_id'])

    def save_summary(self, start_time, file_count, duplicate_count, resume_count):
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
            "total_resumes_detected": resume_count
        }

        # Format the datetime for the filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_filename = f'document_processing_summary_{timestamp}.json'
        summary_file = os.path.join(self.json_output_dir, summary_filename)
        
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
        
        with multiprocessing.Pool(processes=self.num_processes) as pool:
            for i, (filename, content) in enumerate(tqdm(pool.imap_unordered(self.process_document, file_paths), 
                                                         total=len(file_paths), 
                                                         desc="Processing Documents"), 1):
                if content is not None:
                    processed_files[filename] = content
                    if content['is_resume']:
                        resume_count += 1
                else:
                    duplicate_count += 1
                
                if i % self.save_interval == 0:
                    self.save_results(processed_files, i)
        
        if len(processed_files) % self.save_interval != 0:
            self.save_results(processed_files, 'final')
        
        # Save summary
        self.save_summary(start_time, len(file_paths), duplicate_count, resume_count)
        
        return processed_files

if __name__ == "__main__":
    input_directory = "/root/Batch1"
    output_directory = "/root/ProcessedResults"

    processor = DocumentProcessor(input_directory, output_directory)
    processed_documents = processor.process_documents_parallel()
    
    print(f"Total processed documents: {len(processed_documents)}")
    for filename, metadata in list(processed_documents.items())[:5]:  # Print preview of first 5 documents
        print(f"Processed {filename}")
        print(f"Content preview: {metadata['content'][:100]}...")  # Print a preview of the content
