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

from doc_extractors import (
    extract_text_using_pywin32,
    extract_text_using_pypandoc,
    extract_text_using_docx2python,
    extract_text_using_textract,
    extract_text_using_pymupdf,
    extract_text_using_python_docx
)

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
    def extract_text_from_doc(doc_path):
        """
        Extract text from a DOC or DOCX file using multiple methods.
        """
        extract_methods = [
            extract_text_using_pywin32,
            extract_text_using_pypandoc,
            extract_text_using_docx2python,
            extract_text_using_textract,
            extract_text_using_pymupdf,
            extract_text_using_python_docx
        ]
        
        for method in extract_methods:
            try:
                text = method(doc_path)
                return text
            except Exception as e:
                print(f"Method {method.__name__} failed")
        
        raise Exception(f"Failed to extract text from {doc_path} using all available methods.")

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
                text = self.extract_text_from_pdf(file_path)
            elif filename.lower().endswith(('.doc', '.docx')):
                text = self.extract_text_from_doc(file_path)
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
    
    input_directory = r"C:\Users\91798\Downloads\resumes_7Sep2024\Batch1"
    output_directory = r"C:\Users\91798\Downloads\llm-rag\ProcessedResults"    
    
    num_processes = 8
    save_interval = 100

    processor = DocumentProcessor(input_directory, output_directory, num_processes, save_interval)
    #processor.process_documents_parallel()


    specific_files = [
        "zGqUR7byLmNGF4OoBfrMI4EplZbx2Qf1uaTtt9PI.doc",
        "ZHQI0pwKtJtVsP10Gn6e29vrKRSBOqo2M5EWYd25.doc",
        "zHxvdk8P7P81f80V202109aiwwvOMkPhDkXVVKorwvCobUuOBeon5n3coGD55c.doc",
        "zIgmEHPW2M0FH26r202202qpICLa8wbcjKrYnXCmRWiuYpsv2B6KLSmIVGcZ6q.DOC",
        "zIvtlLniI1F8jILM202307FdHTx7wgKSrJ0N577FoLslZQXCVvFbPepVNh6wmN.doc",
        "zj63QYxSw9zXwyLRMMXjh0Lv4ksWCptsFJw7Ob8w.doc",
        "ZjBpqwzjHnHzGW3Q5n8U7iYmjXb2sXrioJgUICpbyDBI348cWcg2lAhgeZZLm1vb.pdf",
        "Zk0ZaDjdcQklNTvhH6sant50WJGvZI0U7caqY5uy.doc",
        "zkcJWsSqvngKj0b8202109QZRzzqvqRs1PmgRagtAskG6UcNtUwoA0shnNg4Ya.doc",
        "zkFuC03W6OUM4vUn202104z9b9QIBERVzloMSfHhBMUnCIGukBOV5t1QZgYcNX.doc",
        "ZKrkNY1mrjeTZiZymIIIGiy2hu4cWVfpmKz0hzDF.doc",
        "ZLVktw3FY73hJZvV4RLLwGMMOlw1wYNOUjthXTMe.doc",
        "Zm29Um2OQt7TZJMx1SfPuzx62bdfrlP5pMBQpaim.doc",
        "ZM2IzMZK5ooor7zYUS.doc",
        "zMEd4FjyuH3JzgfK202305HxmmzAlJ6rPlmW9nHitIQcOrFfiUgsW1YWSGIMzL.doc",
        "zmFVsKgKFNXirp9O202104gHYlZcoeqonIkuJ6SNKvrLKiHOUJZZQaeh4CdDfs.doc",
        "zmgyGdeRJqwb6WPL2022062KB1NdE3WOzvfC71ToOJeSAqVoFxFAqqoBYf0Osp.doc",
        "Zmhd2Xmig7g4V0KL202208bHcgCKjpKRobieu4qZFT035EsfpKYZS0QDEk5EiI.doc",
        "zn0sg6fwFc1P2mYp202208bs1A7JY4pKU2ByLobk1i9Kq2qxhexmOwyHzcHJ7gmRzkcbfOSRGdXBP9En7FHS7L.pdf",
        "ZNGtmM0Z4DL3TiyJx6TVDSLy3NGhNGFC2xzC0NiA.pdf",
        "zNHJttEGnQG34H2qAnas.pdf",
        "znPfQSaBAdG2qBu0202206eGJsnmqv2SxWk4n4JYAGNlhz04aW7QY22ZElCBoMsi2a6oVDkmD2BiPlAGBEtl63.pdf",
        "zoe7fXmiOyTEyax4202107FV5biuCUHvmnjjIB5Ay8D3xvviarSfrQVrKQRfAp.doc",
        "ZolxeT54DxAjVLZE202208rQHGXvkCZY7sBlbCgCVlSBbBFWVOjXQoA4pLFkjW.doc",
        "ZoY5GpNn9Dd7Wo1G202108nHsJ44E3JAFhnZkwmHtQDskcG4OmpkkKojYrK6ab.doc",
        "ZpUb2cKo0PmP3gcc202208muUcCjZ1RI3ijmxAyFS5W35HosFVT5VT6J3DCUTxd2CDj9Hk8UYodeBK96nzOnV8.pdf",
        "ZQCyxVoKrqeyRjOl202302NXYC8R2PHxX7xtJGut4UcHgbDuRQ2rwqpMAxdooS.doc",
        "ZqDrC7wJ367y3Sp1aX3TiFQZs2ku2PIaR0UNAOVF.doc",
        "zqrgZ47UZKZ0P83G202304EJO0B94rp8z9FabosZSJm7ewYs0Chv25aPYW32Iu.doc",
        "zQTrR78R8HMbfYyblAzg6Rt8pwDB0z0HPN5Ddw5N.doc",
        "zqr3aqh7f1RFI9zSgAwttI4jm6KWag8zxcdYHrgR5l3CxpObwKgfFAGECh2UNR1i.pdf",
        "ZqvhwCVglcqaJEog202210ZmuTL7pnNJqEywRONjlsKenyJAiamL23NobeEinF.docx",
        "zqU93byyIhYQ6Nzd202204AKTXRgude8asVogtRniHl2lOkWllMwu5wuCEKf3a05doITQTDAQeq9tynFnlHS7w.pdf",
        "zr0syP7l2JbdvirA202209nm28HeVNmVlETtxtv82ISvwPtNacWZ1XEhSjEoT8kmngYgRfniCY7GCXPMuIsQto.pdf",
        "Zr4v8SUNNeJJBu8L202303HKdhwhfVVyjgd0l22WiTq7CRWmt9xW2cccAK88C0.doc",
        "zRdVFQBhIm5yHOyWykpIADknByUxsas1k3seyUXu.pdf",
        "ZribQKCMjpO0kCgF3AtTGs5HtwcqGdpwCmXrGH9h1RKPA2YYsAdZ00mxmrAfvOJb.pdf",
        "ZrTZe3MKLt83rcVyOhHJPy9cZsF7ojvoWrJnAJi7DIU43kIP7OLxyCMsh9yxriiq.pdf",
        "ZrylKXhTYxNCke3SjlgU9LvdZuiBrdvPdXS6j6pE.doc",
        "ZRkJv2zV9Q9fXpSk8bplEbyIVhdZ5fcKmMU9fgT289vspfcvEXMAJWxsqmNTOqUv.pdf",
        "zs6h6gzv27XugHPbXW1PrsCRU8XLwB2tBRIUhAPM.doc",
        "zrkhXTSefFvPowgNJMc5KMfzPCoUEA9xukVUekZloXXmKCTOS7rjag1f6TwLgass.pdf",
        "ZSmxs4C2waNYe12rsvCG4ZdPLGuIxRMdfNNFM8YX.doc",
        "zsGIvrn6Yl93UXzVfNc4EQJAsBbwmRlZqYgSiI8zqtEHLSoOnpdxwULY81uH4tOe.pdf",
        "zsWmShjE7rCqp7ZCSLK5pFAc9Rp76GgsjJXGigxZzb4KUwZEvVVjyDoWk8vczQir.pdf",
        "zt38uyjl9QOLkBz0xmdr0anBgHNPsOg6Au1hPsfNaR52rDhtRNrUSyFWzjlmw2GV.pdf",
        "ztV7JdFGds2jw0gv202208HawnCmrIWGwBVSc8Ukqh3MOlJNCxVQOeq5y7KWokQP3JPMeSEWnud3TuU6bom8Vj.pdf",
        "Zubair_Hangargekar_Resume.pdf",
        "ZuD2jgjxSG1JAzsxRGoIL2q0nSPBrNHWKAdjNlhY.doc",
        "ZUCyd4ABfjFx5J2A202109aazKB1XfciUVLmRomBImyDPVZ9RBlaHGVHJwvApJtfuc6cwfhyGaM6IBmNwl4sv4.pdf",
        "Zuby_Ahmed_resume.pdf",
        "zumkVwmKOfDrIa2oASHISH.docx",
        "Zuoke_Okoro_resume.pdf",
        "zuqoskqjrWmgPzPMDzZKCpt2mJCFkA01iJ7VA5rL.doc",
        "zuJqAX4QckkNPeRj2022116UwRyBzD2eGMjPEhWhdtIZTfVv9QibqWbIUC1Xxk7EcYSEVtGFXQcFBcaLXf5TxD.pdf",
        "ZUXtJjkFZ1l7nHVkwIXgnalMSpp1da2IotvLgOUV.pdf",
        "Zv5lXvvuP4EzEUelEVELYN.doc",
        "ZV9Kmt3AkWhOJjqYGurpinder.docx",
        "zVe7WUNovHzNZwDb2021083718fhTMrL04lWvZAfCGaI5DAb0tOnirt95C1Gtb.doc",
        "zvvfeWYEBvL5oLxxBhwOlUlL7DWsE5uQfuqRfpDK.pdf",
        "ZwfovBd6VpEAciCr2022076skdlBhK3ygx8dbBVERN.docx",
        "ZW6p7llYWjQeq2nk202107zH3ynmil6EiVj8wx5FvMO87x3cO4tIKEltEVYVwrqS7gttYUMJBUXajulMheOflg.pdf",
        "ZwiGZFEqKjZRCKymSsZFwE1ETtlRLFmz6SqkdlMz.doc",
        "ZwJqVyGOnnEOdNHhVzS7RBCjmdlLJAlpdU67yWul.pdf",
        "ZWlJXsw5xGDbZb4AMahesh.doc",
        "zwM1e2iO35AA12Is202109aazKB1XfciUVLmRomBImyDPVZ9RBlaHGVHJwvApJtfuc6cwfhyGaM6IBmNwl4sv4.pdf",
        "ZWSH8GCyAKVUYcgJ202209UWPObCEYi9gxNRg4ZwdqqDcvqiFw9mjwxdVWC2HZ.doc",
        "ZwVbS0utv655lESV2022076skdlBhK3ygx8dbBVERN.docx",
        "zx0DNSZJAGd0T8PIif6ew1RbKhZ4F5mXTBs0qW1i.pdf",
        "ZwVk4jWVHlUuic12SvmqImCynao7gDaDBsbhF0wY2ksaA9zeHRMI4PvRkNUzMe00.pdf",
        "ZXglIGiCAXhmEWXycj2BRqasVDNDE0o9oviJjG2kTrjydkUKgEKlen7H3rMsge6v.pdf",
        "ZXjkM4OD4QvSwI1q202105BhJdxOW8u3zn8eTSUS.pdf",
        "ZxWrsKfLJo4Bgco32022077uAJaP4eb0XKDTms9C8WZFa0U1Ihh7r8ICGVw7gC.doc",
        "zYKIqKaEgaV7zDM1m5UOlshKXemMRMsHB3754TtCS9rD59mvfRlZINOvcxTIoSmz.pdf",
        "ZYqmbqLEyhMldK9EUS.doc",
        "zyk0762VlHrzyWzTcf9aE2gcbKBLdees9izWVvyZwVmcSoBCG2H3WS26YeHz6oI1.pdf",
        "ZYsIIALrrM9wlGJF202207Cg0w7hfLxkhatzNlXriGO9xZYSIYQB5jQAWUljx5StbIkCagnxmAUvBotCcILsxf.pdf",
        "zyMvnuqLrJajpH8B2022066OzOqQ5B8PYkquwwPZUKuMKHyi14hR9llJZWX84RzhkBDfVKPVtptlNwEox3i9k5.pdf",
        "zySgOncQr3gYIdZS202206w7W2NGDHCf8oCruavArv7MNSTPkstQ6EaMaGwQ7KYBlj3itDtsXmuKOe2rgiwc4C.pdf",
        "zyQfZEQCyAfg7uRkXo7gWQ14pNeEKtYASHyu5TNoYyBVe2odeH29pEfpoy0XHcoq.pdf",
        "ZzL77TogK6CasKyGOaVe7yXtSl3kSq300V42PpFy.doc",
        "zzaVi8Rn0BWFpcua202206eGJsnmqv2SxWk4n4JYAGNlhz04aW7QY22ZElCBoMsi2a6oVDkmD2BiPlAGBEtl63.pdf",
        "zzMR1oFcnJNohHKt0LkVtTq8tl8zWrQYVCiI2IRJ.doc",
        "zZiType37mzUBSCP202206eIkq8MFMAo8LPgoqNvHXDSlWGTyYRfhL7JkL5dFSrZfbDGpQcWZsdlYw4dSzQ0NX.pdf",
        "zzs7CajVXe1c6pDwaFIp016IdfzPA5vXGjstwbc0ldMJgz5qAs1bNffiZAZbEM5x.pdf",
        "ZZtjfHiFFj99f28UlqCEy9oXoGxGMZzgkfS5LhLbzDIRCw18UZgLUIfNal7bcUql.pdf",
        "zztJ9zUUIPCrvitxMz27qXhQLB6vEXN74PqkRdPfiHJ6sCtMmH2u6ANvciGHTjvU.pdf"
    ]
    # Process only the files in the list
    processor.process_specific_files(specific_files)

    