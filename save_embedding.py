import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from collections import OrderedDict
import logging
import glob
import os
import csv
from datetime import datetime
import numpy as np
import zlib
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str):
        if key not in self.cache:
            return None
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: str, value: Any):
        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = value
        self.cache.move_to_end(key)

model_cache = LRUCache(capacity=1)

class EmbeddingManager:
    def __init__(self, embedding_model_name: str, output_folder: str):
        self.embedding_model = self.load_embedding_model(embedding_model_name)
        self.output_folder = output_folder
        self.embeddings_folder = os.path.join(self.output_folder, "embeddings")
        os.makedirs(self.embeddings_folder, exist_ok=True)
        self.tracking_folder = os.path.join(self.output_folder, "embedding_tracking")
        os.makedirs(self.tracking_folder, exist_ok=True)

    def load_embedding_model(self, model_name: str) -> SentenceTransformer:
        model = model_cache.get(model_name)
        if model is None:
            logger.info(f"Loading embedding model: {model_name}")
            model = SentenceTransformer(model_name, trust_remote_code=True)
            model_cache.put(model_name, model)
        return model

    def process_and_save_embeddings(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        json_filename = os.path.basename(file_path)
        tracking_file = os.path.join(self.tracking_folder, f"{json_filename}.csv")
        
        # Check if the tracking file exists and load processed documents
        processed_documents = set()
        if os.path.exists(tracking_file):
            with open(tracking_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                processed_documents = set(row[0] for row in reader if row[1] == 'success')

        # Process the JSON file
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)

            # Create a tqdm progress bar
            pbar = tqdm(total=len(data), desc=f"Processing {json_filename}", unit="document")

            for document_name, value in data.items():
                if document_name in processed_documents or ".html" in document_name:
                    logger.info(f"Skipping already processed document: {document_name} or html documenthtop")
                    pbar.update(1)
                    continue
                if not value["is_resume"]:
                    logger.info(f"Skipping non resume document: {document_name}")
                    pbar.update(1)
                    continue

                try:
                    state_of_the_union = value["content"]
                    logger.info(f"Processing Document: {document_name}")

                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        length_function=len,
                        is_separator_regex=False,
                    )

                    texts = text_splitter.create_documents([state_of_the_union])
                    embeddings = self.generate_and_save_embeddings([doc.page_content for doc in texts], document_name)
                    
                    self.update_tracking_file(tracking_file, document_name, 'success')
                    logger.info(f"Successfully processed and saved embeddings for: {document_name}")
                except Exception as doc_ex:
                    logger.error(f"Error processing document {document_name}: {doc_ex}")
                    self.update_tracking_file(tracking_file, document_name, 'fail', str(doc_ex))
                finally:
                    pbar.update(1)

            pbar.close()
            logger.info(f"Completed processing file: {json_filename}")
        except Exception as file_ex:
            logger.error(f"Error processing file {json_filename}: {file_ex}")

    def generate_and_save_embeddings(self, texts: List[str], document_name: str) -> List[bytes]:
        embeddings = []
        document_file = os.path.join(self.embeddings_folder, f"{document_name}.npz")

        all_vectors = []
        for text in texts:
            text_id = str(uuid.uuid4())
            vector = self.embedding_model.encode(text).tolist()
            compressed_vector = zlib.compress(np.array(vector, dtype=np.float32).tobytes())
            all_vectors.append(compressed_vector)
            embeddings.append(compressed_vector)

        np.savez_compressed(document_file, vectors=all_vectors)
        logger.debug(f"Saved embeddings for document {document_name} in {document_file}")

        return embeddings

    def update_tracking_file(self, tracking_file: str, document_name: str, status: str, error_message: str = ''):
        file_exists = os.path.exists(tracking_file)
        with open(tracking_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['document_name', 'status', 'error_message', 'timestamp'])
            writer.writerow([document_name, status, error_message, datetime.now().isoformat()])

def bulk_process_json(folder_path: str, embedding_manager: EmbeddingManager, chunk_size: int = 1000, chunk_overlap: int = 200):
    json_path_list = glob.glob(os.path.join(folder_path, "*.json"))

    # Create a tqdm progress bar for JSON files
    file_pbar = tqdm(total=len(json_path_list), desc="Processing JSON files", unit="file")

    for file_path in json_path_list:
        embedding_manager.process_and_save_embeddings(file_path, chunk_size, chunk_overlap)
        file_pbar.update(1)

    file_pbar.close()

if __name__ == "__main__":
    embedding_manager = EmbeddingManager(
        embedding_model_name="mixedbread-ai/mxbai-embed-large-v1",
        output_folder="/root/data/ProcessedResults"
    )

    folder_path = "/root/data/ProcessedResults/json/"
    bulk_process_json(folder_path, embedding_manager)