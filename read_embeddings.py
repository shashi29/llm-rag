import os
import csv
import numpy as np
import zlib
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, UpdateStatus, OptimizersConfigDiff
from tqdm import tqdm  # Import tqdm for progress tracking

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantEmbeddingInserter:
    def __init__(self, qdrant_url: str, tracking_file: str, max_workers: int = 4):
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.tracking_file = tracking_file
        self.processed_files = self.load_processed_files()
        self.max_workers = max_workers  # Max workers for parallel processing

    def load_processed_files(self):
        """
        Load processed files from the tracking CSV file.
        """
        processed_files = set()
        if os.path.exists(self.tracking_file):
            with open(self.tracking_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip the header
                for row in reader:
                    processed_files.add(row[0])
        return processed_files

    def update_tracking_file(self, file_name: str):
        """
        Update the tracking CSV file with the processed file name.
        """
        with open(self.tracking_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([file_name])

    def create_collection(self, collection_name: str, vector_size: int, shard_number: int = 4):
        """
        Create a collection in Qdrant with shard number and indexing disabled for initial bulk upload.
        """
        try:
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size, 
                    distance=Distance.COSINE,
                    on_disk=True  # Store vectors on disk
                ),
                shard_number=shard_number,  # Split data into multiple shards
                optimizers_config=OptimizersConfigDiff(indexing_threshold=0)  # Disable indexing for faster upload
            )
            logger.info(f"Collection '{collection_name}' created successfully.")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise

    def update_indexing(self, collection_name: str, indexing_threshold: int = 20000):
        """
        Update collection indexing after bulk upload to enable indexing.
        """
        try:
            self.qdrant_client.update_collection(
                collection_name=collection_name,
                optimizers_config=OptimizersConfigDiff(indexing_threshold=indexing_threshold)
            )
            logger.info(f"Indexing enabled with threshold {indexing_threshold} for collection '{collection_name}'.")
        except Exception as e:
            logger.error(f"Error updating collection indexing: {e}")
            raise

    def check_collection_exists(self, collection_name: str) -> bool:
        """
        Check if the collection already exists in Qdrant.
        """
        try:
            collections = self.qdrant_client.get_collections()
            return any(collection.name == collection_name for collection in collections.collections)
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            raise

    def insert_embeddings_from_file(self, file_path: str, collection_name: str):
        """
        Read embeddings from an .npz file and insert them into a Qdrant collection.
        :param file_path: Path to the .npz file containing the embeddings.
        :param collection_name: Name of the Qdrant collection.
        """
        try:
            document_name = os.path.basename(file_path)
            
            # Load compressed embeddings from the .npz file
            with np.load(file_path, allow_pickle=True) as data:
                compressed_vectors = data['vectors']

            # Decompress the vectors and prepare for insertion
            points = []
            for compressed_vector in compressed_vectors:
                decompressed_bytes = zlib.decompress(compressed_vector)
                vector = np.frombuffer(decompressed_bytes, dtype=np.float32).tolist()
                
                # Create a point with a unique ID and payload
                text_id = str(uuid.uuid4())
                payload = {
                    "text_id": text_id,
                    "document_name": document_name
                }
                points.append(PointStruct(id=text_id, vector=vector, payload=payload))
            
            # Insert points into the collection
            operation_info = self.qdrant_client.upsert(
                collection_name=collection_name,
                wait=True,
                points=points
            )
            
            if operation_info.status == UpdateStatus.COMPLETED:
                logger.info(f"Successfully inserted {len(points)} embeddings from '{document_name}' into collection '{collection_name}'.")
                self.update_tracking_file(document_name)  # Mark file as processed
            else:
                raise Exception("Failed to insert embeddings.")
        
        except Exception as e:
            logger.error(f"Error inserting embeddings from file '{file_path}': {e}")
            raise

    def process_all_embeddings(self, embeddings_folder: str, collection_name: str):
        """
        Process all .npz files in the embeddings folder and insert their embeddings into the Qdrant collection.
        This function uses parallel processing to speed up the insertion process.
        """
        npz_files = [f for f in os.listdir(embeddings_folder) if f.endswith('.npz')]
        
        # Filter out the files that have already been processed
        files_to_process = [f for f in npz_files if f not in self.processed_files]

        if not files_to_process:
            logger.info("No new files to process.")
            return

        logger.info(f"Found {len(files_to_process)} new files to process.")

        # Parallel processing for file uploads with tqdm for progress tracking
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.insert_embeddings_from_file, os.path.join(embeddings_folder, file_name), collection_name): file_name for file_name in files_to_process}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                file_name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing file '{file_name}': {e}")

# Usage example
if __name__ == "__main__":
    qdrant_url = "http://localhost:6333"
    collection_name = "embedding_collection"
    embeddings_folder = "/root/data/ProcessedResults/embeddings/"
    tracking_file = "/root/data/ProcessedResults/processed_files.csv"
    
    # Initialize QdrantEmbeddingInserter
    qdrant_inserter = QdrantEmbeddingInserter(qdrant_url=qdrant_url, tracking_file=tracking_file)

    # Check if collection exists, otherwise create it with shard number and indexing disabled
    embedding_dim = 1024  # Replace this with the actual embedding dimension
    if not qdrant_inserter.check_collection_exists(collection_name):
        qdrant_inserter.create_collection(collection_name, vector_size=embedding_dim)

    # Process all embeddings from the folder and insert into the collection
    qdrant_inserter.process_all_embeddings(embeddings_folder, collection_name)

    # After upload, re-enable indexing for the collection
    qdrant_inserter.update_indexing(collection_name, indexing_threshold=20000)
