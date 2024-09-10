# import json
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# import uuid
# from typing import List, Dict, Any, Optional
# from qdrant_client import QdrantClient
# from qdrant_client.http.models import UpdateStatus
# from qdrant_client.models import PointStruct, VectorParams, Distance, SearchRequest
# from sentence_transformers import SentenceTransformer
# from collections import OrderedDict
# import logging
# import glob

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class LRUCache:
#     def __init__(self, capacity: int):
#         self.cache = OrderedDict()
#         self.capacity = capacity

#     def get(self, key: str):
#         if key not in self.cache:
#             return None
#         else:
#             self.cache.move_to_end(key)
#             return self.cache[key]

#     def put(self, key: str, value: Any):
#         if len(self.cache) >= self.capacity:
#             self.cache.popitem(last=False)
#         self.cache[key] = value
#         self.cache.move_to_end(key)

# model_cache = LRUCache(capacity=1)

# class QdrantOperations:
#     def __init__(self, qdrant_url: str, embedding_model_name: str):
#         self.qdrant_client = QdrantClient(url=qdrant_url)
#         self.embedding_model = self.load_embedding_model(embedding_model_name)
        

#     def load_embedding_model(self, model_name: str) -> SentenceTransformer:
#         model = model_cache.get(model_name)
#         if model is None:
#             logger.info(f"Loading embedding model: {model_name}")
#             model = SentenceTransformer(model_name, trust_remote_code=True)
#             model_cache.put(model_name, model)
#         return model

#     def create_collection(self, collection_name):
#         try:
#             self.qdrant_client.create_collection(
#                 collection_name=collection_name,
#                 vectors_config=VectorParams(size=self.embedding_model.get_sentence_embedding_dimension(), distance=Distance.COSINE),
#             )
#             logger.info(f"Collection '{collection_name}' created successfully.")
#         except Exception as e:
#             logger.error(f"Error creating collection: {e}")
#             raise

#     def delete_collection(self, collection_name):
#         try:
#             self.qdrant_client.delete_collection(collection_name=collection_name)
#             logger.info(f"Collection '{self.collection_name}' deleted successfully.")
#         except Exception as e:
#             logger.error(f"Error deleting collection: {e}")
#             raise

#     def check_collection_exists(self, collection_name) -> bool:
#         try:
#             collections = self.qdrant_client.get_collections()
#             return any(collection.name == collection_name for collection in collections.collections)
#         except Exception as e:
#             logger.error(f"Error checking collection existence: {e}")
#             raise

#     def get_all_collections(self) -> List[str]:
#         try:
#             collections = self.qdrant_client.get_collections()
#             return [collection.name for collection in collections.collections]
#         except Exception as e:
#             logger.error(f"Error getting all collections: {e}")
#             raise

#     def insert_points(self, texts: List[str],  document_name:str, collection_name: str) -> List[str]:
#         points = []
#         text_ids = []
#         for text in texts:
#             text_id = str(uuid.uuid4())
#             vector = self.embedding_model.encode(text).tolist()
#             payload = {
#                 "text_id": text_id,
#                 "text": text,
#                 "document_name":document_name
#             }
#             point = PointStruct(id=text_id, vector=vector, payload=payload)
#             points.append(point)
#             text_ids.append(text_id)

#         try:
#             operation_info = self.qdrant_client.upsert(
#                 collection_name=collection_name,
#                 wait=True,
#                 points=points
#             )
#             if operation_info.status != UpdateStatus.COMPLETED:
#                 raise Exception("Failed to insert data")
#             logger.info(f"Successfully inserted {len(points)} points.")
#             return text_ids
#         except Exception as e:
#             logger.error(f"Error inserting data: {e}")
#             raise

#     def search(self, query: str, collection_name: str, limit: int = 5) -> List[Dict[str, Any]]:
#         try:
#             query_vector = self.embedding_model.encode(query).tolist()
#             search_result = self.qdrant_client.search(
#                 collection_name=collection_name,
#                 query_vector=query_vector,
#                 limit=limit
#             )
#             return [{"text_id": hit.payload["text_id"], "text": hit.payload["text"],"document_name":hit.payload["document_name"], "score": hit.score} for hit in search_result]
#         except Exception as e:
#             logger.error(f"Error searching: {e}")
#             raise

#     def delete_points(self, text_ids: List[str], collection_name:str):
#         try:
#             self.qdrant_client.delete(
#                 collection_name=collection_name,
#                 points_selector=text_ids
#             )
#             logger.info(f"Successfully deleted {len(text_ids)} points.")
#         except Exception as e:
#             logger.error(f"Error deleting points: {e}")
#             raise

#     def process_file(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
#         with open(file_path, 'r') as file:
#             data = json.load(file)
        
#         # Assuming the structure of your JSON file, adjust as necessary
#         content = next(iter(data.values()))["content"]
        
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap,
#             length_function=len,
#             is_separator_regex=False,
#         )
        
#         texts = text_splitter.create_documents([content])
#         return [doc.page_content for doc in texts]
    
#     def bulk_quadrant_insert(self, folder_path:str, collection_name: str, chunk_size: int = 1000, chunk_overlap: int = 200):
#         json_path_list = glob.glob(f"{folder_path}*.json")

#         for file_path in json_path_list:
#             try:
#                 with open(file_path, 'r') as file:
#                     data = json.load(file)
#                     for document_name, value in data.items():
#                         state_of_the_union = data[document_name]["content"]     
#                         logger.info(f"Inserting Document:{document_name}")

#                         text_splitter = RecursiveCharacterTextSplitter(
#                             chunk_size=1000,
#                             chunk_overlap=200,
#                             length_function=len,
#                             is_separator_regex=False,
#                         )

#                         texts = text_splitter.create_documents([state_of_the_union])
#                         inserted_ids = self.insert_points([doc.page_content for doc in texts], document_name, collection_name)
#             except Exception as ex:
#                 logging.info("Issue while bulk insert")

# # Usage example
# if __name__ == "__main__":
#     qdrant_ops = QdrantOperations(
#         qdrant_url="http://localhost:6333",
#         embedding_model_name="nomic-ai/nomic-embed-text-v1",
#     )
    
#     collection_name="new_collection"
#     folder_path="/root/ProcessedResults/json/"
    
#     # Create collection if it doesn't exist
#     if not qdrant_ops.check_collection_exists(collection_name):
#         qdrant_ops.create_collection(collection_name)

#     # Process file and insert points
#     #file_path = "/root/ProcessedResults/json/document_processing_results_20240908_015541_100.json"
#     #texts = qdrant_ops.process_file(file_path, )
#     #inserted_ids = qdrant_ops.insert_points(texts)

#     #qdrant_ops.bulk_quadrant_insert(folder_path, collection_name)

#     # Perform a search
#     search_results = qdrant_ops.search("Show all the full stack developer", collection_name)
#     print("Search results:", search_results)

#     # Delete some points
#     # qdrant_ops.delete_points(inserted_ids[:5])

#     # Get all collections
#     # all_collections = qdrant_ops.get_all_collections()
#     # print("All collections:", all_collections)

#     # Delete the collection (uncomment if needed)
#     # qdrant_ops.delete_collection()


import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import UpdateStatus
from qdrant_client.models import PointStruct, VectorParams, Distance, SearchRequest
from sentence_transformers import SentenceTransformer
from collections import OrderedDict
import logging
import glob
import os
import csv
from datetime import datetime
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

class QdrantOperations:
    def __init__(self, qdrant_url: str, embedding_model_name: str, output_folder: str):
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.embedding_model = self.load_embedding_model(embedding_model_name)
        self.output_folder = output_folder
        self.tracking_folder = os.path.join(output_folder, "qdrant_tracking")
        os.makedirs(self.tracking_folder, exist_ok=True)

    def load_embedding_model(self, model_name: str) -> SentenceTransformer:
        model = model_cache.get(model_name)
        if model is None:
            logger.info(f"Loading embedding model: {model_name}")
            model = SentenceTransformer(model_name, trust_remote_code=True)
            model_cache.put(model_name, model)
        return model

    def create_collection(self, collection_name):
        try:
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.embedding_model.get_sentence_embedding_dimension(), distance=Distance.COSINE),
            )
            logger.info(f"Collection '{collection_name}' created successfully.")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise

    def delete_collection(self, collection_name):
        try:
            self.qdrant_client.delete_collection(collection_name=collection_name)
            logger.info(f"Collection '{collection_name}' deleted successfully.")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise

    def check_collection_exists(self, collection_name) -> bool:
        try:
            collections = self.qdrant_client.get_collections()
            return any(collection.name == collection_name for collection in collections.collections)
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            raise

    def get_all_collections(self) -> List[str]:
        try:
            collections = self.qdrant_client.get_collections()
            return [collection.name for collection in collections.collections]
        except Exception as e:
            logger.error(f"Error getting all collections: {e}")
            raise

    def insert_points(self, texts: List[str], document_name: str, collection_name: str) -> List[str]:
        points = []
        text_ids = []
        for text in texts:
            text_id = str(uuid.uuid4())
            vector = self.embedding_model.encode(text).tolist()
            payload = {
                "text_id": text_id,
                "text": text,
                "document_name": document_name
            }
            point = PointStruct(id=text_id, vector=vector, payload=payload)
            points.append(point)
            text_ids.append(text_id)

        try:
            operation_info = self.qdrant_client.upsert(
                collection_name=collection_name,
                wait=True,
                points=points
            )
            if operation_info.status != UpdateStatus.COMPLETED:
                raise Exception("Failed to insert data")
            logger.info(f"Successfully inserted {len(points)} points.")
            return text_ids
        except Exception as e:
            logger.error(f"Error inserting data: {e}")
            raise

    def search(self, query: str, collection_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        try:
            query_vector = self.embedding_model.encode(query).tolist()
            search_result = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit
            )
            return [{"text_id": hit.payload["text_id"], "text": hit.payload["text"], "document_name": hit.payload["document_name"], "score": hit.score} for hit in search_result]
        except Exception as e:
            logger.error(f"Error searching: {e}")
            raise

    def delete_points(self, text_ids: List[str], collection_name: str):
        try:
            self.qdrant_client.delete(
                collection_name=collection_name,
                points_selector=text_ids
            )
            logger.info(f"Successfully deleted {len(text_ids)} points.")
        except Exception as e:
            logger.error(f"Error deleting points: {e}")
            raise

    def process_and_insert_json(self, file_path: str, collection_name: str, chunk_size: int = 1000, chunk_overlap: int = 200):
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
                if document_name in processed_documents:
                    logger.info(f"Skipping already processed document: {document_name}")
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
                    inserted_ids = self.insert_points([doc.page_content for doc in texts], document_name, collection_name)
                    
                    self.update_tracking_file(tracking_file, document_name, 'success')
                    logger.info(f"Successfully processed and inserted: {document_name}")
                except Exception as doc_ex:
                    logger.error(f"Error processing document {document_name}: {doc_ex}")
                    self.update_tracking_file(tracking_file, document_name, 'fail', str(doc_ex))
                finally:
                    pbar.update(1)

            pbar.close()
            logger.info(f"Completed processing file: {json_filename}")
        except Exception as file_ex:
            logger.error(f"Error processing file {json_filename}: {file_ex}")

    def update_tracking_file(self, tracking_file: str, document_name: str, status: str, error_message: str = ''):
        file_exists = os.path.exists(tracking_file)
        with open(tracking_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['document_name', 'status', 'error_message', 'timestamp'])
            writer.writerow([document_name, status, error_message, datetime.now().isoformat()])

    def bulk_qdrant_insert(self, folder_path: str, collection_name: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        json_path_list = glob.glob(os.path.join(folder_path, "*.json"))

        # Create a tqdm progress bar for JSON files
        file_pbar = tqdm(total=len(json_path_list), desc="Processing JSON files", unit="file")

        for file_path in json_path_list:
            self.process_and_insert_json(file_path, collection_name, chunk_size, chunk_overlap)
            file_pbar.update(1)

        file_pbar.close()

# Usage example
if __name__ == "__main__":
    qdrant_ops = QdrantOperations(
        qdrant_url="http://localhost:6333",
        embedding_model_name="nomic-ai/nomic-embed-text-v1",
        output_folder="/root/ProcessedResults"
    )
    
    collection_name = "new_collection"
    folder_path = "/root/ProcessedResults/json/"
    
    # Create collection if it doesn't exist
    if not qdrant_ops.check_collection_exists(collection_name):
        qdrant_ops.create_collection(collection_name)

    # Process all JSON files in the folder
    qdrant_ops.bulk_qdrant_insert(folder_path, collection_name)

    # Perform a search (example)
    # search_results = qdrant_ops.search("Show all the full stack developer", collection_name)
    # print("Search results:", search_results)

    # Uncomment the following lines if you want to delete the collection
    # qdrant_ops.delete_collection(collection_name)