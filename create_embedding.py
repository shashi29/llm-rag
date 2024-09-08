# import json
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# import uuid
# from typing import List, Dict, Any
# from qdrant_client import QdrantClient
# from qdrant_client.http.models import UpdateStatus
# from qdrant_client.models import PointStruct, VectorParams, Distance
# from sentence_transformers import SentenceTransformer
# from collections import OrderedDict
# import logging

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # LRU Cache class
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

# # Initialize LRU cache
# model_cache = LRUCache(capacity=1)  # Adjust capacity as needed

# file_path = "/root/ProcessedResults/json/document_processing_results_20240908_015541_100.json"

# with open(file_path, 'r') as file:
#     data = json.load(file)
    
# state_of_the_union = data["ZPRZK1tSWArHtpCo202205E4ZgaqgRHOVvGcaIQ1WHwJ6Ua14ykBLR7ZUsk1yy2VeR3yLsycc2RQ0KPAV29QXV.pdf"]["content"]


# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200,
#     length_function=len,
#     is_separator_regex=False,
# )

# texts = text_splitter.create_documents([state_of_the_union])



# def load_embedding_model(model_name: str) -> SentenceTransformer:
#     model = model_cache.get(model_name)
#     if model is None:
#         logger.info(f"Loading embedding model: {model_name}")
#         model = SentenceTransformer(model_name, trust_remote_code=True)
#         model_cache.put(model_name, model)
#     return model

# embedding_model = load_embedding_model("nomic-ai/nomic-embed-text-v1")
# qdrant_client = QdrantClient(url="http://localhost:6333")
# collection_name = "text_collection1"

# from qdrant_client.models import Distance, VectorParams

# qdrant_client.create_collection(
#     collection_name=collection_name,
#     vectors_config=VectorParams(size=embedding_model.get_sentence_embedding_dimension(), distance=Distance.COSINE),
# )

# points = []
# for index in range(0,len(texts)):
#     text_id = str(uuid.uuid4())
#     vector = embedding_model.encode(texts[index].page_content).tolist()
#     payload = {
#         "text_id": text_id,
#         "text": texts[index].page_content,
#     }
#     point = PointStruct(id=text_id, vector=vector, payload=payload)
#     points.append(point)

#     try:
#         operation_info = qdrant_client.upsert(
#             collection_name=collection_name,
#             wait=True,
#             points=points
#         )
#         if operation_info.status != UpdateStatus.COMPLETED:
#             raise Exception("Failed to insert data")
#     except Exception as e:
#         logger.error(f"Error inserting data: {e}")
#         raise


import glob
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from collections import OrderedDict
import logging
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LRU Cache class
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

# Initialize LRU cache
model_cache = LRUCache(capacity=1)  # Adjust capacity as needed

def load_embedding_model(model_name: str) -> SentenceTransformer:
    model = model_cache.get(model_name)
    if model is None:
        logger.info(f"Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name, trust_remote_code=True)
        model_cache.put(model_name, model)
    return model


embedding_model = load_embedding_model("nomic-ai/nomic-embed-text-v1")
json_path_list = glob.glob(f"/root/ProcessedResults/json/*.json")

for file_path in json_path_list:
    with open(file_path, 'r') as file:
        data = json.load(file)
        for key, value in data.items():
            state_of_the_union = data[key]["content"]     
        
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False,
            )

            texts = text_splitter.create_documents([state_of_the_union])
            for index in range(0,len(texts)):
                #print(texts[index].page_content)
                vector = embedding_model.encode(texts[index].page_content).tolist()