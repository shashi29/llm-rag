import json
import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import sqlite3
from tqdm import tqdm  # Add tqdm for progress tracking
from concurrent.futures import ThreadPoolExecutor, as_completed
from resume_job_matching import ResumeJobMatchingService
from diskcache import Cache
import zlib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize services
qdrant_client = QdrantClient(url="http://localhost:6333")
embedding_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
resume_service = ResumeJobMatchingService()

# SQLite setup for indexing
DB_NAME = 'document_index.db'
DEFAULT_FOLDER_PATH = "/root/data/ProcessedResults/json"

# Create cache directory
CACHE_DIR = "cache_directory"  # Specify your cache directory
os.makedirs(CACHE_DIR, exist_ok=True)  # Create the cache directory if it doesn't exist

# Initialize cache
cache = Cache(directory=CACHE_DIR)

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

class SearchQuery(BaseModel):
    query: str
    limit: int = 5

def get_document_content(file_path: str, document_name: str) -> str:
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            if document_name in data:
                return data[document_name]["content"]
        return None
    except (json.JSONDecodeError, IOError) as e:
        return f"Error processing file {file_path}: {e}"

def generate_analysis(query: str, resume_text: str) -> Dict[str, Any]:
    return resume_service.generate_match_analysis(query, resume_text)

def create_cache_key(query: str, limit: int) -> str:
    return f"search_query:{zlib.adler32(f'{query}|{limit}'.encode())}"

@app.post("/search")
async def search(query: SearchQuery):
    cache_key = create_cache_key(query.query, query.limit)

    # Check if result is cached
    cached_result = cache.get(cache_key)
    if cached_result:
        logger.info("Retrieved result from cache")
        return cached_result

    try:
        query_vector = embedding_model.encode(query.query).tolist()
        search_result = qdrant_client.search(
            collection_name="embedding_collection",
            query_vector=query_vector,
            limit=query.limit
        )
        
        conn = get_db_connection()
        results = []
        futures = []
        
        with ThreadPoolExecutor() as executor:
            for hit in search_result:
                document_name = hit.payload["document_name"][:-4]  # Remove .json extension
                cursor = conn.execute('SELECT file_path FROM document_index WHERE document_name = ?', (document_name,))
                row = cursor.fetchone()
                if row:
                    file_path = os.path.join(DEFAULT_FOLDER_PATH, row['file_path'])
                    resume_text = get_document_content(file_path, document_name)
                    if resume_text:
                        futures.append(executor.submit(generate_analysis, query.query, resume_text))
                        # Store additional info to results
                        results.append({
                            "document_name": document_name,
                            "score": hit.score
                        })
        
        # Gather results as they are completed
        for future, result in zip(as_completed(futures), results):
            analysis = future.result()
            result["analysis"] = analysis
            result["overall_match_score"] = analysis.get('Overall_Match_Score', 0)  # Add match score

        conn.close()
        
        # Sort results by Overall_Match_Score
        sorted_results = sorted(results, key=lambda x: x['overall_match_score'], reverse=True)

        # Cache the result
        cache.set(cache_key, sorted_results)
        
        return sorted_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/build_index")
async def build_index():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        json_files = [f for f in os.listdir(DEFAULT_FOLDER_PATH) if f.endswith('.json')]
        
        # Add tqdm progress bar for json file processing
        for json_file in tqdm(json_files, desc="Indexing JSON files"):
            file_path = os.path.join(DEFAULT_FOLDER_PATH, json_file)
            with open(file_path, 'r') as file:
                data = json.load(file)
                # Add tqdm progress bar for documents inside each JSON
                for document_name in tqdm(data.keys(), desc=f"Indexing {json_file}", leave=False):
                    cursor.execute('''
                        INSERT OR REPLACE INTO document_index (document_name, file_path)
                        VALUES (?, ?)
                    ''', (document_name, json_file))
        
        conn.commit()
        conn.close()
        return {"message": "Index built successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
