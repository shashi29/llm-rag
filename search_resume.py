import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Any
from collections import OrderedDict
import os
import json
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str) -> Any:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: Any) -> None:
        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = value
        self.cache.move_to_end(key)

class QdrantSearchApp:
    def __init__(self, qdrant_url: str, embedding_model_name: str, collection_name: str):
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        self.model_cache = LRUCache(capacity=1)
        self.embedding_model = self.load_embedding_model()

    def load_embedding_model(self) -> SentenceTransformer:
        model = self.model_cache.get(self.embedding_model_name)
        if model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            model = SentenceTransformer(self.embedding_model_name, trust_remote_code=True)
            self.model_cache.put(self.embedding_model_name, model)
        return model

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        try:
            query_vector = self.embedding_model.encode(query).tolist()
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            return [
                {
                    "text_id": hit.payload["text_id"],
                    "text": hit.payload["text"],
                    "document_name": hit.payload["document_name"],
                    "score": hit.score
                }
                for hit in search_result
            ]
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise

class StreamlitUI:
    def __init__(self, search_app: QdrantSearchApp):
        self.search_app = search_app

    def render(self):
        st.title("Qdrant Search Application")
        st.write("Search through your document chunks stored in Qdrant.")

        query = st.text_input("Enter your search query:")
        if st.button("Search"):
            self.handle_search(query)

    def handle_search(self, query: str):
        if not query.strip():
            st.warning("Please enter a query to search.")
            return

        try:
            with st.spinner("Searching..."):
                results = self.search_app.search(query)
            self.display_results(results)
        except Exception as e:
            st.error(f"An error occurred while searching: {str(e)}")

    def display_results(self, results: List[Dict[str, Any]]):
        if not results:
            st.info("No results found for your query.")
            return
            
        for i, result in enumerate(results, 1):
            print(f"**Document Name:** {result['document_name']}")
            json_folder = "/root/ProcessedResults/json"
            document_contents = read_document_from_json(json_folder, result['document_name'])
            st.markdown(f"**Result {i}:**")
            st.write(f"**Text:** {result['text']}")
            st.write(f"**Document Name:** {result['document_name']}")
            st.write(f"**Score:** {result['score']:.4f}")
            st.write(f"**Resume** {document_contents}")
            st.write("---")

def read_document_from_json(json_folder: str, target_key: str):
    """
    Reads JSON files from the specified folder and looks for the specified key.
    
    Parameters:
        json_folder (str): The folder containing JSON files.
        target_key (str): The key to search for in the JSON files.

    Returns:
        dict: A dictionary where keys are filenames and values are the document content.
    """
    document_contents = ""
    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]

    # Progress bar to track reading progress
    for json_file in tqdm(json_files, desc="Reading JSON files", unit="file"):
        file_path = os.path.join(json_folder, json_file)
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                
                # Check if the target key exists in the JSON data
                if target_key in data:
                    document_contents = data[target_key]["content"]
                    return document_contents
                else:
                    print(f"Key '{target_key}' not found in {json_file}")
        
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in file {json_file}: {e}")
        except Exception as e:
            print(f"Error reading file {json_file}: {e}")

    return document_contents

def main():
    # Configuration
    qdrant_url = "http://localhost:6333"  # Replace with your Qdrant server URL
    embedding_model_name = "nomic-ai/nomic-embed-text-v1"
    collection_name = "new_collection"  # Replace with your actual collection name

    # Initialize the application
    search_app = QdrantSearchApp(qdrant_url, embedding_model_name, collection_name)
    ui = StreamlitUI(search_app)

    # Run the Streamlit UI
    ui.render()

if __name__ == "__main__":
    main()