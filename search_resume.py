import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Any
from collections import OrderedDict
import os
import pandas as pd
from resume_job_matching import ResumeJobMatchingService

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
        #self.model_cache = LRUCache(capacity=1)
        self.embedding_model = self.load_embedding_model()

    def load_embedding_model(self) -> SentenceTransformer:
        model = None#self.model_cache.get(self.embedding_model_name)
        if model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            model = SentenceTransformer(self.embedding_model_name, trust_remote_code=True)
            #self.model_cache.put(self.embedding_model_name, model)
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
    def __init__(self, search_app: QdrantSearchApp, resume_folder: str):
        self.search_app = search_app
        self.resume_folder = resume_folder

    def render(self):
        st.title("Qdrant Search Application")
        st.write("Search through your document chunks stored in Qdrant.")

        # Add a button to clear the cache
        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared successfully!")

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
            self.display_results(query, results)
        except Exception as e:
            st.error(f"An error occurred while searching: {str(e)}")

    def display_results(self, query: str, results: List[Dict[str, Any]]):
        if not results:
            st.info("No results found for your query.")
            return
            
        resume_service = ResumeJobMatchingService()
        for i, result in enumerate(results, 1):
            st.markdown(f"**Result {i}:**")
            st.write(f"**Document Name:** {result['document_name']}")

            # Add download link for the resume
            resume_path = os.path.join(self.resume_folder, result['document_name'])
            if os.path.exists(resume_path):
                self.get_download_link(resume_path, result['document_name'], f"download_button_{i}")
            else:
                st.warning(f"Resume file not found: {result['document_name']}")

            # Assume we have the resume content in result['text'] for analysis
            analysis = resume_service.generate_match_analysis(query, result['text'])
            
            # Display analysis results
            self.display_analysis(analysis)

            st.write("---")

    def get_download_link(self, file_path: str, file_name: str, key: str):
        """Create a download link for the resume file."""
        with open(file_path, "rb") as file:
            st.download_button(
                label="Download Resume",
                data=file,
                file_name=file_name,
                mime="application/octet-stream",
                key=key
            )

    def display_analysis(self, analysis: Dict[str, Any]):
        # Overall Match Score
        st.markdown(f"### **Overall Match Score:** {analysis['Overall_Match_Score']}")

        # Skill Match Breakdown (Table format)
        st.markdown("### **Skill Match Breakdown**")
        skills_data = {
            "Skill Type": ["Technical Skills", "Soft Skills", "Certifications"],
            "Match": [
                analysis['Skill_Match_Breakdown']['Technical_Skills'],
                analysis['Skill_Match_Breakdown']['Soft_Skills'],
                analysis['Skill_Match_Breakdown']['Certifications'],
            ]
        }
        skills_df = pd.DataFrame(skills_data)
        st.table(skills_df)

        # Experience Relevance
        st.markdown("### **Experience Relevance**")
        experience_relevance = analysis['Experience_Relevance']
        st.markdown(f"- **Years of Relevant Experience:** {experience_relevance['Years_of_Relevant_Experience']}")
        st.markdown(f"- **Experience Quality:** {experience_relevance['Experience_Quality']}")

        # Project Alignment (Table format)
        st.markdown("### **Project Alignment**")
        project_data = [
            {
                "Project Name": project['Project_Name'],
                "Description": project['Description'],
                "Relevance Score": project['Relevance_Score']
            }
            for project in analysis['Project_Alignment']
        ]
        project_df = pd.DataFrame(project_data)
        st.table(project_df)

        # Key Strengths and Notable Gaps
        st.markdown("### **Key Strengths**")
        for strength in analysis['Key_Strengths']:
            st.markdown(f"- {strength}")

        st.markdown("### **Notable Gaps**")
        for gap in analysis['Notable_Gaps']:
            st.markdown(f"- {gap}")

        # Overall Assessment
        st.markdown("### **Overall Assessment**")
        st.write(analysis['Overall_Assessment'])

        # Recommendations (Next Steps and Skill Enhancement)
        st.markdown("### **Recommendations**")
        st.markdown(f"- **Next Steps:** {analysis['Recommendations']}")
        #st.markdown(f"- **Skill Enhancement:** {analysis['Recommendations']['Areas_for_Development']}")

def main():
    # Configuration
    qdrant_url = "http://localhost:6333"  # Replace with your Qdrant server URL
    embedding_model_name = "mixedbread-ai/mxbai-embed-large-v1"
    collection_name = "new_collection"  # Replace with your actual collection name
    resume_folder = "/root/Batch1"  # Replace with the actual path to your resume documents

    # Initialize the application
    search_app = QdrantSearchApp(qdrant_url, embedding_model_name, collection_name)
    ui = StreamlitUI(search_app, resume_folder)

    # Run the Streamlit UI
    ui.render()

if __name__ == "__main__":
    main()