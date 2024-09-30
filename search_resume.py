import os
import json
import requests
import pandas as pd
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
import zipfile
import io

# FastAPI backend URL
BACKEND_URL = "http://localhost:8000"

def search_documents(query: str, limit: int):
    response = requests.post(f"{BACKEND_URL}/search", json={"query": query, "limit": limit})
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None

def create_zip_file(resume_file_paths):
    """Creates a zip file containing the resumes and returns a BytesIO object."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for file_path in resume_file_paths:
            if os.path.exists(file_path):
                zip_file.write(file_path, os.path.basename(file_path))
    zip_buffer.seek(0)
    return zip_buffer

def display_analysis(analysis, resume_file_path, document_name, index):
    with st.expander(f"Result {index + 1}: {document_name}", expanded=False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"### **Overall Match Score:** {analysis['Overall_Match_Score']}")

        with col2:
            # Only try to access the file if it exists
            if os.path.exists(resume_file_path):
                with open(resume_file_path, "rb") as f:
                    # Use `st.download_button` without causing a script re-run
                    st.download_button(
                        label="Download Resume",
                        data=f,
                        file_name=os.path.basename(resume_file_path),
                        mime="application/pdf",  # Adjust mime type as needed
                        key=f"download_{index}",
                        help="Download the resume file."
                    )
            else:
                st.warning("Resume file not found.")

        # Display the analysis details (rest of the content remains unchanged)
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

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### **Experience Relevance**")
            experience_relevance = analysis['Experience_Relevance']
            st.markdown(f"- **Years of Relevant Experience:** {experience_relevance['Years_of_Relevant_Experience']}")
            st.markdown(f"- **Experience Quality:** {experience_relevance['Experience_Quality']}")

        with col2:
            st.markdown("### **Key Strengths**")
            for strength in analysis['Key_Strengths']:
                st.markdown(f"- {strength}")

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

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### **Notable Gaps**")
            for gap in analysis['Notable_Gaps']:
                st.markdown(f"- {gap}")

        with col2:
            st.markdown("### **Recommendations**")
            st.markdown(f"- **Next Steps:** {analysis['Recommendations']}")

        st.markdown("### **Overall Assessment**")
        st.write(analysis['Overall_Assessment'])

def main():
    st.set_page_config(page_title="Resume Search Application", layout="wide")
    
    colored_header(
        label="Resume Search Application",
        description="Find the best candidates for your job openings",
        color_name="blue-70"
    )

    add_vertical_space(2)

    # Search section
    st.header("Search Resume Documents")
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Enter your search query:")
    with col2:
        limit = st.number_input("Number of results:", min_value=1, max_value=100, value=5)

    if st.button("Search", type="primary"):
        if query:
            with st.spinner("Searching..."):
                results = search_documents(query, limit)
            if results:
                resume_file_paths = []
                for i, result in enumerate(results):
                    resume_file_path = f"/root/data/ProcessedResults/resume/{result['document_name']}"
                    resume_file_paths.append(resume_file_path)
                    display_analysis(result['analysis'], resume_file_path, result['document_name'], i)

                # Create a zip file for all resumes
                if resume_file_paths:
                    zip_file = create_zip_file(resume_file_paths)
                    st.download_button(
                        label="Download All Resumes",
                        data=zip_file,
                        file_name="resumes.zip",
                        mime="application/zip",
                        help="Download all resumes as a zip file."
                    )
            else:
                st.info("No results found.")
        else:
            st.warning("Please enter a search query.")

    add_vertical_space(2)
    st.markdown("---")
    add_vertical_space(1)

if __name__ == "__main__":
    main()
