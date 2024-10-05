import streamlit as st
import sys
import os
from typing import Dict

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_pipeline import RAGPipeline
from src.singleton_config import ConfigSingleton

def initialize_pipeline():
    config = ConfigSingleton()
    return RAGPipeline(config.to_dict())

def execute_query(pipeline: RAGPipeline, query: str) -> Dict[str, str]:
    return pipeline.query(query)

def main():
    st.title("RAG System Query Interface")

    # Initialize the pipeline
    pipeline = initialize_pipeline()

    # Create a text input for the user's query
    user_query = st.text_input("Enter your question:")

    if user_query:
        # Execute the query and get the result
        with st.spinner("Generating response..."):
            result = execute_query(pipeline, user_query)
        
        # Display the response
        st.subheader("Response:")
        st.write(result['response'])

        # Optionally display search results
        if st.checkbox("Show search results"):
            st.subheader("Search Results:")
            for i, chunk in enumerate(result['search_results']):
                st.write(f"Chunk {i+1}:")
                st.write(chunk['chunk_text'])
                st.write("---")

if __name__ == "__main__":
    main()
