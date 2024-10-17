import sys
import streamlit as st
import sqlite3
import faiss
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np      

from paths import get_db_path, get_faiss_path
from src.singleton_config import ConfigSingleton

# Initialize configuration
config = ConfigSingleton()

# Set up database and FAISS index connections
project_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
db_path = os.path.join(project_folder, 'data', 'metadata.db')
faiss_path = os.path.join(project_folder, 'data', 'faiss_index.bin')

conn = sqlite3.connect(db_path)
index = faiss.read_index(faiss_path)

def get_documents():
    query = "SELECT document_id, title FROM documents_metadata"
    return pd.read_sql_query(query, conn)

def get_document_chunks(document_id):
    query = """
    SELECT c.chunk_id, c.document_id, c.chunk_text, c.start_index, c.end_index, c.chunk_length
    FROM document_chunks_metadata c
    WHERE c.document_id = ?
    """
    return pd.read_sql_query(query, conn, params=(document_id,))

def get_faiss_vector(chunk_id):
    try:
        # Assuming chunk_id is used as the FAISS ID
        vector = index.reconstruct(int(chunk_id))
        return vector
    except RuntimeError:
        return None

st.title("Document and Chunk Viewer")

# Fetch and display document dropdown
documents = get_documents()
selected_document = st.selectbox(
    "Select a document",
    documents['document_id'],
    format_func=lambda x: documents[documents['document_id'] == x]['title'].iloc[0]
)

if selected_document:
    st.write(f"Displaying chunks for document: {documents[documents['document_id'] == selected_document]['title'].iloc[0]}")
    
    # Fetch chunks for the selected document
    chunks = get_document_chunks(selected_document)
    
    # Add FAISS vector information to the chunks dataframe
    chunks['vector'] = chunks['chunk_id'].apply(get_faiss_vector)
    chunks['vector_available'] = chunks['vector'].apply(lambda x: "Yes" if x is not None else "No")
    chunks['vector_preview'] = chunks['vector'].apply(lambda x: str(x[:5]) + "..." if x is not None else "N/A")
    
    # Display the chunks table
    st.dataframe(chunks[[
        'chunk_id', 'document_id', 'start_index', 'end_index', 'chunk_length',
        'vector_available', 'vector_preview'
    ]])
    
    # Option to view full chunk text
    if st.checkbox("Show full chunk text"):
        chunk_to_view = st.selectbox("Select a chunk to view", chunks['chunk_id'])
        st.text_area("Chunk Text", chunks[chunks['chunk_id'] == chunk_to_view]['chunk_text'].iloc[0], height=200)

conn.close()