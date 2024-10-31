import sys
import streamlit as st
import sqlite3
import faiss
import os 
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from paths import get_db_path, get_faiss_path
from src.singleton_config import ConfigSingleton

class DocumentViewer:
    def __init__(self):
        print("Initializing DocumentViewer...")
        self.config = ConfigSingleton()
        self.project_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.db_path = os.path.join(self.project_folder, 'data', 'metadata.db')
        self.faiss_path = os.path.join(self.project_folder, 'data', 'faiss_index.bin')
        
        print(f"DB Path: {self.db_path}")
        print(f"FAISS Path: {self.faiss_path}")
        
        # Check if files exist
        print(f"DB exists: {os.path.exists(self.db_path)}")
        print(f"FAISS exists: {os.path.exists(self.faiss_path)}")
        
        self.conn = sqlite3.connect(self.db_path)
        self.index = self._load_faiss_index()

    def _load_faiss_index(self) -> Optional[faiss.Index]:
        """Load FAISS index with error handling."""
        try:
            if os.path.exists(self.faiss_path):
                index = faiss.read_index(self.faiss_path)
                print(f"Loaded FAISS index with {index.ntotal} vectors")
                return index
            else:
                print("No FAISS index found")
                return None
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            return None

    def get_documents(self) -> pd.DataFrame:
        """Get list of documents from SQLite."""
        try:
            query = "SELECT document_id, title, author, date_added FROM documents_metadata"
            df = pd.read_sql_query(query, self.conn)
            print(f"Found {len(df)} documents in database")
            return df
        except Exception as e:
            print(f"Error getting documents: {e}")
            return pd.DataFrame()

    def get_document_chunks(self, document_id: str) -> pd.DataFrame:
        """Get chunks for a specific document."""
        try:
            query = """
            SELECT 
                c.chunk_id,
                c.document_id,
                c.chunk_text,
                c.start_index,
                c.end_index,
                c.chunk_length,
                c.date_added
            FROM document_chunks_metadata c
            WHERE c.document_id = ?
            """
            chunks_df = pd.read_sql_query(query, self.conn, params=(document_id,))
            print(f"Found {len(chunks_df)} chunks for document {document_id}")
            
            if self.index is not None:
                print("Processing vectors...")
                vectors = []
                for chunk_id in chunks_df['chunk_id']:
                    try:
                        vector = self.index.reconstruct(int(chunk_id))
                        vectors.append(vector)
                        print(f"Got vector for chunk {chunk_id}")
                    except Exception as e:
                        print(f"Failed to get vector for chunk {chunk_id}: {e}")
                        vectors.append(None)
                
                chunks_df['vector'] = vectors
                chunks_df['vector_available'] = chunks_df['vector'].apply(lambda x: x is not None)
                chunks_df['vector_preview'] = chunks_df['vector'].apply(self.format_vector_preview)
            else:
                print("No FAISS index available")
                chunks_df['vector_available'] = False
                chunks_df['vector_preview'] = 'No FAISS index'
            
            return chunks_df
            
        except Exception as e:
            print(f"Error getting chunks: {e}")
            return pd.DataFrame()

    def format_vector_preview(self, vector: Optional[np.ndarray]) -> str:
        if vector is None:
            return "No vector available"
        preview = ", ".join(f"{x:.4f}" for x in vector[:5])
        return f"[{preview}, ...]"

    def run(self):
        """Main UI logic."""
        print("Starting DocumentViewer UI...")
        
        st.title("Document and Chunk Viewer")

        # Document selection
        documents = self.get_documents()
        if len(documents) == 0:
            st.error("No documents found in database")
            return

        selected_document = st.selectbox(
            "Select a document",
            documents['document_id'].tolist(),
            format_func=lambda x: documents[documents['document_id'] == x]['title'].iloc[0]
        )

        if selected_document:
            # Display document info
            doc_row = documents[documents['document_id'] == selected_document].iloc[0]
            st.write("## Document Information")
            st.write(f"Title: {doc_row['title']}")
            st.write(f"Author: {doc_row['author']}")
            st.write(f"Added: {doc_row['date_added']}")
            
            # Get and display chunks
            chunks = self.get_document_chunks(selected_document)
            st.write(f"## Chunks ({len(chunks)} total)")
            
            if len(chunks) > 0:
                st.dataframe(chunks[[
                    'chunk_id', 
                    'chunk_length', 
                    'vector_available',
                    'vector_preview'
                ]])
            else:
                st.warning("No chunks found for this document")

def main():
    print("Starting application...")
    viewer = DocumentViewer()
    viewer.run()

if __name__ == "__main__":
    main()