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
        self.config = ConfigSingleton()
        self.project_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.db_path = os.path.join(self.project_folder, 'data', 'metadata.db')
        self.faiss_path = os.path.join(self.project_folder, 'data', 'faiss_index.bin')
        self.conn = sqlite3.connect(self.db_path)
        self.index = self._load_faiss_index()

    def _load_faiss_index(self) -> Optional[faiss.Index]:
        """Load FAISS index with error handling."""
        try:
            if os.path.exists(self.faiss_path):
                index = faiss.read_index(self.faiss_path)
                st.sidebar.success(f"Loaded FAISS index with {index.ntotal} vectors")
                return index
            else:
                st.sidebar.warning("No FAISS index found")
                return None
        except Exception as e:
            st.sidebar.error(f"Error loading FAISS index: {e}")
            return None

    def get_documents(self) -> pd.DataFrame:
        """Get list of documents from SQLite."""
        query = "SELECT document_id, title, author, date_added FROM documents_metadata"
        return pd.read_sql_query(query, self.conn)

    def get_document_chunks(self, document_id: str) -> pd.DataFrame:
        """Get chunks for a specific document with vector information."""
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
        
        # Add vector information if FAISS index is available
        if self.index is not None:
            chunks_df['vector'] = chunks_df['chunk_id'].apply(self.get_faiss_vector)
            chunks_df['vector_available'] = chunks_df['vector'].apply(lambda x: x is not None)
            chunks_df['vector_preview'] = chunks_df['vector'].apply(self.format_vector_preview)
            chunks_df['vector_norm'] = chunks_df['vector'].apply(
                lambda x: np.linalg.norm(x) if x is not None else None
            )
        else:
            chunks_df['vector_available'] = False
            chunks_df['vector_preview'] = 'FAISS index not available'
            chunks_df['vector_norm'] = None
            
        return chunks_df

    def get_faiss_vector(self, chunk_id: int) -> Optional[np.ndarray]:
        """Retrieve vector from FAISS index with error handling."""
        try:
            vector = self.index.reconstruct(int(chunk_id))
            return vector
        except Exception as e:
            st.sidebar.warning(f"Could not retrieve vector for chunk {chunk_id}: {e}")
            return None

    def format_vector_preview(self, vector: Optional[np.ndarray]) -> str:
        """Format vector for display."""
        if vector is None:
            return "No vector available"
        preview = ", ".join(f"{x:.4f}" for x in vector[:5])
        return f"[{preview}, ...]"

    def display_document_info(self, doc_row: pd.Series):
        """Display document metadata in an expandable section."""
        with st.expander("Document Information", expanded=True):
            cols = st.columns(2)
            with cols[0]:
                st.write("Title:", doc_row['title'])
                st.write("Author:", doc_row['author'])
            with cols[1]:
                st.write("ID:", doc_row['document_id'])
                st.write("Added:", doc_row['date_added'])

    def display_chunk_details(self, chunk_row: pd.Series):
        """Display detailed chunk information in an expandable section."""
        with st.expander(f"Chunk {chunk_row['chunk_id']} Details", expanded=False):
            cols = st.columns(2)
            with cols[0]:
                st.write("Chunk ID:", chunk_row['chunk_id'])
                st.write("Length:", chunk_row['chunk_length'])
                st.write("Added:", chunk_row['date_added'])
            with cols[1]:
                st.write("Start Index:", chunk_row['start_index'])
                st.write("End Index:", chunk_row['end_index'])
                if chunk_row['vector_available']:
                    st.write("Vector Norm:", f"{chunk_row['vector_norm']:.4f}")

            if chunk_row['vector_available']:
                with st.expander("Full Vector", expanded=False):
                    vector = chunk_row['vector']
                    st.write("Shape:", vector.shape)
                    st.write("First 10 components:", vector[:10])
                    
                    # Optional: Add vector visualization
                    if st.checkbox("Show vector visualization"):
                        st.line_chart(pd.DataFrame(vector))

            st.text_area("Chunk Text", chunk_row['chunk_text'], height=100)

    def run(self):
        """Main UI logic."""
        st.title("Document and Chunk Viewer")

        # Index statistics in sidebar
        if self.index is not None:
            with st.sidebar:
                st.write("FAISS Index Statistics:")
                st.write(f"Total vectors: {self.index.ntotal}")
                st.write(f"Dimension: {self.index.d}")
                st.write(f"Index size: {os.path.getsize(self.faiss_path) / 1024:.2f} KB")

        # Document selection
        documents = self.get_documents()
        selected_document = st.selectbox(
            "Select a document",
            documents['document_id'].tolist(),
            format_func=lambda x: documents[documents['document_id'] == x]['title'].iloc[0]
        )

        if selected_document:
            doc_row = documents[documents['document_id'] == selected_document].iloc[0]
            self.display_document_info(doc_row)
            
            # Get and display chunks
            chunks = self.get_document_chunks(selected_document)
            st.write(f"## Chunks ({len(chunks)} total)")
            
            # Chunk filtering options
            col1, col2 = st.columns(2)
            with col1:
                show_vectors_only = st.checkbox("Show only chunks with vectors")
            with col2:
                sort_by = st.selectbox(
                    "Sort by",
                    ['chunk_id', 'date_added', 'chunk_length', 'vector_norm']
                )

            # Filter and sort chunks
            if show_vectors_only:
                chunks = chunks[chunks['vector_available']]
            chunks = chunks.sort_values(sort_by)

            # Create display DataFrame
            display_df = chunks[[
                'chunk_id', 'chunk_length', 'vector_available',
                'vector_preview', 'vector_norm'
            ]]
            
            # Display chunks in a table
            st.dataframe(display_df)

            # Detailed chunk view
            selected_chunk_id = st.selectbox(
                "Select a chunk for detailed view",
                chunks['chunk_id'].tolist()
            )
            
            if selected_chunk_id:
                chunk_row = chunks[chunks['chunk_id'] == selected_chunk_id].iloc[0]
                self.display_chunk_details(chunk_row)

def main():
    viewer = DocumentViewer()
    viewer.run()

if __name__ == "__main__":
    main()    

    