# src/rag_search_client.py

import os  
import sys
import faiss
import sqlite3
import numpy as np
from typing import Dict, Any, List, Union
import json    

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.singleton_config import ConfigSingleton
from src.paths import get_db_path, get_faiss_path
from src.logging_config import get_logger

class RAGSearchClient:
    """Lightweight client for RAG search operations."""
    
    def __init__(self):
        self.logger = get_logger('search')
        self.config = ConfigSingleton()
        self.db_path = get_db_path()
        self.faiss_path = get_faiss_path()
        
        # Validate that necessary files exist
        self._validate_resources()

    def _validate_resources(self):
        """Validate that required index and database files exist."""
        if not os.path.exists(self.faiss_path):
            raise FileNotFoundError(f"FAISS index not found at {self.faiss_path}")
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"SQLite database not found at {self.db_path}")

    def _format_citation(self, source_info: Dict[str, Any]) -> str:
        """Format source information into a citation string."""
        try:
            if source_info.get('author'):
                citation = f"{source_info['author']}"
            else:
                citation = "Unknown Author"
                
            if source_info.get('title'):
                citation += f", '{source_info['title']}'"
                
            if source_info.get('source'):
                citation += f" ({source_info['source']})"
                
            if source_info.get('start_index') is not None and source_info.get('end_index') is not None:
                citation += f", section {source_info['start_index']}-{source_info['end_index']}"
                
            return citation
            
        except Exception as e:
            self.logger.warning(f"Citation formatting failed: {str(e)}")
            return "Citation unavailable"


    def search(self, query_vector: Union[np.ndarray, List[float]], k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors and fetch corresponding metadata."""
        try:
            self.logger.debug("Starting vector search", 
                            extra={'operation': 'vector_search'})

            # Ensure query vector is in correct format
            if isinstance(query_vector, list):
                query_vector = np.array(query_vector, dtype=np.float32)
            if len(query_vector.shape) == 1:
                query_vector = query_vector.reshape(1, -1)

            # 1. FAISS Search
            index = faiss.read_index(self.faiss_path)
            distances, faiss_ids = index.search(query_vector, k)
            
            self.logger.debug(f"Found {len(faiss_ids[0])} matches", 
                            extra={'operation': 'vector_search'})

            # 2. Fetch Metadata from SQLite
            results = []
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for i, faiss_id in enumerate(faiss_ids[0]):
                    if faiss_id == -1:  # FAISS returns -1 for no match
                        continue
                        
                    # Get chunk metadata and document metadata in one query
                    cursor.execute('''
                        SELECT 
                            c.chunk_id,
                            c.chunk_text,
                            c.document_id,
                            c.start_index,
                            c.end_index,
                            c.metadata as chunk_metadata,
                            d.title,
                            d.author,
                            d.source,
                            d.metadata as doc_metadata
                        FROM document_chunks_metadata c
                        LEFT JOIN documents_metadata d ON c.document_id = d.document_id
                        WHERE c.faiss_id = ?
                    ''', (int(faiss_id),))
                    
                    row = cursor.fetchone()
                    if row:
                        chunk_id, chunk_text, doc_id, start_index, end_index, \
                        chunk_metadata, title, author, source, doc_metadata = row
                        
                        # Calculate relevance score (1 / (1 + distance))
                        relevance_score = float(1 / (1 + distances[0][i]))
                        
                        # Combine into a result with all metadata
                        results.append({
                            "chunk_id": chunk_id,
                            "chunk_text": chunk_text,
                            "document_id": doc_id,
                            "relevance_score": relevance_score,
                            "distance": float(distances[0][i]),
                            "metadata": json.loads(chunk_metadata) if chunk_metadata else {},
                            "source_info": {
                                "title": title or "Unknown Document",
                                "author": author or "Unknown Author",
                                "source": source or "Unknown Source",
                                "start_index": start_index,
                                "end_index": end_index
                            },
                            "document_metadata": json.loads(doc_metadata) if doc_metadata else {}
                        })

                        print("\nDEBUG: Search Results: ==============================")
                        for i, result in enumerate(results):
                            print(f"\nResult {i+1}:")
                            print(f"Keys: {result.keys()}")
                            print(f"Source Info: {result.get('source_info', {})}")
                            print(f"Metadata: {result.get('metadata', {})}")
                            print(f"Title: {result.get('source_info', {}).get('title', 'No title')}")
                            print(f"Author: {result.get('source_info', {}).get('author', 'No author')}")
                            print(f"Relevance: {result.get('relevance_score', 'No score')}")

            self.logger.info(f"Search completed, found {len(results)} results", 
                            extra={'operation': 'search_complete'})
            return results

        except Exception as e:
            self.logger.error(f"Search operation failed: {str(e)}")
            raise



    def get_document_metadata(self, document_id: str) -> Dict[str, Any]:
        """Retrieve metadata for a specific document."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT title, author, source, date_added, document_length, 
                       summary, tags, metadata
                FROM documents_metadata
                WHERE document_id = ?
            ''', (document_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    "title": row[0],
                    "author": row[1],
                    "source": row[2],
                    "date_added": row[3],
                    "document_length": row[4],
                    "summary": row[5],
                    "tags": row[6],
                    "additional_metadata": json.loads(row[7])
                }
            return None
            
        finally:
            conn.close()

    def get_index_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the search index."""
        try:
            index = faiss.read_index(self.faiss_path)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM document_chunks_metadata')
            chunk_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM documents_metadata')
            doc_count = cursor.fetchone()[0]
            
            return {
                "total_vectors": index.ntotal,
                "dimension": index.d,
                "total_chunks": chunk_count,
                "total_documents": doc_count
            }
            
        finally:
            conn.close()