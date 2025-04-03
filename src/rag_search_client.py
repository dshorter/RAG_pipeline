import os  
import sys
import faiss
import sqlite3
import numpy as np
from typing import Dict, Any, List, Union
import json    
import time
from src.metrics_collector import MetricsCollector
from src.singleton_config import ConfigSingleton
from src.paths import get_db_path, get_faiss_path
from src.logging_config import get_logger  


class RAGSearchClient:
    def __init__(self):
        self.logger = get_logger('search')
        self.config = ConfigSingleton()
        self.db_path = get_db_path()      
        self.metrics_collector = MetricsCollector()  
        self.faiss_path = get_faiss_path()  
        self._validate_resources()

    def _validate_resources(self):
        if not os.path.exists(self.faiss_path):
            raise FileNotFoundError(f"FAISS index not found at {self.faiss_path}")
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"SQLite database not found at {self.db_path}")

    def search(self, query_vector: Union[np.ndarray, List[float]], k: int = 5) -> List[Dict[str, Any]]:
        start_time = time.time()
        try:
            if isinstance(query_vector, list):
                query_vector = np.array(query_vector, dtype=np.float32)
            if len(query_vector.shape) == 1:
                query_vector = query_vector.reshape(1, -1)

            # FAISS Search
            index = faiss.read_index(self.faiss_path)
            distances, faiss_ids = index.search(query_vector, k)
            
            # Get chunks with citation info
            results = self._get_chunk_info(faiss_ids[0], distances[0])

            # Fire and forget metrics
            self.metrics_collector.collect(
                operation='vector_search',
                component='search_client',
                metrics={
                    'duration_ms': (time.time() - start_time) * 1000,
                    'num_requested': k,
                    'num_returned': len(results),
                    'avg_distance': float(np.mean(distances)) if len(distances) > 0 else 0,
                    'min_distance': float(np.min(distances)) if len(distances) > 0 else 0,
                    'max_distance': float(np.max(distances)) if len(distances) > 0 else 0
                }
            )

            self.logger.info(f"Search completed. Found {len(results)} results.")
            return results

        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            # Fire and forget error metrics
            self.metrics_collector.collect(
                operation='vector_search',
                component='search_client',
                metrics={
                    'duration_ms': (time.time() - start_time) * 1000,
                    'success': False,
                    'error': str(e)
                }
            )
            raise

    def _get_chunk_info(self, faiss_ids: np.ndarray, distances: np.ndarray) -> List[Dict[str, Any]]:
        """Get chunk information from SQLite database."""
        results = []  
        
        self.logger.info(f"Getting chunk info for {len(faiss_ids)} FAISS IDs")
        
        with sqlite3.connect(self.db_path) as conn:
            for i, faiss_id in enumerate(faiss_ids):
                if faiss_id == -1:
                    continue
                    
                # Log the FAISS ID we're looking up
                self.logger.debug(f"Looking up chunk_id for FAISS ID: {faiss_id}")
            
                cursor = conn.execute("""
                    SELECT 
                        c.chunk_id,
                        c.chunk_text,
                        c.document_id,
                        c.start_index,
                        c.end_index,
                        d.title,
                        d.author,
                        d.source
                    FROM document_chunks_metadata c
                    LEFT JOIN documents_metadata d ON c.document_id = d.document_id
                    WHERE c.faiss_id = ?
                """, (int(faiss_id),))
                
                row = cursor.fetchone()
                if row:
                    chunk_id, text, doc_id, start, end, title, author, source = row
                    self.logger.debug(f"Found chunk with id: {chunk_id}")
                    
                    results.append({
                        'chunk_id': chunk_id,
                        'chunk_text': text,
                        'document_id': doc_id,
                        'relevance_score': float(1 / (1 + distances[i])),
                        'source_info': {
                            'title': title or 'Unknown Document',
                            'author': author or 'Unknown Author',
                            'source': source or 'Unknown Source',
                            'start_index': start,
                            'end_index': end
                        }
                    })
                else:
                    self.logger.warning(f"No chunk found for FAISS ID: {faiss_id}")
        
        self.logger.info(f"Retrieved {len(results)} chunks from database")
        return results