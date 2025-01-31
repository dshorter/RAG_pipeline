import os
import sqlite3
import sys
import time
import faiss
import uuid
import hashlib
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple    

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.singleton_config import ConfigSingleton
from src.paths import get_db_path, get_faiss_path
from src.logging_config import get_logger    
from src.metrics_collector import  MetricsCollector   

class RAGSystem:
    def __init__(self):
        self.logger = get_logger('system')
        self.config = ConfigSingleton()
        self.db_path = get_db_path()
        self.faiss_path = get_faiss_path()  
        self.metrics_collector = MetricsCollector(self.db_path)     
        self._ensure_storage()

    def _ensure_storage(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS documents_metadata (
                        document_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        author TEXT NOT NULL,
                        source TEXT NOT NULL,
                        date_added TEXT NOT NULL,
                        document_length INTEGER NOT NULL,
                        summary TEXT NOT NULL,
                        tags TEXT NOT NULL
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS document_chunks_metadata (
                        chunk_id TEXT PRIMARY KEY,
                        faiss_id INTEGER NOT NULL UNIQUE,
                        document_id TEXT NOT NULL,
                        chunk_text TEXT NOT NULL,
                        start_index INTEGER NOT NULL,
                        end_index INTEGER NOT NULL,
                        chunk_length INTEGER NOT NULL,
                        date_added TEXT NOT NULL,
                        relevance_score REAL,
                        vector_stats TEXT,
                        FOREIGN KEY (document_id) REFERENCES documents_metadata(document_id)
                    )
                """)

            if not os.path.exists(self.faiss_path):
                dimension = 1536
                index = faiss.IndexIDMap2(faiss.IndexFlatL2(dimension))
                faiss.write_index(index, self.faiss_path)
                
        except Exception as e:
            self.logger.error(f"Storage initialization failed: {e}")
            raise

    def add_vector(self, chunk: str, vector: np.ndarray, document_id: str,
                  source: str, start_index: int, end_index: int,
                  additional_metadata: Dict[str, Any] = None,
                  doc_metadata: Dict[str, Any] = None):
        start_time = time.time()
        try:
            chunk_id = str(uuid.uuid4())
            faiss_id = self._generate_faiss_id(chunk_id)
            
            # Add to FAISS
            index = faiss.read_index(self.faiss_path)
            index.add_with_ids(
                np.array(vector).reshape(1, -1).astype('float32'),
                np.array([faiss_id], dtype=np.int64)
            )
            faiss.write_index(index, self.faiss_path)
            
            # Add to SQLite
            with sqlite3.connect(self.db_path) as conn:
                # Insert document metadata if new
                if doc_metadata:
                    conn.execute("""
                        INSERT OR IGNORE INTO documents_metadata 
                        (document_id, title, author, source, date_added, 
                         document_length, summary, tags)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        document_id,
                        doc_metadata.get('title', ''),
                        doc_metadata.get('author', ''),
                        source,
                        datetime.now().isoformat(),
                        len(chunk.split()),
                        doc_metadata.get('summary', ''),
                        json.dumps(doc_metadata.get('tags', []))
                    ))
                
                # Insert chunk metadata
                conn.execute("""
                    INSERT INTO document_chunks_metadata 
                    (chunk_id, faiss_id, document_id, chunk_text,
                     start_index, end_index, chunk_length, date_added)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk_id,
                    faiss_id,
                    document_id,
                    chunk,
                    start_index,
                    end_index,
                    len(chunk.split()),
                    datetime.now().isoformat()
                ))

            self.metrics_collector.collect(
                operation='vector_storage',
                component='rag_system',
                metrics={
                    'duration_ms': (time.time() - start_time) * 1000,
                    'vector_dimension': len(vector),
                    'chunk_length': len(chunk.split()),
                    'document_id': document_id,
                    'chunk_id': chunk_id
                }
            )

        except Exception as e:
            self.logger.error(f"Failed to add vector: {e}")
            self.metrics_collector.collect(
                operation='vector_storage',
                component='rag_system',
                metrics={
                    'duration_ms': (time.time() - start_time) * 1000,
                    'success': False,
                    'error': str(e)
                }
            )
            raise

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        start_time = time.time()
        try:
            index = faiss.read_index(self.faiss_path)
            distances, indices = index.search(query_vector.reshape(1, -1), k)
            
            results = []
            with sqlite3.connect(self.db_path) as conn:
                for i, idx in enumerate(indices[0]):
                    if idx == -1:
                        continue
                        
                    cursor = conn.execute("""
                        SELECT 
                            c.chunk_id, c.chunk_text, c.document_id,
                            c.start_index, c.end_index,
                            d.title, d.author, d.source
                        FROM document_chunks_metadata c
                        LEFT JOIN documents_metadata d ON c.document_id = d.document_id
                        WHERE c.faiss_id = ?
                    """, (int(idx),))
                    
                    row = cursor.fetchone()
                    if row:
                        results.append({
                            'chunk_id': row[0],
                            'chunk_text': row[1],
                            'document_id': row[2],
                            'relevance_score': float(1 / (1 + distances[0][i])),
                            'source_info': {
                                'title': row[5] or 'Unknown',
                                'author': row[6] or 'Unknown',
                                'source': row[7] or 'Unknown',
                                'start_index': row[3],
                                'end_index': row[4]
                            }
                        })

            self.metrics_collector.collect(
                operation='vector_search',
                component='rag_system',
                metrics={
                    'duration_ms': (time.time() - start_time) * 1000,
                    'num_requested': k,
                    'num_returned': len(results),
                    'avg_distance': float(np.mean(distances))
                }
            )

            return results

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            self.metrics_collector.collect(
                operation='vector_search',
                component='rag_system',
                metrics={
                    'duration_ms': (time.time() - start_time) * 1000,
                    'success': False,
                    'error': str(e)
                }
            )
            raise

    def _generate_faiss_id(self, chunk_id: str) -> int:
        return int(hashlib.sha256(chunk_id.encode()).hexdigest(), 16) % (2**63 - 1)
 