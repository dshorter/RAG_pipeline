from datetime import datetime
import sqlite3
import faiss
import uuid
import hashlib
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Union
import os
from src.singleton_config import ConfigSingleton
from src.paths import *
from src.logging_config import get_logger

class RAGSystem:
    def __init__(self):
        self.logger = get_logger('system')
        self.logger.debug("Initializing RAG System", 
                         extra={
                             'component': 'rag_system',
                             'operation': 'initialization',
                             'module_line': 'rag_system:init'
                         })
        
        self.config = ConfigSingleton()
        self.db_path = get_db_path()
        self.faiss_path = get_faiss_path()
        
        # Initialize storage
        self._initialize_storage()
        
        self.logger.info("System initialized", 
                        extra={
                            'component': 'rag_system',
                            'operation': 'initialization_complete',
                            'module_line': 'rag_system:init'
                        })

    def _generate_faiss_id(self, chunk_id: str) -> int:
        """
        Generate deterministic FAISS ID from chunk ID using the last 63 bits of SHA-256 hash.
        
        Args:
            chunk_id: The chunk ID to generate FAISS ID from
            
        Returns:
            A positive integer suitable for FAISS indexing
        """
        return int(hashlib.sha256(chunk_id.encode()).hexdigest(), 16) % (2**63 - 1)  
    
    
    def _initialize_storage(self):
        """Initialize SQLite and FAISS storage."""
        try:
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.faiss_path), exist_ok=True)
            
            # Initialize SQLite
            conn = sqlite3.connect(self.db_path)
            self._create_tables(conn)
            conn.close()

            # Initialize FAISS if needed
            if not os.path.exists(self.faiss_path):
                dimension = 1536
                index = faiss.IndexIDMap2(faiss.IndexFlatL2(dimension))
                faiss.write_index(index, self.faiss_path)
                
                self.logger.info("Created new FAISS index", 
                               extra={
                                   'component': 'rag_system',
                                   'operation': 'faiss_initialization',
                                   'module_line': 'rag_system:_initialize_storage'
                               })
                
        except Exception as e:
            self.logger.error("Storage initialization failed", 
                            extra={
                                'component': 'rag_system',
                                'operation': 'storage_initialization',
                                'module_line': 'rag_system:_initialize_storage',
                                'error_id': str(uuid.uuid4())[:8],
                                'function_name': '_initialize_storage'
                            })
            raise
#####################
    def _create_tables(self, conn):
        """Create necessary SQLite tables with hash ID support."""
        try:
            self.logger.debug("Starting table creation", 
                            extra={
                                'component': 'rag_system',
                                'operation': 'table_creation',
                                'module_line': 'rag_system:_create_tables'
                            })
            
            cursor = conn.cursor()
            
            # Documents table remains unchanged
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents_metadata (
                document_id TEXT PRIMARY KEY NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                author TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT '',
                date_added TEXT NOT NULL,
                document_length INTEGER NOT NULL DEFAULT 0,
                summary TEXT NOT NULL DEFAULT '',
                tags TEXT NOT NULL DEFAULT '',
                metadata TEXT NOT NULL DEFAULT '{}',
                CONSTRAINT valid_document_id CHECK(document_id != ''),
                CONSTRAINT valid_date_format CHECK(date_added IS strftime('%Y-%m-%d %H:%M:%S', date_added))
            )
            ''')

            # Updated chunks table with hash ID support
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_chunks_metadata (
                chunk_id TEXT PRIMARY KEY NOT NULL,
                faiss_id INTEGER NOT NULL UNIQUE,  -- New field for hash-based ID
                document_id TEXT NOT NULL,
                chunk_text TEXT NOT NULL,
                start_index INTEGER NOT NULL,
                end_index INTEGER NOT NULL,
                chunk_length INTEGER NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                date_added TEXT NOT NULL,
                relevance_score REAL,              -- New field for relevance tracking
                vector_stats TEXT,                 -- New field for vector statistics
                FOREIGN KEY (document_id) REFERENCES documents_metadata(document_id) ON DELETE CASCADE,
                CONSTRAINT valid_chunk_id CHECK(chunk_id != ''),
                CONSTRAINT valid_indices CHECK(end_index > start_index),
                CONSTRAINT valid_length CHECK(chunk_length > 0),
                CONSTRAINT valid_date_format CHECK(date_added IS strftime('%Y-%m-%d %H:%M:%S', date_added))
            )
            ''')
            
            # Update indices
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_document_chunks_faiss_id 
            ON document_chunks_metadata(faiss_id)
            ''')
            
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id 
            ON document_chunks_metadata(document_id)
            ''')

            conn.commit()
            
            self.logger.info("Tables created successfully", 
                        extra={
                            'component': 'rag_system',
                            'operation': 'table_creation',
                            'module_line': 'rag_system:_create_tables'
                        })

        except sqlite3.Error as e:
            error_id = str(uuid.uuid4())[:8]
            self.logger.error("Failed to create tables", 
                            extra={
                                'component': 'rag_system',
                                'operation': 'table_creation',
                                'module_line': 'rag_system:_create_tables',
                                'error_id': error_id,
                                'function_name': '_create_tables',
                                'error': str(e)
                            })
            raise


##################### 
    def _insert_document_metadata(self, conn, document_id: str, title: str, 
                                author: str, source: str, document_length: int,
                                summary: str, tags: str, metadata: Dict[str, Any]):
        """Insert document metadata with type validation."""
        try:
            self.logger.debug("Inserting document metadata", 
                            extra={
                                'component': 'rag_system',
                                'operation': 'document_metadata_insertion',
                                'module_line': 'rag_system:_insert_document_metadata',
                                'document_id': document_id
                            })
            
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO documents_metadata 
                (document_id, title, author, source, date_added, document_length, 
                 summary, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(document_id),
                str(title),
                str(author),
                str(source),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                int(document_length),
                str(summary),
                str(tags),
                json.dumps(metadata)
            ))
            
            self.logger.info("Document metadata inserted successfully", 
                           extra={
                               'component': 'rag_system',
                               'operation': 'document_metadata_insertion',
                               'module_line': 'rag_system:_insert_document_metadata',
                               'document_id': document_id
                           })

        except sqlite3.Error as e:
            error_id = str(uuid.uuid4())[:8]
            self.logger.error("Failed to insert document metadata", 
                            extra={
                                'component': 'rag_system',
                                'operation': 'document_metadata_insertion',
                                'module_line': 'rag_system:_insert_document_metadata',
                                'error_id': error_id,
                                'function_name': '_insert_document_metadata',
                                'document_id': document_id,
                                'error': str(e)
                            })
            raise sqlite3.Error(f"Failed to insert document metadata: {str(e)}")

    def _insert_chunk_metadata(self, conn, chunk_id: str, document_id: str,
                             chunk_text: str, start_index: int, end_index: int,
                             embedding_id: int, metadata: Dict[str, Any]):
        """Insert chunk metadata with type validation."""
        try:
            self.logger.debug("Inserting chunk metadata", 
                            extra={
                                'component': 'rag_system',
                                'operation': 'chunk_metadata_insertion',
                                'module_line': 'rag_system:_insert_chunk_metadata',
                                'chunk_id': chunk_id,
                                'document_id': document_id
                            })
            
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO document_chunks_metadata 
                (chunk_id, document_id, chunk_text, start_index, end_index, 
                 chunk_length, metadata, date_added, embedding_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(chunk_id),
                str(document_id),
                str(chunk_text),
                int(start_index),
                int(end_index),
                int(len(chunk_text)),
                str(metadata),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                int(embedding_id)
            ))
            
            self.logger.info("Chunk metadata inserted successfully", 
                           extra={
                               'component': 'rag_system',
                               'operation': 'chunk_metadata_insertion',
                               'module_line': 'rag_system:_insert_chunk_metadata',
                               'chunk_id': chunk_id,
                               'document_id': document_id
                           })

        except sqlite3.Error as e:
            error_id = str(uuid.uuid4())[:8]
            self.logger.error("Failed to insert chunk metadata", 
                            extra={
                                'component': 'rag_system',
                                'operation': 'chunk_metadata_insertion',
                                'module_line': 'rag_system:_insert_chunk_metadata',
                                'error_id': error_id,
                                'function_name': '_insert_chunk_metadata',
                                'chunk_id': chunk_id,
                                'document_id': document_id,
                                'error': str(e)
                            })
            raise sqlite3.Error(f"Failed to insert chunk metadata: {str(e)}")

#########################
    def add_vector(self, document_id: str, chunk: str, vector: Union[np.ndarray, List[float]], 
              source: str, start_index: int, end_index: int,
              additional_metadata: Dict[str, Any] = None, 
              doc_metadata: Dict[str, Any] = None) -> Tuple[str, int]:
        """Add vector and metadata with new hash ID system."""
        try:
            self.logger.debug("Starting vector addition", 
                            extra={
                                'component': 'rag_system',
                                'operation': 'vector_addition',
                                'module_line': 'rag_system:add_vector'
                            })
            
            # Initialize default values
            additional_metadata = additional_metadata or {}
            doc_metadata = doc_metadata or {}
            
            # Ensure vector is numpy array with correct type
            if isinstance(vector, list):
                vector = np.array(vector, dtype=np.float32)
            vector = vector.astype(np.float32)
            if len(vector.shape) == 1:
                vector = vector.reshape(1, -1)

            # Generate IDs
            chunk_id = str(uuid.uuid4())
            faiss_id = self._generate_faiss_id(chunk_id)

            # Calculate vector statistics
            vector_stats = {
                "norm": float(np.linalg.norm(vector)),
                "mean": float(np.mean(vector)),
                "std": float(np.std(vector)),
                "min": float(np.min(vector)),
                "max": float(np.max(vector))
            }

            # Initialize SQLite connection
            conn = sqlite3.connect(self.db_path)
            try:
                # Start transaction
                conn.execute('BEGIN')
                
                self._insert_document_metadata(
                    conn,
                    document_id=document_id,
                    title=doc_metadata.get('title', ''),
                    author=doc_metadata.get('author', ''),
                    source=source,
                    document_length=len(chunk),
                    summary=doc_metadata.get('summary', ''),
                    tags=doc_metadata.get('tags', ''),
                    metadata=additional_metadata
                )

                # Insert chunk with new fields
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO document_chunks_metadata 
                    (chunk_id, faiss_id, document_id, chunk_text, start_index, end_index, 
                    chunk_length, metadata, date_added, vector_stats)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    chunk_id,
                    faiss_id,
                    document_id,
                    chunk,
                    start_index,
                    end_index,
                    len(chunk),
                    json.dumps(additional_metadata),
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    json.dumps(vector_stats)
                ))

                conn.commit()
                
                self.logger.info("Database operations completed", 
                            extra={
                                'component': 'rag_system',
                                'operation': 'db_insert',
                                'module_line': 'rag_system:add_vector'
                            })

            except sqlite3.Error as e:
                conn.rollback()
                error_id = str(uuid.uuid4())[:8]
                self.logger.error(f"Database operation failed: {str(e)}", 
                                extra={
                                    'component': 'rag_system',
                                    'operation': 'db_insert',
                                    'module_line': 'rag_system:add_vector',
                                    'error_id': error_id,
                                    'function_name': 'add_vector'
                                })
                raise
            finally:
                conn.close()

            # Handle FAISS operations
            try:
                if os.path.exists(self.faiss_path):
                    index = faiss.read_index(self.faiss_path)
                else:
                    index = faiss.IndexIDMap2(faiss.IndexFlatL2(vector.shape[1]))

                index.add_with_ids(vector, np.array([faiss_id], dtype=np.int64))
                faiss.write_index(index, self.faiss_path)
                
                self.logger.info("FAISS operations completed", 
                            extra={
                                'component': 'rag_system',
                                'operation': 'faiss_insert',
                                'module_line': 'rag_system:add_vector'
                            })
                
                return chunk_id, faiss_id

            except Exception as e:
                error_id = str(uuid.uuid4())[:8]
                self.logger.error(f"FAISS operation failed: {str(e)}", 
                                extra={
                                    'component': 'rag_system',
                                    'operation': 'faiss_insert',
                                    'module_line': 'rag_system:add_vector',
                                    'error_id': error_id,
                                    'function_name': 'add_vector'
                                })
                self._cleanup_failed_insert(chunk_id, document_id)
                raise

        except Exception as e:
            error_id = str(uuid.uuid4())[:8]
            self.logger.error(f"Vector addition failed: {str(e)}", 
                            extra={
                                'component': 'rag_system',
                                'operation': 'vector_addition',
                                'module_line': 'rag_system:add_vector',
                                'error_id': error_id,
                                'function_name': 'add_vector'
                            })
            raise