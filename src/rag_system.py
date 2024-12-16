import os
import sqlite3
import faiss
import uuid
import hashlib
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Union
from src.singleton_config import ConfigSingleton
from src.paths import *
from src.logging_config import get_logger

class RAGSystem:
    def __init__(self, db_path: str = None):
        self.logger = get_logger('system')
        self.logger.debug("Initializing RAG System", 
                         extra={
                             'component': 'rag_system',
                             'operation': 'initialization',
                             'module_line': 'rag_system:init'                                
                         })    
        self.db_path = os.path.abspath(db_path or get_db_path())
        self.config = ConfigSingleton()
        self.faiss_path = os.path.abspath(get_faiss_path())
        
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
        """
        return int(hashlib.sha256(chunk_id.encode()).hexdigest(), 16) % (2**63 - 1)  
    
    def _initialize_storage(self):
        """Initialize SQLite and FAISS storage."""
        try:
            db_dir = os.path.dirname(self.db_path)
            if db_dir:  
                os.makedirs(db_dir, exist_ok=True)
            
            faiss_dir = os.path.dirname(self.faiss_path)
            if faiss_dir:  
                os.makedirs(faiss_dir, exist_ok=True)
            
            try:
                conn = sqlite3.connect(self.db_path)
                self._create_tables(conn)
                conn.close()
            except sqlite3.OperationalError as e:
                self.logger.error(f"Failed to connect to the SQLite database at {self.db_path}", 
                                  extra={'error': str(e)})
                raise e

            if not os.path.exists(self.faiss_path):
                try:
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
                    self.logger.error("Failed to initialize FAISS", 
                                    extra={
                                        'component': 'rag_system',
                                        'operation': 'faiss_initialization',
                                        'error': str(e)
                                    })
                    raise e
                    
        except Exception as e:
            self.logger.error("Storage initialization failed", 
                            extra={
                                'component': 'rag_system',
                                'operation': 'storage_initialization',
                                'module_line': 'rag_system:_initialize_storage',
                                'error_id': str(uuid.uuid4())[:8],
                                'error': str(e)
                            })
            raise e
    
    def _create_tables(self, conn):
        try:
            self.logger.debug("Starting table creation", 
                            extra={
                                'component': 'rag_system',
                                'operation': 'table_creation'
                            })
            
            cursor = conn.cursor()
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents_metadata (
                document_id TEXT PRIMARY KEY NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                author TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT '',
                date_added TEXT NOT NULL,
                document_length INTEGER NOT NULL DEFAULT 0,
                summary TEXT NOT NULL DEFAULT '',
                tags TEXT NOT NULL DEFAULT ''
            );
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_chunks_metadata (
                chunk_id TEXT PRIMARY KEY NOT NULL,
                faiss_id INTEGER NOT NULL UNIQUE,
                document_id TEXT NOT NULL,
                chunk_text TEXT NOT NULL,
                start_index INTEGER NOT NULL,
                end_index INTEGER NOT NULL,
                chunk_length INTEGER NOT NULL,
                date_added TEXT NOT NULL,
                relevance_score REAL,
                vector_stats TEXT
            );
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_metrics (
                metric_id TEXT PRIMARY KEY NOT NULL,
                document_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                operation_type TEXT NOT NULL,
                chunk_count INTEGER,
                token_count INTEGER,
                processing_time REAL NOT NULL,
                success BOOLEAN NOT NULL,
                error_message TEXT,
                additional_metrics JSON
            );
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_metrics (
                metric_id TEXT PRIMARY KEY NOT NULL,
                timestamp TEXT NOT NULL,
                query_text TEXT NOT NULL,
                embedding_time REAL NOT NULL,
                search_time REAL NOT NULL,
                total_time REAL NOT NULL,
                num_chunks_requested INTEGER NOT NULL,
                num_chunks_returned INTEGER NOT NULL,
                relevance_scores JSON NOT NULL,
                success BOOLEAN NOT NULL,
                error_message TEXT,
                additional_metrics JSON
            );
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                metric_id TEXT PRIMARY KEY NOT NULL,
                timestamp TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT NOT NULL,
                context JSON
            );
            ''')
            
            conn.commit()
            self.logger.info("Tables and indices created successfully", 
                        extra={
                            'component': 'rag_system',
                            'operation': 'table_creation'
                        })
        except sqlite3.Error as e:
            error_id = str(uuid.uuid4())[:8]
            self.logger.error("Failed to create tables", 
                            extra={
                                'component': 'rag_system',
                                'operation': 'table_creation',
                                'error_id': error_id,
                                'error': str(e)
                            })
            raise e
