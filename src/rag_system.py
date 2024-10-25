import sqlite3
import faiss
import uuid
import hashlib
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, List    
import os  
from src.singleton_config import ConfigSingleton  
from src.paths import *   
from src.logging_config import get_logger

class RAGSystem:
    def __init__(self):
        self.logger = get_logger('system')
        self.logger.debug("Initializing RAG System", 
                         extra={'operation': 'initialization'})
        
        self.config = ConfigSingleton()
        self.db_path = get_db_path()
        self.faiss_path = get_faiss_path()
        
        self.logger.info("System initialized", 
                        extra={
                            'operation': 'initialization',
                            'db_path': self.db_path,
                            'faiss_path': self.faiss_path
                        })

    def add_vector(self, chunk: str, vector: np.array, document_id: str,
                  source: str, start_index: int, end_index: int,
                  additional_metadata: Dict[str, Any] = {},
                  doc_metadata: List[Dict] = {}):
        
        self.logger.debug("Processing vector addition", 
                         extra={
                             'operation': 'vector_addition',
                             'document_id': document_id,
                             'chunk_size': len(chunk)
                         })

        chunk_id = self.generate_chunk_id()
        hashed_id = self.get_hashed_id(chunk_id)

        try:
            # Initialize SQLite connection
            sql_conn = sqlite3.connect(self.db_path)
            self.conn = sql_conn
            self.conn.execute('BEGIN')

            self.logger.debug("Database connection established", 
                            extra={'operation': 'db_connection'})

            self.create_documents_table()
            self.create_document_chunks_table()

            self.insert_document_metadata(
                document_id, 
                doc_metadata["title"], 
                doc_metadata["author"],
                source, len(chunk), 
                "Summary", "Tags", 
                additional_metadata
            )

            self.insert_chunk_metadata(
                hashed_id, document_id, chunk,
                start_index, end_index, len(chunk),
                additional_metadata
            )
            
            self.conn.commit()
            self.logger.info("Metadata inserted successfully", 
                           extra={
                               'operation': 'metadata_insertion',
                               'document_id': document_id
                           })

            # Handle FAISS vector addition
            self.add_vector_to_faiss(vector, hashed_id)
            self.logger.info("Vector added successfully", 
                           extra={
                               'operation': 'faiss_addition',
                               'hashed_id': hashed_id
                           })

        except sqlite3.Error as e:
            self.conn.rollback()
            self.logger.error("Database operation failed", 
                            extra={
                                'operation': 'db_operation',
                                'error': str(e),
                                'document_id': document_id
                            })
            raise e
        except Exception as e:
            if hasattr(self, 'conn'):
                self.conn.rollback()
            self.logger.error("Vector addition failed", 
                            extra={
                                'operation': 'vector_addition',
                                'error': str(e),
                                'document_id': document_id
                            })
            raise e

    def add_vector_to_faiss(self, vector: np.array, hashed_id: int):
        try:
            self.logger.debug("Adding vector to FAISS", 
                            extra={
                                'operation': 'faiss_operation',
                                'vector_shape': np.array([vector]).shape
                            })
            
            index = faiss.IndexIDMap(faiss.IndexFlatL2(1536))
            index.add_with_ids(
                np.array([vector]).astype('float32'),
                np.array([hashed_id], dtype='int64')
            )
            
            index_path = self.faiss_path
            faiss.write_index(index, index_path)
            
            self.logger.info("FAISS index updated successfully", 
                           extra={
                               'operation': 'faiss_operation',
                               'index_path': index_path
                           })

        except Exception as e:
            self.logger.error("FAISS operation failed", 
                            extra={
                                'operation': 'faiss_operation',
                                'error': str(e),
                                'hashed_id': hashed_id
                            })
            raise e

    def insert_document_metadata(self, document_id: str, title: str, 
                               author: str, source: str, document_length: int,
                               summary: str, tags: str, 
                               metadata: Dict[str, Any]):
        cursor = self.conn.cursor()
        try:
            self.logger.debug("Inserting document metadata", 
                            extra={
                                'operation': 'metadata_insertion',
                                'document_id': document_id
                            })
            
            cursor.execute('''
                INSERT INTO documents_metadata (
                    document_id, title, author, source, date_added,
                    document_length, summary, tags, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                document_id,
                title,
                author,
                source,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                document_length,
                summary,
                tags,
                json.dumps(metadata)
            ))
            
            self.logger.info("Document metadata inserted", 
                           extra={
                               'operation': 'metadata_insertion',
                               'document_id': document_id
                           })

        except sqlite3.IntegrityError as e:
            self.logger.warning("Document metadata already exists", 
                              extra={
                                  'operation': 'metadata_insertion',
                                  'document_id': document_id,
                                  'error': str(e)
                              })
        except sqlite3.Error as e:
            self.logger.error("Failed to insert document metadata", 
                            extra={
                                'operation': 'metadata_insertion',
                                'document_id': document_id,
                                'error': str(e)
                            })
            raise e

    def insert_chunk_metadata(self, chunk_id: int, document_id: str,
                            chunk_text: str, start_index: int,
                            end_index: int, chunk_length: int,
                            additional_metadata: Dict[str, Any]):
        cursor = self.conn.cursor()
        try:
            self.logger.debug("Inserting chunk metadata", 
                            extra={
                                'operation': 'chunk_insertion',
                                'chunk_id': chunk_id,
                                'document_id': document_id
                            })
            
            cursor.execute('''
                INSERT INTO document_chunks_metadata (
                    chunk_id, document_id, chunk_text, start_index,
                    end_index, chunk_length, metadata, date_added
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                chunk_id,
                document_id,
                chunk_text,
                start_index,
                end_index,
                chunk_length,
                json.dumps(additional_metadata),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
            
            self.logger.info("Chunk metadata inserted", 
                           extra={
                               'operation': 'chunk_insertion',
                               'chunk_id': chunk_id,
                               'document_id': document_id
                           })
                           
        except sqlite3.Error as e:
            self.logger.error("Failed to insert chunk metadata", 
                            extra={
                                'operation': 'chunk_insertion',
                                'chunk_id': chunk_id,
                                'error': str(e)
                            })
            raise e

    def create_documents_table(self):
        cursor = self.conn.cursor()
        try:
            self.logger.debug("Creating documents table", 
                            extra={'operation': 'table_creation'})
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents_metadata (
                document_id TEXT PRIMARY KEY,
                title TEXT,
                author TEXT,
                source TEXT,
                date_added TEXT,
                document_length INTEGER,
                summary TEXT,
                tags TEXT,
                metadata TEXT
            )
            ''')
            
            self.conn.commit()
            self.logger.debug("Documents table created/verified", 
                            extra={'operation': 'table_creation'})
                            
        except sqlite3.Error as e:
            self.logger.error("Failed to create documents table", 
                            extra={
                                'operation': 'table_creation',
                                'error': str(e)
                            })
            raise e

    def create_document_chunks_table(self):
        cursor = self.conn.cursor()
        try:
            self.logger.debug("Creating chunks table", 
                            extra={'operation': 'table_creation'})
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_chunks_metadata (
                chunk_id INTEGER PRIMARY KEY,
                document_id TEXT,
                chunk_text TEXT,
                start_index INTEGER,
                end_index INTEGER,
                chunk_length INTEGER,
                metadata TEXT,
                date_added TEXT,
                FOREIGN KEY (document_id) 
                    REFERENCES documents_metadata(document_id)
            )
            ''')
            
            self.conn.commit()
            self.logger.debug("Chunks table created/verified", 
                            extra={'operation': 'table_creation'})
                            
        except sqlite3.Error as e:
            self.logger.error("Failed to create chunks table", 
                            extra={
                                'operation': 'table_creation',
                                'error': str(e)
                            })
            raise e

    def generate_chunk_id(self) -> str:
        chunk_id = uuid.uuid4().hex
        self.logger.debug("Generated chunk ID", 
                         extra={
                             'operation': 'id_generation',
                             'chunk_id': chunk_id
                         })
        return chunk_id

    def get_hashed_id(self, chunk_id: str) -> int:
        hashed = int(hashlib.sha256(chunk_id.encode()).hexdigest(), 16) % (2**63 - 1)
        self.logger.debug("Generated hashed ID", 
                         extra={
                             'operation': 'id_generation',
                             'chunk_id': chunk_id,
                             'hashed_id': hashed
                         })
        return hashed