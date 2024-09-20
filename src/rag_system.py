import sqlite3
import faiss
import uuid
import hashlib
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any

class RAGSystem:
    def __init__(self, conn: sqlite3.Connection, index: faiss.IndexIDMap, 
                 faiss_index_path: str):
        self.conn = conn     
        self.index = index
        self.faiss_index_path = faiss_index_path  # Path to save the FAISS index to disk
        self.document_id = uuid.uuid4().hex

    def add_vector(self, chunk: str, vector: np.array, source: str, start_index: int, end_index: int, additional_metadata: Dict[str, Any] = {}):
        chunk_id = self.generate_chunk_id()
        hashed_id = self.get_hashed_id(chunk_id)
        document_id =  self.document_id    

        try:
            # Step 1: Handle SQLite transaction
            self.conn.execute('BEGIN')    

            self.create_documents_table( )
            self.create_document_chunks_table( )  

            self.insert_document_metadata(document_id, "Title", "Author", source, len(chunk), "Summary", "Tags", additional_metadata)
            self.insert_chunk_metadata(hashed_id, document_id, chunk, start_index, end_index, len(chunk), additional_metadata)
            self.conn.commit()

            # Step 2: Handle FAISS vector addition
            self.add_vector_to_faiss(vector, hashed_id)

            # Step 3: Save the FAISS index to disk after successful addition
            self.save_faiss_index()

        except sqlite3.Error as e:
            # Rollback in case of SQLite error
            self.conn.rollback()
            print(f"SQLite transaction failed: {e}")
            raise e
        except Exception as e:
            # Rollback SQLite if FAISS fails, since FAISS has no rollback
            print(f"FAISS operation failed, rolling back SQLite: {e}")
            self.conn.rollback()  # Rollback metadata insertion due to FAISS failure
            raise e

    def add_vector_to_faiss(self, vector: np.array, hashed_id: int):
        try:
            # Print the shape of the input vector (after wrapping in an extra array)
            print(f"Vector shape: {np.array([vector]).shape}")  # Will print (1, dimension)        
            # Print the dimensionality of the FAISS index
            print(f"FAISS index dimension: {self.index.d}")  # FAISS index dimension
            
            self.index.add_with_ids(np.array([vector]).astype('float32'), np.array([hashed_id], dtype='int64'))
        except Exception as e:
            print(f"FAISS operation failed: {e}")
            raise e

    def save_faiss_index(self):
        try:
            faiss.write_index(self.index, self.faiss_index_path)
            print(f"FAISS index saved to {self.faiss_index_path}")
        except Exception as e:
            print(f"Failed to save FAISS index to disk: {e}")
            raise e

    def insert_document_metadata(self, document_id: str, title: str, author: str, 
                                 source: str, document_length: int, summary: str, tags: str, metadata: Dict[str, Any]):
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO documents_metadata (document_id, title, author, 
                           source, date_added, document_length, summary, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                document_id,
                title,
                author,
                source,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # SQLite-friendly date format
                document_length,
                summary,
                tags,
                json.dumps(metadata)  # Store metadata as JSON string
        ))            
        except sqlite3.IntegrityError as e:
            # Handle the unique constraint error, or log it and continue
            print(f"IntegrityError: {e} - Skipping this document (ID: {document_id})")

        except sqlite3.Error as e:
            # Handle any other database errors
            print(f"SQLite error: {e}")
            raise e  # Optionally, re-raise the error if you want it to bubble up



    def insert_chunk_metadata(self, chunk_id: int, document_id: str, chunk_text: str, 
                              start_index: int, end_index: int, chunk_length: int, 
                              additional_metadata: Dict[str, Any]):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO document_chunks_metadata (chunk_id, document_id, chunk_text, 
                       start_index, end_index, chunk_length, metadata, date_added)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            chunk_id,
            document_id,
            chunk_text,
            start_index,
            end_index,
            chunk_length,
            json.dumps(additional_metadata),  # Store metadata as JSON string
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # SQLite-friendly date format
        ))

    def generate_chunk_id(self) -> str:
        return uuid.uuid4().hex

    def get_hashed_id(self, chunk_id: str) -> int:
        return int(hashlib.sha256(chunk_id.encode()).hexdigest(), 16) % (2**63 - 1)
    
    def create_documents_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents_metadata (
            document_id  TEXT  PRIMARY KEY,
            title TEXT,
            author TEXT,
            source TEXT,
            date_added TEXT,  -- Storing dates as TEXT in ISO 8601 format
            document_length INTEGER,
            summary TEXT,
            tags TEXT,
            metadata TEXT  -- Store JSON as TEXT
        )
        ''')
        self.conn.commit()

    def create_document_chunks_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_chunks_metadata (
            chunk_id INTEGER PRIMARY KEY,
            document_id TEXT,
            chunk_text TEXT,
            start_index INTEGER,
            end_index INTEGER,
            chunk_length INTEGER,
            metadata TEXT,  -- Store JSON as TEXT
            date_added TEXT,  -- Storing dates as TEXT in ISO 8601 format
            FOREIGN KEY (document_id) REFERENCES documents_metadata(document_id)
        )
        ''')
        self.conn.commit()



