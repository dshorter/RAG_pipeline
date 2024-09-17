from datetime import datetime
import hashlib
import uuid
import sqlite3
from typing import List, Dict, Any
import faiss
import numpy as np
import json
import os

class RAGSystem:
    def __init__(self, vector_dimension: int, db_path: str = 'metadata.db'):
        self.vector_dimension = vector_dimension
        self.db_path = db_path
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(vector_dimension))
        self.conn = sqlite3.connect(db_path)
        self.create_documents_table()
        self.create_document_chunks_table()

    def create_documents_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents_metadata (
            document_id TEXT PRIMARY KEY,
            title TEXT,
            author TEXT,
            source TEXT,
            date_added TIMESTAMP,
            document_length INTEGER,
            summary TEXT,
            tags TEXT,
            metadata JSON
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
            metadata JSON,
            date_added TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents_metadata(document_id)
        )
        ''')
        self.conn.commit()

    def add_vector(self, chunk: str, vector: np.array, 
                  source: str, start_index: int, end_index: int,  
                  all_metrics: Dict[str, Dict[str,Any]] = {},   
                  additional_metadata: Dict[str, Any] = {}):
        
            chunk_id = uuid.uuid4().hex
            # Create a 64-bit hash from the UUID
            hashed_id = int(hashlib.sha256(chunk_id.encode()).hexdigest(), 16) % (2**63 - 1)  # Modulo to fit in 64-bit
            # Ensure the hashed ID is wrapped in a NumPy array of type int64
            id_array = np.array([hashed_id], dtype='int64')
            
            # Add to FAISS index
            self.index.add_with_ids(np.array([vector]).astype('float32'), id_array )  
            faiss.write_index(self.index, path)            
            
            # Method to insert document metadata into documents_metadata table
    def insert_document_metadata(self, document_id, title, author, source, document_length, summary, tags, metadata):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO documents_metadata (document_id, title, author, source, date_added, document_length, summary, tags, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            document_id,                      # document_id, unique identifier for the document
            title,                            # title of the document
            author,                           # author of the document
            source,                           # source or origin of the document
            datetime.now(),                   # current timestamp for date_added
            document_length,                  # length of the document (e.g., word count)
            summary,                          # brief summary of the document
            tags,                             # tags or keywords associated with the document
            json.dumps(metadata)              # additional metadata stored as JSON
        ))
        self.conn.commit()

    # Method to insert chunk metadata into document_chunks_metadata table
    def insert_chunk_metadata(self, hashed_id, document_id, chunk_text, start_index, end_index, chunk_length, additional_metadata):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO document_chunks_metadata (chunk_id, document_id, chunk_text, start_index, end_index, chunk_length, metadata, date_added)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            hashed_id,                        # chunk_id, hashed 64-bit integer ID
            document_id,                      # document_id, links to documents_metadata
            chunk_text,                       # the actual text content of the chunk
            start_index,                      # starting position of the chunk within the document
            end_index,                        # ending position of the chunk within the document
            chunk_length,                     # length of the chunk (e.g., word count or character count)
            json.dumps(additional_metadata),  # additional metadata stored as JSON
            datetime.now()                    # current timestamp for date_added
        ))
        self.conn.commit()


    def search(self, query_vector: np.array, k: int = 5) -> List[Dict[str, Any]]:
        distances, indices = self.index.search(np.array([query_vector]).astype('float32'), k)
        
        results = []
        for idx in indices[0]:
            if idx != -1:  # -1 indicates no match found
                uuid_hex = hex(idx)[2:].zfill(32)  # Convert back to UUID hex string
                cursor = self.conn.cursor()
                cursor.execute('SELECT * FROM metadata WHERE uuid = ?', (uuid_hex,))
                result = cursor.fetchone()
                if result:
                    metadata = {
                        'uuid': result[0],
                        'source': result[1],
                        'start_index': result[2],
                        'end_index': result[3],
                        'additional_metadata': json.loads(result[4])
                    }
                    results.append(metadata)
        
        return results

    def save_index(self, path: str):
        faiss.write_index(self.index, path)

    def load_index(self, path: str):
        if os.path.exists(path):
            self.index = faiss.read_index(path)
        else:
            raise FileNotFoundError(f"Index file not found: {path}")

    def __del__(self):
        self.conn.close()

# Usage example
if __name__ == "__main__":
    rag_system = RAGSystem(vector_dimension=128)
    
    # Adding a chunk
    chunk = "This is a sample chunk of text."
    vector = np.random.rand(128).astype('float32')  # Simulated vector
    rag_system.add_vector(
        chunk, 
        vector, 
        source="Document A", 
        start_index=0, 
        end_index=len(chunk), 
        additional_metadata={"importance": "high", "author": "John Doe"}
    )
    
    # Searching
    query_vector = np.random.rand(128).astype('float32')  # Simulated query vector
    results = rag_system.search(query_vector)
    
    for result in results:
        print(f"Found chunk from {result['source']}, index {result['start_index']}:{result['end_index']}")
        print(f"Additional metadata: {result['additional_metadata']}")
        print(f"UUID: {result['uuid']}")
        print("---")

    # Save and load index example
    rag_system.save_index("faiss_index.bin")
    rag_system.load_index("faiss_index.bin")