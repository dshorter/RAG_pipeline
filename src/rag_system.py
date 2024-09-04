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
        self.create_metadata_table()

    def create_metadata_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            uuid TEXT PRIMARY KEY,
            source TEXT,
            start_index INTEGER,
            end_index INTEGER,
            additional_metadata TEXT
        )
        ''')
        self.conn.commit()

    def add_chunk(self, chunk: str, vector: np.array, source: str, start_index: int, end_index: int, additional_metadata: Dict[str, Any] = {}):
        chunk_id = uuid.uuid4().hex
        
        # Add to FAISS index
        self.index.add_with_ids(np.array([vector]).astype('float32'), np.array([int(chunk_id, 16)]))
        
        # Add to SQLite metadata
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO metadata (uuid, source, start_index, end_index, additional_metadata)
        VALUES (?, ?, ?, ?, ?)
        ''', (chunk_id, source, start_index, end_index, json.dumps(additional_metadata)))
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
    rag_system.add_chunk(
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