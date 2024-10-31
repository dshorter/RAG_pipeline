import argparse
import faiss
import numpy as np
import sqlite3
import os

def repair_faiss_index(index_path: str, db_path: str, backup: bool = False ):
    """Repair FAISS index and ID mapping."""
    
    # Backup original files if requested
    # if backup:
    #     if os.path.exists(index_path):
    #         os.rename(index_path, f"{index_path}.bak")
    #     if os.path.exists(db_path):
    #         os.rename(db_path, f"{db_path}.bak")
    
    # Create new index
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all chunks and their vectors
    cursor.execute('''
        SELECT c.chunk_id, c.chunk_text, m.faiss_id
        FROM document_chunks_metadata c
        JOIN chunk_id_mapping m ON c.chunk_id = m.chunk_id
    ''')
    chunks = cursor.fetchall()
    
    # Create new FAISS index
    dimension = 1536  # Standard for OpenAI embeddings
    new_index = faiss.IndexIDMap2(faiss.IndexFlatL2(dimension))
    
    # Process chunks
    processed = 0
    errors = 0
    
    for chunk_id, chunk_text, faiss_id in chunks:
        try:
            # Get vector from old index (if it exists)
            if os.path.exists(f"{index_path}.bak"):
                old_index = faiss.read_index(f"{index_path}.bak")
                try:
                    vector = old_index.reconstruct(int(faiss_id))
                except:
                    # If vector can't be retrieved, regenerate it
                    # You'll need to implement vector regeneration based on your embedding method
                    continue
                
                # Add to new index
                vector = vector.reshape(1, -1).astype(np.float32)
                new_index.add_with_ids(vector, np.array([int(faiss_id)], dtype=np.int64))
                processed += 1
                
        except Exception as e:
            print(f"Error processing chunk {chunk_id}: {str(e)}")
            errors += 1
    
    # Save new index
    faiss.write_index(new_index, index_path)
    
    return {
        'processed': processed,
        'errors': errors,
        'total_chunks': len(chunks)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FAISS Index Repair Tool')
    parser.add_argument('--index-path', required=True, help='Path to FAISS index')
    parser.add_argument('--db-path', required=True, help='Path to SQLite database')
    parser.add_argument('--no-backup', action='store_true', help='Skip backup of original files')
    
    args = parser.parse_args()
    
    results = repair_faiss_index(args.index_path, args.db_path, not args.no_backup)
    print(f"Repair completed: {results}")