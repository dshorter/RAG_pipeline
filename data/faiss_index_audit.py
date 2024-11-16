import sqlite3
import faiss
import numpy as np
import hashlib
from pathlib import Path

def get_hashed_id(chunk_id: str) -> int:
    """Generate a stable integer hash for FAISS indexing."""
    return int(hashlib.sha256(str(chunk_id).encode()).hexdigest(), 16) % (2**63 - 1)

def audit_indexes(db_path: str, faiss_path: str):
    """
    Audit FAISS and SQLite for consistency.
    """
    print("\n=== Starting Index Audit ===")
    
    try:
        # Get SQLite chunk IDs
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT chunk_id FROM document_chunks_metadata")
        sqlite_chunk_ids = set(get_hashed_id(row[0]) for row in cursor.fetchall())
        
        print(f"\nSQLite Statistics:")
        print(f"Total chunks in SQLite: {len(sqlite_chunk_ids)}")
        
        # Get FAISS IDs
        index = faiss.read_index(faiss_path)
        faiss_ids = set(int(id_) for id_ in faiss.vector_to_array(index.id_map))
        
        print(f"\nFAISS Statistics:")
        print(f"Total vectors in FAISS: {len(faiss_ids)}")
        
        # Find mismatches
        orphaned_vectors = faiss_ids - sqlite_chunk_ids
        missing_vectors = sqlite_chunk_ids - faiss_ids
        
        print("\nAudit Results:")
        print(f"Orphaned vectors in FAISS: {len(orphaned_vectors)}")
        print(f"Missing vectors for SQLite chunks: {len(missing_vectors)}")
        
        if orphaned_vectors:
            print("\nOrphaned Vector IDs:")
            for id_ in sorted(orphaned_vectors)[:10]:  # Show first 10
                print(f"- {id_}")
            if len(orphaned_vectors) > 10:
                print(f"... and {len(orphaned_vectors) - 10} more")
                
        if missing_vectors:
            print("\nChunks Missing Vectors:")
            for id_ in sorted(missing_vectors)[:10]:  # Show first 10
                print(f"- {id_}")
            if len(missing_vectors) > 10:
                print(f"... and {len(missing_vectors) - 10} more")
        
        # Option to clean up orphaned vectors
        if orphaned_vectors and input("\nWould you like to remove orphaned vectors? (y/n): ").lower() == 'y':
            print("\nCreating new clean index...")
            new_index = faiss.IndexIDMap2(faiss.IndexFlatL2(index.d))
            vectors_transferred = 0
            
            # Get all IDs from the index
            all_ids = faiss.vector_to_array(index.id_map)
            
            # Process vectors in batches
            batch_size = 1000
            for start_idx in range(0, len(all_ids), batch_size):
                end_idx = min(start_idx + batch_size, len(all_ids))
                batch_ids = all_ids[start_idx:end_idx]
                
                # Filter out orphaned IDs
                keep_mask = np.array([id_val not in orphaned_vectors for id_val in batch_ids])
                keep_ids = batch_ids[keep_mask]
                
                if len(keep_ids) > 0:
                    # Get vectors for the IDs we're keeping
                    vectors = np.vstack([
                        index.reconstruct(int(id_val)) 
                        for id_val in keep_ids
                    ])
                    
                    # Add to new index
                    new_index.add_with_ids(
                        vectors.astype(np.float32),
                        keep_ids.astype(np.int64)
                    )
                    vectors_transferred += len(keep_ids)
            
            # Save new index
            faiss.write_index(new_index, faiss_path)
            print(f"\nCleaned index saved:")
            print(f"- Original vectors: {len(faiss_ids)}")
            print(f"- Vectors kept: {vectors_transferred}")
            print(f"- Vectors removed: {len(faiss_ids) - vectors_transferred}")
            
    except Exception as e:
        print(f"\nError during audit: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

def main():
    # Get paths
    project_root = Path(__file__).parent.parent
    db_path = project_root / 'data' / 'metadata.db'
    faiss_path = project_root / 'data' / 'faiss_index.bin'
    
    print(f"Database path: {db_path}")
    print(f"FAISS index path: {faiss_path}")
    
    if not all(p.exists() for p in [db_path, faiss_path]):
        print("One or more required files not found!")
        return
        
    audit_indexes(str(db_path), str(faiss_path))

if __name__ == "__main__":
    main()