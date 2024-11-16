import sqlite3
import faiss
import os
import sys
import numpy as np
from pathlib import Path
import hashlib
import shutil
from datetime import datetime

def get_hashed_id(chunk_id: str) -> int:
    """Generate a stable integer hash for FAISS indexing."""
    return int(hashlib.sha256(str(chunk_id).encode()).hexdigest(), 16) % (2**63 - 1)

def backup_files(db_path: str, faiss_path: str) -> str:
    """Create backup before we touch anything."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f"backup_{timestamp}")
    backup_dir.mkdir(exist_ok=True)
    
    print(f"\nCreating backups in {backup_dir}")
    shutil.copy2(db_path, backup_dir / "metadata.db.bak")
    shutil.copy2(faiss_path, backup_dir / "faiss_index.bin.bak")
    
    return str(backup_dir)

def cleanup_documents(db_path: str, faiss_path: str, dry_run: bool = False):
    """
    Remove test/tika documents WITHOUT nuking the FAISS index 😅
    """
    print("\n=== Starting Safe Document Cleanup ===")
    
    if not dry_run:
        backup_dir = backup_files(db_path, faiss_path)
        print(f"Backups created in: {backup_dir}")
        print("(Just in case... we learned our lesson! 😅)")
    
    try:
        # Connect to SQLite
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Find documents to remove
        cursor.execute("""
            SELECT document_id, title, author, source
            FROM documents_metadata
            WHERE title LIKE 'Test Document%'
               OR LOWER(title) LIKE '%tika%'
               OR LOWER(author) LIKE '%tika%'              
               OR LOWER(source) LIKE '%tika%'
        """)
        documents = cursor.fetchall()
        
        if not documents:
            print("No documents found matching removal criteria")
            return
            
        # Display what we found
        print(f"\nFound {len(documents)} documents to remove:")
        for doc_id, title, author, source in documents:
            print(f"\nDocument: {title}")
            print(f"Author: {author}")
            print(f"Source: {source}")
            print(f"ID: {doc_id}")
        
        if dry_run:
            print("\nDRY RUN - No changes will be made")
            return
            
        # Get confirmation
        response = input("\nProceed with removal? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled")
            return
        
        # Get chunks for these documents
        doc_ids = [doc[0] for doc in documents]
        placeholders = ','.join('?' * len(doc_ids))
        
        cursor.execute(f"""
            SELECT chunk_id, document_id
            FROM document_chunks_metadata
            WHERE document_id IN ({placeholders})
        """, doc_ids)
        chunks = cursor.fetchall()
        
        if chunks:
            print(f"\nFound {len(chunks)} chunks to remove")
            
            # Get FAISS IDs for these chunks
            chunk_faiss_ids = set(get_hashed_id(chunk[0]) for chunk in chunks)
            
            # Handle FAISS vectors - THE SAFE WAY THIS TIME! 
            if os.path.exists(faiss_path):
                print("\nUpdating FAISS index...")
                index = faiss.read_index(faiss_path)
                original_count = index.ntotal
                
                # Create new index with same properties
                new_index = faiss.IndexIDMap2(faiss.IndexFlatL2(index.d))
                transferred = 0
                
                # Get all existing IDs
                all_ids = faiss.vector_to_array(index.id_map)
                print(f"Processing {len(all_ids)} total vectors...")
                
                # Transfer vectors in batches
                batch_size = 100
                for start_idx in range(0, len(all_ids), batch_size):
                    batch_ids = all_ids[start_idx:min(start_idx + batch_size, len(all_ids))]
                    keep_mask = np.array([int(id_) not in chunk_faiss_ids for id_ in batch_ids])
                    keep_ids = batch_ids[keep_mask]
                    
                    if len(keep_ids) > 0:
                        # Get vectors we're keeping
                        try:
                            vectors = np.vstack([
                                index.reconstruct(int(id_)) for id_ in keep_ids
                            ])
                            
                            # Add to new index
                            new_index.add_with_ids(
                                vectors.astype(np.float32),
                                keep_ids.astype(np.int64)
                            )
                            transferred += len(keep_ids)
                            
                            if transferred % 1000 == 0:
                                print(f"Transferred {transferred} vectors...")
                                
                        except Exception as e:
                            print(f"Warning: Error processing batch: {e}")
                
                # Save updated index
                faiss.write_index(new_index, faiss_path)
                print(f"\nFAISS index updated:")
                print(f"- Original vectors: {original_count}")
                print(f"- Vectors kept: {transferred}")
                print(f"- Vectors removed: {original_count - transferred}")
            
            # Remove SQLite records
            print("\nUpdating SQLite database...")
            cursor.execute("BEGIN")
            
            try:
                # Remove chunks
                cursor.execute(f"""
                    DELETE FROM document_chunks_metadata 
                    WHERE document_id IN ({placeholders})
                """, doc_ids)
                chunks_deleted = cursor.rowcount
                
                # Remove documents
                cursor.execute(f"""
                    DELETE FROM documents_metadata 
                    WHERE document_id IN ({placeholders})
                """, doc_ids)
                docs_deleted = cursor.rowcount
                
                cursor.execute("COMMIT")
                
                print(f"\nSQLite cleanup complete:")
                print(f"- Documents removed: {docs_deleted}")
                print(f"- Chunks removed: {chunks_deleted}")
            
            except Exception as e:
                cursor.execute("ROLLBACK")
                print(f"Error updating SQLite: {e}")
                print("Rolling back changes...")
                raise
        
        print("\nCleanup completed successfully!")
        print("And this time we kept your vectors safe 😅")
        
    except Exception as e:
        print(f"\nError during cleanup: {e}")
        if not dry_run:
            print(f"\nDon't panic! Your backups are in: {backup_dir}")
    finally:
        if 'conn' in locals():
            conn.close()

def main():
    project_root = Path(__file__).parent.parent
    db_path = project_root / 'data' / 'metadata.db'
    faiss_path = project_root / 'data' / 'faiss_index.bin'
    
    print(f"Database path: {db_path}")
    print(f"FAISS index path: {faiss_path}")
    
    if not all(p.exists() for p in [db_path, faiss_path]):
        print("One or more required files not found!")
        return
    
    # First do a dry run
    print("\nPerforming dry run to show what would be removed...")
    cleanup_documents(str(db_path), str(faiss_path), dry_run=True)
    
    # If user wants to proceed, do the actual cleanup
    response = input("\nWould you like to proceed with actual removal? (y/n): ")
    if response.lower() == 'y':
        cleanup_documents(str(db_path), str(faiss_path), dry_run=False)
    else:
        print("Operation cancelled")

if __name__ == "__main__":
    main()