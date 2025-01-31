import hashlib
import sqlite3
import faiss
import numpy as np
from typing import Dict, Tuple  

import os, sys 

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.singleton_config import ConfigSingleton
from src.paths import get_db_path, get_faiss_path
import logging
from src.logging_config import get_logger

def verify_faiss_sqlite_alignment() -> Dict:
    """Verify alignment between FAISS vectors and SQLite chunks at document level."""
    logger = get_logger('system')
    config = ConfigSingleton()
    
    faiss_path = get_faiss_path()
    db_path = get_db_path()
    
    logger.info("Starting detailed FAISS-SQLite verification", 
                extra={
                    'component': 'verification',
                    'operation': 'alignment_check'
                })
    
    try:
        # Load FAISS index
        index = faiss.read_index(faiss_path)
        total_vectors = index.ntotal
        
        # Get SQLite document and chunk data
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get detailed document and chunk data
        cursor.execute("""
            SELECT 
                d.document_id,
                d.title,
                c.chunk_id,
                c.chunk_text
            FROM documents_metadata d
            LEFT JOIN document_chunks_metadata c ON d.document_id = c.document_id
            ORDER BY d.document_id, c.chunk_id
        """)
        
        # Organize by document
        documents = {}
        for doc_id, title, chunk_id, chunk_text in cursor.fetchall():
            if doc_id not in documents:
                documents[doc_id] = {
                    'title': title,
                    'chunks': [],
                    'chunk_count': 0,
                    'vector_count': 0
                }
            if chunk_id:  # Some documents might have no chunks
                documents[doc_id]['chunks'].append(chunk_id)
                documents[doc_id]['chunk_count'] += 1

        # Compare with FAISS vectors
        vector_mismatches = []
        for doc_id, doc_info in documents.items():
            vectors_found = 0
            for chunk_id in doc_info['chunks']:
                try:
                    # Convert chunk_id to FAISS ID using your hash function
                    faiss_id = int(hashlib.sha256(chunk_id.encode()).hexdigest(), 16) % (2**63 - 1)
                    # Try to reconstruct vector (will raise error if not found)
                    vector = index.reconstruct(int(faiss_id))
                    vectors_found += 1
                except Exception as e:
                    vector_mismatches.append({
                        'document': doc_info['title'],
                        'chunk_id': chunk_id,
                        'error': str(e)
                    })
            
            doc_info['vector_count'] = vectors_found
        
        # Prepare detailed report
        results = {
            "total_summary": {
                "faiss_total_vectors": total_vectors,
                "sqlite_total_chunks": sum(d['chunk_count'] for d in documents.values()),
                "total_documents": len(documents)
            },
            "per_document": {
                doc_id: {
                    'title': info['title'],
                    'chunks': info['chunk_count'],
                    'vectors': info['vector_count'],
                    'aligned': info['chunk_count'] == info['vector_count']
                }
                for doc_id, info in documents.items()
            },
            "mismatches": vector_mismatches
        }
        
        # Print detailed report
        print("\n🔍 FAISS-SQLite Alignment Report")
        print("==============================")
        print(f"\n📊 Overall Statistics:")
        print(f"  📑 Total Documents: {results['total_summary']['total_documents']}")
        print(f"  🧩 Total Chunks: {results['total_summary']['sqlite_total_chunks']}")
        print(f"  🎯 Total Vectors: {results['total_summary']['faiss_total_vectors']}")
        
        print("\n📚 Per-Document Analysis:")
        for doc_id, info in results['per_document'].items():
            alignment = '✅' if info['aligned'] else '❌'
            print(f"\n  {alignment} {info['title']}")
            print(f"     Chunks: {info['chunks']}")
            print(f"     Vectors: {info['vectors']}")
            if not info['aligned']:
                print(f"     ⚠️  Difference: {abs(info['chunks'] - info['vectors'])}")
        
        if vector_mismatches:
            print("\n⚠️ Vector Mismatches Found:")
            for mismatch in vector_mismatches:
                print(f"  • {mismatch['document']}: Chunk {mismatch['chunk_id']}")
                print(f"    Error: {mismatch['error']}")
        
        conn.close()     
        
        
        return results
        
    except Exception as e:
        logger.error("Verification failed", 
                    extra={
                        'component': 'verification',
                        'operation': 'alignment_check',
                        'error': str(e)
                    })
        raise

if __name__ == "__main__":
    try:
        results = verify_faiss_sqlite_alignment()
        
        # Summary status
        all_aligned = all(doc['aligned'] for doc in results['per_document'].values())
        if all_aligned and not results['mismatches']:
            print("\n✅ Complete system alignment verified!")
        else:
            print("\n⚠️ Alignment issues detected!")
            print(f"   Documents with mismatches: {sum(1 for doc in results['per_document'].values() if not doc['aligned'])}")
            
    except Exception as e:
        print(f"\n❌ Verification failed: {str(e)}")