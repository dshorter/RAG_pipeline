# scripts/test_search_client.py

import sys
import os
import numpy as np
from datetime import datetime
import logging
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_search_client import RAGSearchClient
from src.logging_config import setup_rag_logging, get_logger
from src.singleton_config import ConfigSingleton

def test_search_client():
    # Set up logging
    logger = setup_rag_logging(log_dir='logs', unified_log=True)
    search_logger = get_logger('search_test')
    
    search_logger.info("Starting search client validation test")
    
    try:
        # Initialize search client
        search_logger.info("Initializing RAG Search Client")
        client = RAGSearchClient()
        
        # Get and log index statistics
        stats = client.get_index_stats()
        search_logger.info("Index Statistics:", extra={
            'component': 'search_test',
            'operation': 'index_stats',
            'total_vectors': stats['total_vectors'],
            'dimension': stats['dimension'],
            'total_chunks': stats['total_chunks'],
            'total_documents': stats['total_documents']
        })

        # Test search with a random vector
        search_logger.info("Testing search with random vector")
        test_vector = np.random.rand(1536).astype(np.float32)  # Using known dimension
        results = client.search(test_vector, k=3)
        
        search_logger.info(f"Search returned {len(results)} results")
        
        # Log detailed results
        for i, result in enumerate(results, 1):
            search_logger.info(f"Result {i}:", extra={
                'component': 'search_test',
                'operation': 'search_result',
                'chunk_id': result['chunk_id'],
                'document_id': result['document_id'],
                'relevance_score': result['relevance_score'],
                'citation': result['citation']
            })
            
            # Get and log document metadata
            doc_metadata = client.get_document_metadata(result['document_id'])
            if doc_metadata:
                search_logger.info(f"Document metadata for result {i}:", extra={
                    'component': 'search_test',
                    'operation': 'document_metadata',
                    'title': doc_metadata['title'],
                    'author': doc_metadata['author'],
                    'date_added': doc_metadata['date_added']
                })

        search_logger.info("Search client validation completed successfully")
        return True

    except Exception as e:
        search_logger.error(f"Search client validation failed: {str(e)}", extra={
            'component': 'search_test',
            'operation': 'error',
            'error_message': str(e)
        })
        raise

def main():
    try:
        success = test_search_client()
        if success:
            print("\nSearch client validation completed successfully!")
            print("Check the logs for detailed information.")
    except Exception as e:
        print(f"\nSearch client validation failed: {str(e)}")
        print("Check the logs for error details.")

if __name__ == "__main__":
    main()