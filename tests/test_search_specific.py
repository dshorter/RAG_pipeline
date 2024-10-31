# scripts/test_search_specific.py

import sys
import os
import numpy as np
from datetime import datetime
import logging

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_search_client import RAGSearchClient
from src.logging_config import setup_rag_logging, get_logger

def test_specific_scenarios():
    logger = setup_rag_logging(log_dir='logs', unified_log=True)
    test_logger = get_logger('search_specific_test')
    
    client = RAGSearchClient()
    
    def test_edge_cases():
        test_logger.info("Testing edge cases")
        
        # Test with different k values
        for k in [1, 5, 10]:
            test_logger.info(f"Testing search with k={k}")
            test_vector = np.random.rand(1536).astype(np.float32)
            results = client.search(test_vector, k=k)
            test_logger.info(f"Retrieved {len(results)} results for k={k}")

        # Test with vector of ones
        test_logger.info("Testing search with vector of ones")
        ones_vector = np.ones(1536, dtype=np.float32)
        results = client.search(ones_vector, k=1)
        test_logger.info("Successfully tested search with ones vector")

        # Test with normalized vector
        test_logger.info("Testing search with normalized vector")
        norm_vector = np.random.rand(1536).astype(np.float32)
        norm_vector /= np.linalg.norm(norm_vector)
        results = client.search(norm_vector, k=1)
        test_logger.info("Successfully tested search with normalized vector")

    def test_metadata_retrieval():
        test_logger.info("Testing metadata retrieval")
        
        # Get stats first
        stats = client.get_index_stats()
        test_logger.info("Retrieved index stats", extra={'stats': stats})
        
        # Get first search result and test metadata
        test_vector = np.random.rand(1536).astype(np.float32)
        results = client.search(test_vector, k=1)
        
        if results:
            doc_id = results[0]['document_id']
            metadata = client.get_document_metadata(doc_id)
            test_logger.info("Retrieved document metadata", extra={'metadata': metadata})
            
            # Test citation formatting
            citation = results[0]['citation']
            test_logger.info("Citation format test", extra={'citation': citation})

    try:
        test_logger.info("Starting specific scenario tests")
        
        # Run tests
        test_edge_cases()
        test_metadata_retrieval()
        
        test_logger.info("All specific scenario tests completed successfully")
        return True
        
    except Exception as e:
        test_logger.error(f"Specific scenario tests failed: {str(e)}")
        raise

def main():
    try:
        success = test_specific_scenarios()
        if success:
            print("\nSpecific scenario tests completed successfully!")
            print("Check the logs for detailed information.")
    except Exception as e:
        print(f"\nSpecific scenario tests failed: {str(e)}")
        print("Check the logs for error details.")

if __name__ == "__main__":
    main()