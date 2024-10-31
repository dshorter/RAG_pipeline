# tests/test_rag_search.py

import unittest
import numpy as np     
import sys            
import os
import tempfile
import json
from datetime import datetime     

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_search_client  import RAGSearchClient
from src.rag_system import RAGSystem
from src.singleton_config import ConfigSingleton
from src.paths import get_db_path, get_faiss_path

class TestRAGSearchClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        cls.config = ConfigSingleton()
        cls.original_vectors_count = None
        
        # First, get the current state of the index
        try:
            client = RAGSearchClient()
            stats = client.get_index_stats()
            cls.original_vectors_count = stats['total_vectors']
            print(f"\nFound existing index with {cls.original_vectors_count} vectors")
        except Exception as e:
            print(f"\nNo existing index found or error reading it: {e}")
        
        # Create test vectors
        cls.test_vectors = []
        cls.test_chunks = []
        cls.test_metadata = []
        
        # Create test vectors (as before)
        base_vector = np.random.rand(1536).astype(np.float32)
        cls.test_vectors.append(base_vector)
        
        for i in range(4):
            noise = np.random.rand(1536).astype(np.float32) * 0.1
            related_vector = base_vector + noise
            related_vector /= np.linalg.norm(related_vector)
            cls.test_vectors.append(related_vector)
        
        # Create corresponding test chunks
        for i in range(5):
            cls.test_chunks.append({
                "text": f"Test chunk {i + 1} with specific content for testing.",
                "metadata": {
                    "source": f"test_doc_{i}",
                    "page": i + 1
                }
            })
        
        # Add test documents to existing index
        cls.rag_system = RAGSystem()
        cls.test_doc_ids = []
        
        for i, (vector, chunk) in enumerate(zip(cls.test_vectors, cls.test_chunks)):
            doc_id = f"test_doc_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}"
            cls.test_doc_ids.append(doc_id)
            
            doc_metadata = {
                "title": f"Test Document {i}",
                "author": f"Test Author {i}",
                "source": f"Test Source {i}",
                "date_added": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            cls.rag_system.add_vector(
                document_id=doc_id,
                chunk=chunk["text"],
                vector=vector,
                source=doc_metadata["source"],
                start_index=0,
                end_index=len(chunk["text"]),
                additional_metadata=chunk["metadata"],
                doc_metadata=doc_metadata
            )
            
            cls.test_metadata.append(doc_metadata)

    def setUp(self):
        """Set up for each test"""
        self.search_client = RAGSearchClient()

    def test_basic_search(self):
        """Test basic search functionality with known vector"""
        results = self.search_client.search(self.test_vectors[0], k=3)
        
        self.assertGreaterEqual(len(results), 1, "Should return at least one result")
        self.assertTrue(all(isinstance(r, dict) for r in results), 
                       "All results should be dictionaries")

    def test_search_with_metadata(self):
        """Test that search results include correct metadata"""
        results = self.search_client.search(self.test_vectors[0], k=1)
        result = results[0]
        
        required_fields = [
            "chunk_id", "chunk_text", "document_id", "metadata",
            "vector_stats", "relevance_score", "citation", "source_info"
        ]
        
        for field in required_fields:
            self.assertIn(field, result, f"Result should contain {field}")

    def test_index_stats(self):
        """Test index statistics retrieval"""
        stats = self.search_client.get_index_stats()
        
        self.assertIn('total_vectors', stats)
        self.assertIn('dimension', stats)
        self.assertEqual(stats['dimension'], 1536)
        
        # If we know the original count, verify new count
        if self.original_vectors_count is not None:
            expected_total = self.original_vectors_count + len(self.test_vectors)
            self.assertEqual(stats['total_vectors'], expected_total,
                           f"Expected {expected_total} vectors (original: {self.original_vectors_count} + test: {len(self.test_vectors)})")

    @classmethod
    def tearDownClass(cls):
        """Clean up test data."""
        print("\nTest statistics:")
        print(f"- Original vectors in index: {cls.original_vectors_count}")
        print(f"- Test vectors added: {len(cls.test_vectors)}")
        print(f"- Test document IDs: {cls.test_doc_ids}")

if __name__ == '__main__':
    unittest.main(verbosity=2)