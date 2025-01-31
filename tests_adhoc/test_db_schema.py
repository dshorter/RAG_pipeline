import os
import sys
import sqlite3
import unittest
from typing import Any, List, Dict
import json

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_system import RAGSystem

import logging


# Set up logging for better test visibility
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TestDatabaseSchema(unittest.TestCase):
    def setUp(self):
        """Create a fresh test database in a dedicated test directory."""
        # Create our test_data directory next to our test file
        test_dir = os.path.join(os.path.dirname(__file__), "test_data")
        logger.info(f"Setting up test environment in {test_dir}")
        os.makedirs(test_dir, exist_ok=True)
        
        # Set up test database path
        self.test_db_path = os.path.join(test_dir, "test_rag.db")
        logger.info(f"Using test database at {self.test_db_path}")
        
        # Remove existing test database if it exists
        if os.path.exists(self.test_db_path):
            logger.info("Removing existing test database")
            os.remove(self.test_db_path)
        
        # Initialize RAG system with test database
        logger.info("Initializing RAG system with test database")
        self.rag_system = RAGSystem(self.test_db_path)
        self.conn = sqlite3.connect(self.test_db_path)
        logger.info("Test setup complete")

    # def tearDown(self):
    #     """Clean up after tests complete."""
    #     logger.info("Starting test cleanup")
        
    #     # Close database connection
    #     if hasattr(self, 'conn') and self.conn:
    #         self.conn.close()
    #         logger.info("Closed database connection")
        
    #     # Remove test database
    #     if os.path.exists(self.test_db_path):
    #         os.remove(self.test_db_path)
    #         logger.info("Removed test database file")
        
    #     # Try to remove test directory
    #     test_dir = os.path.dirname(self.test_db_path)
    #     try:
    #         os.rmdir(test_dir)
    #         logger.info("Removed test directory")
    #     except OSError:
    #         logger.info(f"Note: Test directory {test_dir} not empty, leaving in place")

    def get_table_info(self, table_name: str) -> List[Dict[str, Any]]:
        """Retrieve the table info (columns and constraints) for the specified table."""
        cursor = self.conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        return [{'name': col[1], 'type': col[2], 'notnull': col[3], 'pk': col[5]} for col in columns]

    def get_table_indices(self, table_name: str) -> List[str]:
        """Get list of indices for a table."""
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?", 
                      (table_name,))
        return [row[0] for row in cursor.fetchall()]

    def get_index_on_column(self, table_name: str, column_name: str) -> List[str]:
        """Get indices that reference a specific column."""
        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT name 
            FROM sqlite_master 
            WHERE type='index' 
            AND tbl_name=? 
            AND sql LIKE ? 
        """, (table_name, f'%{column_name}%'))
        return [row[0] for row in cursor.fetchall()]

    def test_documents_metadata_table(self):
        """Test documents_metadata table structure."""
        columns = self.get_table_info('documents_metadata')
        
        expected_columns = {
            'document_id': {'type': 'TEXT', 'notnull': 1, 'pk': 1},
            'title': {'type': 'TEXT', 'notnull': 1},
            'author': {'type': 'TEXT', 'notnull': 1},
            'source': {'type': 'TEXT', 'notnull': 1},
            'date_added': {'type': 'TEXT', 'notnull': 1},
            'document_length': {'type': 'INTEGER', 'notnull': 1},
            'summary': {'type': 'TEXT', 'notnull': 1},
            'tags': {'type': 'TEXT', 'notnull': 1}
        }
        
        for col in columns:
            self.assertIn(col['name'], expected_columns)
            expected = expected_columns[col['name']]
            self.assertEqual(col['type'], expected['type'])
            self.assertEqual(col['notnull'], expected['notnull'])
            if 'pk' in expected:
                self.assertEqual(col['pk'], expected['pk'])

    def test_indices_existence(self):
        """Test that all required indices exist, including automatically created ones.
        
        SQLite creates automatic indices with names like 'sqlite_autoindex_*' for:
        - PRIMARY KEY (creates first auto-index)
        - UNIQUE constraints (creates subsequent auto-indices)
        """
        # Query SQLite's internal system table for all indices on our chunks table
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT name 
            FROM sqlite_master 
            WHERE type='index' AND tbl_name='document_chunks_metadata'
        """)
        
        # Get all index names as a list
        actual_indices = [row[0] for row in cursor.fetchall()]
        
        # Filter to just get the automatically created indices
        auto_indices = [idx for idx in actual_indices 
                    if idx.startswith('sqlite_autoindex')]
        
        # We expect exactly two automatic indices:
        # 1. One for the PRIMARY KEY on chunk_id
        # 2. One for the UNIQUE constraint on faiss_id
        self.assertEqual(
            len(auto_indices), 2,
            f"Expected 2 auto-indices but found {len(auto_indices)}: {auto_indices}"
        )
        
        # Verify that both indices are properly named for our table
        # They should follow the pattern: sqlite_autoindex_document_chunks_metadata_N
        for idx in auto_indices:
            self.assertTrue(
                idx.startswith('sqlite_autoindex_document_chunks_metadata'),
                f"Unexpected index name pattern: {idx}"
            )      


if __name__ == '__main__':
    unittest.main(verbosity=2)
