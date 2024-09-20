import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sqlite3
import faiss
from src.rag_system import RAGSystem

# python -m unittest discover -s tests 

class TestRAGSystem(unittest.TestCase):

    def setUp(self):
        # Mock SQLite connection and FAISS index
        self.mock_conn = MagicMock(sqlite3.Connection)
        self.mock_faiss_index = MagicMock(faiss.IndexIDMap)

        # Instantiate the RAGSystem with mocked dependencies
        self.rag_system = RAGSystem(self.mock_conn, self.mock_faiss_index, faiss_index_path='/path/to/faiss.index')

    @patch('src.rag_system.datetime')
    @patch('src.rag_system.uuid.uuid4')
    @patch('faiss.write_index')
    def test_add_vector_successful(self, mock_write_index, mock_uuid4, mock_datetime_now):
        """Test that adding a vector successfully updates SQLite and FAISS, and saves FAISS index."""
        
        # Mock UUID and datetime
        mock_uuid4.return_value.hex = 'mock-uuid'
        mock_datetime_now.return_value = '2023-01-01'

        # Create a test vector
        test_vector = np.random.rand(128)

        # Call add_vector method
        self.rag_system.add_vector(
            chunk="Test chunk",
            vector=test_vector,
            source="Test source",
            start_index=0,
            end_index=100,
            additional_metadata={"key": "value"}
        )

        # Check if SQLite transaction was correctly handled
        self.mock_conn.execute.assert_called_with('BEGIN')
        self.mock_conn.commit.assert_called_once()

        # Ensure that FAISS add_with_ids and save index were called
        self.mock_faiss_index.add_with_ids.assert_called_once()
        mock_write_index.assert_called_once_with(self.mock_faiss_index, '/path/to/faiss.index')

    def test_add_vector_sqlite_failure(self):
        """Test that SQLite failure leads to a rollback and no FAISS operations are called."""
        
        # Simulate SQLite error
        self.mock_conn.execute.side_effect = sqlite3.Error("SQLite error")

        # Create a test vector
        test_vector = np.random.rand(128)

        # Check if the exception is raised and rollback is called
        with self.assertRaises(sqlite3.Error):
            self.rag_system.add_vector(
                chunk="Test chunk",
                vector=test_vector,
                source="Test source",
                start_index=0,
                end_index=100,
                additional_metadata={"key": "value"}
            )

        # Ensure rollback was called and FAISS was not touched
        self.mock_conn.rollback.assert_called_once()
        self.mock_faiss_index.add_with_ids.assert_not_called()

    @patch('faiss.write_index')
    def test_add_vector_faiss_failure(self, mock_write_index):
        """Test that FAISS failure triggers a rollback of the SQLite transaction."""
        
        # Simulate FAISS error
        self.mock_faiss_index.add_with_ids.side_effect = Exception("FAISS error")

        # Create a test vector
        test_vector = np.random.rand(128)

        # Check if the FAISS failure raises an exception
        with self.assertRaises(Exception):
            self.rag_system.add_vector(
                chunk="Test chunk",
                vector=test_vector,
                source="Test source",
                start_index=0,
                end_index=100,
                additional_metadata={"key": "value"}
            )

        # Ensure rollback was called on SQLite due to FAISS failure
        self.mock_conn.rollback.assert_called_once()

        # Ensure the FAISS save was not attempted after failure
        mock_write_index.assert_not_called()

    @patch('faiss.write_index')
    def test_save_faiss_failure(self, mock_write_index):
        """Test that failure to save FAISS index also leads to SQLite rollback."""
        
        # Simulate successful FAISS vector addition but failure to save FAISS index
        mock_write_index.side_effect = Exception("Failed to save FAISS index")

        # Create a test vector
        test_vector = np.random.rand(128)

        # Check if the FAISS save failure raises an exception
        with self.assertRaises(Exception):
            self.rag_system.add_vector(
                chunk="Test chunk",
                vector=test_vector,
                source="Test source",
                start_index=0,
                end_index=100,
                additional_metadata={"key": "value"}
            )

        # Ensure rollback was called on SQLite due to FAISS save failure
        self.mock_conn.rollback.assert_called_once()

if __name__ == '__main__':
    unittest.main()

