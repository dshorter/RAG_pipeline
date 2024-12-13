# Add to metrics_collector.py

import sqlite3
import os 
import sys  
from datetime import datetime
from typing import Dict, Any, List, Optional
import json

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.logging_config import setup_rag_logging, get_logger
from src.data_classes import QueryAnalysisResult

class ChatHistoryStore:
    def __init__(self, db_path: str = 'rag_history.db'):
        
        if db_path is None:
        # Use data directory from paths.py
            from src.paths import get_data_dir
            self.db_path = os.path.join(get_data_dir(), 'rag_history.db')
        else:
            self.db_path = db_path    

        self.ensure_tables_exist()

    def ensure_tables_exist(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    metrics TEXT,  -- JSON string of metrics
                    processing_time REAL,
                    num_chunks INTEGER,
                    relevance_scores TEXT,  -- JSON string of scores
                    reranking_applied BOOLEAN,
                    training_data_used BOOLEAN,
                    source_chunks TEXT,  -- JSON string of source information
                    error TEXT  -- Any error messages
                )
            ''')
            conn.commit()
        finally:
            conn.close()

def log_interaction(self, query: str, response: Dict[str, Any], results: Dict[str, Any], processing_time: float):
    try:
        # Extract analysis data safely
        analysis = results.get('analysis')
        if isinstance(analysis, QueryAnalysisResult): # analysis is a QueryAnalysisResult object
            analysis_data = {
                'complexity_score': getattr(analysis, 'complexity_score', 0.0),
                'needs_reranking': getattr(analysis, 'needs_reranking', False),
                'recommended_k': getattr(analysis, 'recommended_k', 0),
                'recommended_candidates': getattr(analysis, 'recommended_candidates', 0)
            }
        else:
            analysis_data = {}

        # Extract source chunks and scores
        source_chunks = []
        relevance_scores = []
        for result in results.get('results', []):
            source_chunks.append({
                'chunk_id': result.get('chunk_id'),
                'document_id': result.get('document_id'),
                'source_info': result.get('source_info', {})
            })
            relevance_scores.append({
                'chunk_id': result.get('chunk_id'),
                'score': result.get('relevance_score'),
                'initial_score': result.get('initial_score')
            })

            # Prepare data for storage
            interaction_data = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'response': json.dumps(response),
                'metrics': json.dumps({
                    'processing_time': processing_time,
                    'num_chunks': len(results.get('results', [])),
                    'analysis': analysis_data,
                    'used_training_data': response.get('used_training_data', False)
                }),
                'processing_time': processing_time,
                'num_chunks': len(results.get('results', [])),
                'relevance_scores': json.dumps(relevance_scores),
                'reranking_applied': analysis_data.get('needs_reranking', False),
                'training_data_used': response.get('used_training_data', False),
                'source_chunks': json.dumps(source_chunks),
                'error': results.get('error')
            }

            # Add logging
            logger = get_logger('chat_store')
            logger.info(f"Logging interaction - Query: {query[:50]}...")

            # Store in database
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO chat_history (
                        timestamp, query, response, metrics, processing_time,
                        num_chunks, relevance_scores, reranking_applied,
                        training_data_used, source_chunks, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    interaction_data['timestamp'],
                    interaction_data['query'],
                    interaction_data['response'],
                    interaction_data['metrics'],
                    interaction_data['processing_time'],
                    interaction_data['num_chunks'],
                    interaction_data['relevance_scores'],
                    interaction_data['reranking_applied'],
                    interaction_data['training_data_used'],
                    interaction_data['source_chunks'],
                    interaction_data['error']
                ))
                conn.commit()
                logger.info("Successfully logged interaction to database")
                
            except sqlite3.Error as e:
                logger.error(f"Database error while logging interaction: {str(e)}")
                raise
            finally:
                conn.close()

    except Exception as e:
        logger = get_logger('chat_store')
        logger.error(f"Error logging chat interaction: {str(e)}", exc_info=True)


    def get_recent_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve recent chat history with metrics."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, query, response, metrics, processing_time,
                       num_chunks, relevance_scores, reranking_applied,
                       training_data_used, source_chunks, error
                FROM chat_history
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    'timestamp': row[0],
                    'query': row[1],
                    'response': json.loads(row[2]),
                    'metrics': json.loads(row[3]),
                    'processing_time': row[4],
                    'num_chunks': row[5],
                    'relevance_scores': json.loads(row[6]),
                    'reranking_applied': row[7],
                    'training_data_used': row[8],
                    'source_chunks': json.loads(row[9]),
                    'error': row[10]
                })
            return history
        finally:
            conn.close()

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary metrics for all interactions."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_queries,
                    AVG(processing_time) as avg_processing_time,
                    AVG(num_chunks) as avg_chunks_used,
                    SUM(CASE WHEN reranking_applied THEN 1 ELSE 0 END) as reranking_count,
                    SUM(CASE WHEN training_data_used THEN 1 ELSE 0 END) as training_data_count,
                    SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as error_count
                FROM chat_history
            ''')
            
            row = cursor.fetchone()
            return {
                'total_queries': row[0],
                'avg_processing_time': row[1],
                'avg_chunks_used': row[2],
                'reranking_count': row[3],
                'training_data_count': row[4],
                'error_count': row[5]
            }
        finally:
            conn.close()