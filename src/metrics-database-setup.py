# src/setup_metrics_db.py

import sqlite3
import logging
from pathlib import Path

def setup_metrics_database(db_path: str):
    logger = logging.getLogger('metrics.setup')
    
    try:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(db_path) as conn:
            # Create main metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    metric_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    component TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    chunk_id TEXT,
                    document_id TEXT,
                    metrics_data JSON NOT NULL
                )
            """)
            
            # Create tracking table for failures
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metric_failures (
                    failure_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    error_message TEXT NOT NULL
                )
            """)
            
            # Create indexes
            conn.executescript("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON metrics(timestamp);
                CREATE INDEX IF NOT EXISTS idx_operation ON metrics(operation);
                CREATE INDEX IF NOT EXISTS idx_chunk ON metrics(chunk_id);
                CREATE INDEX IF NOT EXISTS idx_document ON metrics(document_id);
                CREATE INDEX IF NOT EXISTS idx_component ON metrics(component);
            """)
            
            # Create views
            conn.executescript(open('src/metrics_views.sql').read())
            
        logger.info(f"Metrics database initialized at {db_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup metrics database: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    setup_metrics_database("data/metrics.db")

