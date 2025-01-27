# src/metrics_collector.py

import logging
from typing import Dict, Any
from datetime import datetime
import sqlite3
import json
import uuid
from pathlib import Path    
from .paths import *     


class MetricsCollector:
    def __init__(self, db_path: str = ""):
        self.db_path =  get_db_path( )  
        self.logger = logging.getLogger('metrics')
        self._ensure_database()

    def _ensure_database(self):
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
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
                
                conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON metrics(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_operation ON metrics(operation)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk ON metrics(chunk_id)")

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS metric_failures (
                        failure_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        operation TEXT NOT NULL,
                        error_message TEXT NOT NULL
                    )
                """)
        except Exception as e:
            self.logger.error(f"Failed to initialize metrics database: {e}")

    def collect(self, operation: str, component: str, metrics: Dict[str, Any]):
        """Fire and forget metrics collection."""
        try:
            metric_data = {
                'metric_id': str(uuid.uuid4()),
                'timestamp': datetime.utcnow().isoformat(),
                'operation': operation,
                'component': component,
                'success': metrics.get('success', True),
                'chunk_id': metrics.get('chunk_id'),
                'document_id': metrics.get('document_id'),
                'metrics_data': json.dumps(metrics)
            }

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO metrics (
                        metric_id, timestamp, operation, component,
                        success, chunk_id, document_id, metrics_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, tuple(metric_data.values()))

        except Exception as e:
            self.logger.warning(f"Failed to collect metrics for {operation}: {e}")
            self._track_collection_failure(operation, e)

    def _track_collection_failure(self, operation: str, error: Exception):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO metric_failures (
                        failure_id, timestamp, operation, error_message
                    ) VALUES (?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()),
                    datetime.utcnow().isoformat(),
                    operation,
                    str(error)
                ))
        except Exception as e:
            self.logger.error(f"Failed to track metrics failure: {e}")
