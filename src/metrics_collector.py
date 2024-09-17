import csv
from datetime import datetime
import os
from typing import Dict, Any

class MetricsCollector:
    def __init__(self, csv_file='rag_metrics.csv'):
        self.csv_file = csv_file
        self.metrics = {}
        self.ensure_csv_exists()

    def ensure_csv_exists(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['timestamp', 'metric_type', 'metric_name', 'metric_value'])

    def log_metrics(self, metric_type: str, metrics: Dict[str, Any]):
        timestamp = datetime.now().isoformat()
        
        # Store metrics in memory
        if metric_type not in self.metrics:
            self.metrics[metric_type] = {}
        self.metrics[metric_type].update(metrics)
        
        # Log metrics to CSV
        with open(self.csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            for metric_name, metric_value in metrics.items():
                writer.writerow([timestamp, metric_type, metric_name, metric_value])

    def get_metrics(self, metric_type: str) -> Dict[str, Any]:
        """
        Retrieve metrics for a specific metric type.
        
        :param metric_type: The type of metrics to retrieve (e.g., "chunks", "embeddings")
        :return: A dictionary containing the metrics for the specified type
        """
        if metric_type not in self.metrics:
            return {}
        return self.metrics[metric_type]

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve all collected metrics.
        
        :return: A dictionary containing all metrics, organized by metric type
        """
        return self.metrics

# Usage example
if __name__ == "__main__":
    metrics_collector = MetricsCollector()
    
    # Log some example metrics
    chunk_metrics = {
        'total_chunks': 10,
        'avg_chunk_size': 500,
        'max_chunk_size': 550,
        'min_chunk_size': 450,
        'chunking_time': 0.5
    }
    # metrics_collector.log_metrics("chunks", chunk_metrics)
    
    embedding_metrics = {
        'num_embeddings': 10,
        'embedding_dimension': 768,
        'embedding_generation_time': 1.2
    }
    metrics_collector.log_metrics("embeddings", embedding_metrics)
    
    # Retrieve and print metrics
    print("Chunk metrics:", metrics_collector.get_metrics("chunks"))
    print("Embedding metrics:", metrics_collector.get_metrics("embeddings"))
    print("All metrics:", metrics_collector.get_all_metrics())