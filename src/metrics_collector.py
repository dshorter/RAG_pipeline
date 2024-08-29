import csv
from datetime import datetime
import os

class MetricsCollector:
    def __init__(self, csv_file='rag_metrics.csv'):
        self.csv_file = csv_file
        self.ensure_csv_exists()

    def ensure_csv_exists(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['timestamp', 'document', 'total_tokens', 'num_chunks', 
                                 'avg_chunk_size', 'max_chunk_size', 'min_chunk_size', 
                                 'tokenization_time', 'chunking_time'])

    def log_metrics(self, metrics):
        metrics = {'timestamp': datetime.now().isoformat(), **metrics}
        
        with open(self.csv_file, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=metrics.keys())
            
            # If the file is empty (no header), write the header
            if file.tell() == 0:
                writer.writeheader()
            
            writer.writerow(metrics)
    def __init__(self, csv_file='rag_metrics.csv'):
        self.csv_file = csv_file
        self.ensure_csv_exists()

    def ensure_csv_exists(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['timestamp', 'document', 'total_tokens', 'num_chunks', 
                                 'avg_chunk_size', 'max_chunk_size', 'min_chunk_size', 
                                 'tokenization_time', 'chunking_time'])

    def log_metrics(self, metrics):
        metrics['timestamp'] = datetime.now().isoformat()
        with open(self.csv_file, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=metrics.keys())
            writer.writerow(metrics)

# Usage in RAG pipeline
metrics_collector = MetricsCollector()

# After processing a document
metrics = {
    'document': 'Biosafety_Guidance.txt',
    'total_tokens': 6214,
    'num_chunks': 15,
    'avg_chunk_size': 414,
    'max_chunk_size': 500,
    'min_chunk_size': 214,
    'tokenization_time': 0.05,
    'chunking_time': 0.02
}
metrics_collector.log_metrics(metrics)