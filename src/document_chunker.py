import logging
import time
from typing import Dict
from typing import List, Dict, Any  
from nltk.tokenize import sent_tokenize, word_tokenize  
from .metrics_collector import  MetricsCollector   
import nltk

# Download required NLTK data
nltk.download('punkt', quiet=True)

logger = logging.getLogger(__name__)

def chunk_document(content: str, chunk_size: int = 500, chunk_overlap: int = 50) -> Dict[str, Any]:
    start_time = time.time()
    try:
        # Use NLTK for sentence tokenization
        sentences = sent_tokenize(content)
        words = [word for sentence in sentences for word in word_tokenize(sentence)]
        
        chunks = []
        start = 0

        while start < len(words):
            end = start + chunk_size
            chunk_text = ' '.join(words[start:end])
            chunk = {
                'text': chunk_text,
                'start_index': start,
                'end_index': min(end, len(words))
            }
            chunks.append(chunk)
            start = end - chunk_overlap

        metrics = {
            'duration_ms': (time.time() - start_time) * 1000,
            'total_words': len(words),
            'total_sentences': len(sentences),
            'num_chunks': len(chunks),
            'avg_chunk_size': sum(len(chunk['text']) for chunk in chunks) / len(chunks),
            'max_chunk_size': max(len(chunk['text']) for chunk in chunks),
            'min_chunk_size': min(len(chunk['text']) for chunk in chunks),
            'overlap_size': chunk_overlap
        }

        MetricsCollector(self.db_path).collect(
            operation='document_chunking',
            component='document_chunker',
            metrics=metrics
        )

        return {'chunks': chunks, 'metrics': metrics}

    except Exception as e:
        logger.error(f"Chunking failed: {str(e)}")
        MetricsCollector(self.db_path).collect(
            operation='document_chunking',
            component='document_chunker',
            metrics={
                'duration_ms': (time.time() - start_time) * 1000,
                'success': False,
                'error': str(e)
            }
        )
        raise

# Example usage
if __name__ == "__main__":
    sample_text = "This is a sample text. It contains multiple sentences. " * 100
    chunks = chunk_document(sample_text)
    for i, chunk in enumerate(chunks['chunks']):
        print(f"Chunk {i + 1}: {chunk['text'][:50]}...")


