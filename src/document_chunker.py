import logging
import time
from typing import Dict
from typing import List, Dict, Any  

logger = logging.getLogger(__name__)

def chunk_document(content: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    logger.info("Chunking document")
    words = content.split()
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

    logger.info(f"Document chunked into {len(chunks)} parts")

    # Collect metrics
    metrics = {
        'total_words': len(words),
        'num_chunks': len(chunks),
        'avg_chunk_size': sum(len(chunk['text']) for chunk in chunks) / len(chunks),
        'max_chunk_size': max(len(chunk['text']) for chunk in chunks),
        'min_chunk_size': min(len(chunk['text']) for chunk in chunks)
        # 'tokenization_time': tokenization_time,
        # 'chunking_time': chunking_time
    }

    return {'chunks': chunks, 'metrics': metrics}
# Example usage
if __name__ == "__main__":
    sample_text = "This is a sample text. It contains multiple sentences. " * 100
    chunks = chunk_document(sample_text)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}: {chunk[:50]}...")