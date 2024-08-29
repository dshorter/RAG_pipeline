import logging
import time
from typing import Dict

logger = logging.getLogger(__name__)

def chunk_document(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    strategy: str = 'fixed'
) -> Dict[str, any]:
    logger.info(f"Chunking document with strategy: {strategy}")
    logger.info(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")

    start_time = time.time()

    # Tokenization (simple word splitting for now)
    words = text.split()
    tokenization_time = time.time() - start_time

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        logger.debug(f"Created chunk of size {len(chunk.split())} words")
        start = end - chunk_overlap

    chunking_time = time.time() - start_time - tokenization_time

    logger.info(f"Created {len(chunks)} chunks")

    # Collect metrics
    metrics = {
        'total_words': len(words),
        'num_chunks': len(chunks),
        'avg_chunk_size': sum(len(chunk.split()) for chunk in chunks) / len(chunks),
        'max_chunk_size': max(len(chunk.split()) for chunk in chunks),
        'min_chunk_size': min(len(chunk.split()) for chunk in chunks),
        'tokenization_time': tokenization_time,
        'chunking_time': chunking_time
    }

    return {'chunks': chunks, 'metrics': metrics}
# Example usage
if __name__ == "__main__":
    sample_text = "This is a sample text. It contains multiple sentences. " * 100
    chunks = chunk_document(sample_text)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}: {chunk[:50]}...")