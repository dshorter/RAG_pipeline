from dataclasses import dataclass, field
from typing import Dict, List, Any

@dataclass
class ChunkInfo:
    text: str
    start_index: int
    end_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ChunkMetrics:
    total_chunks: int
    avg_chunk_size: float
    max_chunk_size: int
    min_chunk_size: int
    # chunking_time: float

@dataclass
class VectorMetrics:
    num_embeddings: int
    embedding_dimension: int
    embedding_generation_start_time: float
    embedding_generation_end_time: float
    embedding_generation_time: float

@dataclass
class PipelineResult:
    document_name: str
    processed_text: str
    metadata: Dict[str, Any]
    chunks: List[ChunkInfo]
    chunk_metrics: ChunkMetrics
    embeddings: List[List[float]]
    vector_metrics: VectorMetrics

    def get_chunk_text(self, chunk_id: int) -> str:
        return self.chunks[chunk_id].text

    def get_chunk_embedding(self, chunk_id: int) -> List[float]:
        return self.embeddings[chunk_id]

    def get_chunk_with_embedding(self, chunk_id: int) -> tuple:
        return self.chunks[chunk_id], self.embeddings[chunk_id]

    def get_all_chunks_with_embeddings(self) -> List[tuple]:
        return list(zip(self.chunks, self.embeddings))

    def prepare_for_indexing(self) -> List[Dict]:
        return [
            {
                "chunk": chunk.text,
                "vector": embedding,
                "source": chunk.metadata.get('source', ''),
                "start_index": chunk.start_index,
                "end_index": chunk.end_index
            }
            for chunk, embedding in zip(self.chunks, self.embeddings)
        ]

    def summary(self) -> str:
        return f"""
        Document: {self.document_name}
        Total Chunks: {self.chunk_metrics.total_chunks}
        Average Chunk Size: {self.chunk_metrics.avg_chunk_size:.2f}
        Total Embeddings: {self.vector_metrics.num_embeddings}
        Embedding Dimension: {self.vector_metrics.embedding_dimension}
        Chunking Time:   {21} seconds
        Embedding Time: {self.vector_metrics.embedding_generation_time:.2f} seconds
        """