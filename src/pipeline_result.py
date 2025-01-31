from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class ChunkInfo:
    text: str
    start_index: int
    end_index: int

@dataclass
class PipelineResult:
    document_id: str
    document_name: str
    processed_text: str
    metadata: Dict[str, Any]
    chunks: List[ChunkInfo]
    embeddings: List[List[float]]
    
    def get_chunk_text(self, chunk_id: int) -> str:
        return self.chunks[chunk_id].text

    def get_chunk_embedding(self, chunk_id: int) -> List[float]:
        return self.embeddings[chunk_id]

    def get_chunk_with_embedding(self, chunk_id: int) -> tuple:
        return self.chunks[chunk_id], self.embeddings[chunk_id]

    def prepare_for_indexing(self) -> List[Dict]:
        return [
            {
                "document_id": self.document_id,
                "chunk": chunk.text,
                "vector": embedding,
                "start_index": chunk.start_index,
                "end_index": chunk.end_index
            }
            for chunk, embedding in zip(self.chunks, self.embeddings)
        ]

    def summary(self) -> str:
        return f"""
        Document: {self.document_name}
        Total Chunks: {len(self.chunks)}
        Embeddings: {len(self.embeddings)}
        """