# src/data_classes.py
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class Document:
    document_id: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    source: Optional[str] = None
    date_added: Optional[str] = None
    document_length: Optional[int] = 0  
    summary: Optional[str] = None
    tags: Optional[str] = None
    metadata_jason: Optional[str] = None

@dataclass
class ChunkMetrics:
    chunk_id: Optional[int] = None 
    document: Optional[int] = None 
    chunk_text: Optional[str] = None 
    start_index: Optional[int] = None 
    end_index: Optional[int] = None 
    chunk_length: Optional[int] = None 
    metadata: Optional[str] = None 
    date_added: Optional[str] = None    


@dataclass
class ChunkMetrics:
    chunk_id: Optional[int] = None 
    document: Optional[int] = None 
    chunk_text: Optional[str] = None 
    start_index: Optional[int] = None 
    end_index: Optional[int] = None 
    chunk_length: Optional[int] = None 
    metadata: Optional[str] = None 
    date_added: Optional[str] = None

@dataclass
class QueryAnalysisResult:
    """Results from analyzing a query's complexity."""
    complexity_score: float
    needs_reranking: bool
    recommended_k: int
    recommended_candidates: int
    features: Dict[str, float]
    explanation: List[str]

@dataclass
class RerankingResult:
    """Represents a re-ranked chunk with both original and new scores."""
    chunk_id: str
    chunk_text: str
    initial_score: float
    reranked_score: float
    metadata: Dict[str, Any]


    
