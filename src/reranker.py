from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class RerankingResult:
    chunk_id: str
    chunk_text: str
    initial_score: float
    reranked_score: float
    metadata: Dict[str, Any]

class ReRankerBase(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[RerankingResult]:
        """
        Re-rank candidate chunks based on query relevance.
        
        Args:
            query: The user's query
            candidates: List of candidate chunks with scores
            top_k: Number of results to return after re-ranking
            
        Returns:
            List of re-ranked results with both initial and new scores
        """
        pass

    @abstractmethod
    def get_score(self, query: str, text: str) -> float:
        """
        Get relevance score for a single query-text pair.
        
        Args:
            query: The query text
            text: The candidate text
            
        Returns:
            Relevance score between 0 and 1
        """
        pass