import torch
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any
from src.singleton_config import ConfigSingleton
from src.logging_config import get_logger
from src.reranker_base import ReRankerBase, RerankingResult

class HuggingFaceReRanker(ReRankerBase):
    def __init__(self):
        self.config = ConfigSingleton()
        self.logger = get_logger('reranker.huggingface')
        
        # Get model config
        model_config = self.config.get_reranking_config()
        self.model_name = model_config.models['huggingface']['model_name']
        self.batch_size = model_config.models['huggingface']['batch_size']
        self.max_length = model_config.models['huggingface']['max_length']
        
        # Determine device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            self.model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device
            )
            self.logger.info(
                f"Initialized re-ranker with model: {self.model_name}",
                extra={
                    'component': 'reranker',
                    'operation': 'init',
                    'model': self.model_name,
                    'device': self.device  # Use our stored device value
                }
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize re-ranker: {str(e)}")
            raise

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[RerankingResult]:
        try:
            pairs = [(query, cand['chunk_text']) for cand in candidates]
            
            all_scores = []
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i:i + self.batch_size]
                scores = self.model.predict(
                    batch,
                    show_progress_bar=False,
                    batch_size=self.batch_size
                )
                all_scores.extend(scores)

            results = []
            for score, candidate in zip(all_scores, candidates):
                results.append(
                    RerankingResult(
                        chunk_id=candidate['chunk_id'],
                        chunk_text=candidate['chunk_text'],
                        initial_score=candidate.get('relevance_score', 0.0),
                        reranked_score=float(score),
                        metadata=candidate.get('metadata', {})
                    )
                )
            
            results.sort(key=lambda x: x.reranked_score, reverse=True)
            
            if top_k:
                results = results[:top_k]
            
            return results
            
        except Exception as e:
            self.logger.error(f"Re-ranking failed: {str(e)}")
            raise

    def get_score(self, query: str, text: str) -> float:
        try:
            score = float(self.model.predict([(query, text)]))
            return score
        except Exception as e:
            self.logger.error(f"Scoring failed: {str(e)}")
            raise