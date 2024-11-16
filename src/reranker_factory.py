from src.singleton_config import ConfigSingleton
from src.reranker_base import ReRankerBase
from src.hugging_face_reranker import HuggingFaceReRanker

class ReRankerFactory:
    @staticmethod
    def create() -> ReRankerBase:
        """
        Create a re-ranker instance based on configuration.
        Uses ConfigSingleton instead of passing config manually.
        """
        config = ConfigSingleton()
        reranking_config = config.get_reranking_config()
        
        if reranking_config.provider == 'huggingface':
            return HuggingFaceReRanker()
        else:
            raise ValueError(f"Unknown re-ranker provider: {reranking_config.provider}")    
        