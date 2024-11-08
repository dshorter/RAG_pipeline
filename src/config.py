import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import yaml
from dotenv import load_dotenv     
from src.data_classes import Document, ChunkMetrics

@dataclass
class GPTConfig:
    provider: str
    model_name: str
    temperature: float
    max_tokens: int
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    azure_tennant_id: Optional[str] = None
    azure_client_id: Optional[str] = None
    azure_client_secret: Optional[str] = None    

@dataclass
class OpenAIEmbeddingConfig:
    model_name: str
    dimension: int
    azure_tennant_id: str          
    azure_client_id: str      
    azure_client_secret: str      
    deployment_name: str
    api_key: str 
    api_base: str 
    api_version: str 

@dataclass
class HuggingFaceEmbeddingConfig:
    model_name: str
    dimension: int

@dataclass
class EmbeddingConfig:
    provider: str
    models: Dict[str, Any]
    active_model: str

@dataclass
class PipelineConfig:
    embedding: EmbeddingConfig
    chunk_size: int
    chunk_overlap: int
    raw_docs_dir: str
    processed_docs_dir: str

@dataclass
class QueryAnalysisConfig:
    """Configuration for query analysis."""
    enabled: bool
    thresholds: Dict[str, float]  # Complexity thresholds
    weights: Dict[str, float]     # Feature weights for scoring

@dataclass
class RerankingConfig:  
    """Configuration for re-ranking."""
    enabled: bool
    provider: str
    models: Dict[str, Dict[str, Any]]
    thresholds: Dict[str, float]
    active_model: str

@dataclass
class Config:
    pipeline: PipelineConfig
    gpt: GPTConfig
    faiss_index_dir: str
    reranking: RerankingConfig
    query_analysis: QueryAnalysisConfig
    additional_settings: Dict[str, Any] = field(default_factory=dict)

class Configuration:
    def __init__(self, config_file: str):
        self.document = Document()
        self.chunk_metrics = ChunkMetrics()
        
        load_dotenv()  # Load environment variables from .env file
        
        # Get the directory of the current file (config.py)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the path to config.yml
        config_path = os.path.join(current_dir, config_file)    

        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Load embedding configuration
        embedding_config = config_data['pipeline']['embedding']
        embedding_config['models']['azure_openai']['api_key'] = os.getenv('OPENAI_API_KEY') or embedding_config['models']['azure_openai'].get('api_key')
        embedding_config['models']['azure_openai']['api_base'] = embedding_config['models']['azure_openai'].get('api_base')
        embedding_config['models']['azure_openai']['api_version'] = os.getenv('OPENAI_API_VERSION') or embedding_config['models']['azure_openai'].get('api_version')
        embedding_config['models']['azure_openai']['azure_tennant_id'] = os.getenv('AZURE_TENANT_ID') 
        embedding_config['models']['azure_openai']['azure_client_secret'] = os.getenv('AZURE_CLIENT_SECRET')     
        embedding_config['models']['azure_openai']['azure_client_id'] = os.getenv('AZURE_CLIENT_ID')     
        embedding_config['models']['azure_openai']['deployment_name'] = embedding_config['models']['azure_openai'].get('deployment_name')
                
        # Create EmbeddingConfig instance
        embedding = EmbeddingConfig(**embedding_config)
        
        # Create PipelineConfig instance
        pipeline_config = config_data['pipeline']
        pipeline_config['embedding'] = embedding
        pipeline = PipelineConfig(**pipeline_config)
         
        # Create GPTConfig    
        gpt_config = config_data['gpt'] 
        gpt = GPTConfig(**gpt_config)

        # Create RerankingConfig
        reranking_config = config_data.get('reranking', {})
        reranking = RerankingConfig(
            enabled=reranking_config.get('enabled', True),
            provider=reranking_config.get('provider', 'huggingface'),
            models=reranking_config.get('models', {
                'huggingface': {
                    'model_name': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                    'batch_size': 32,
                    'max_length': 512
                }
            }),
            thresholds=reranking_config.get('thresholds', {
                'complexity_threshold': 1.2,
                'min_score': 0.6,
                'max_candidates': 20
            }),
            active_model=reranking_config.get('active_model', 'huggingface')
        )

        # Create QueryAnalysisConfig
        analysis_config = config_data.get('query_analysis', {})
        query_analysis = QueryAnalysisConfig(
            enabled=analysis_config.get('enabled', True),
            thresholds=analysis_config.get('thresholds', {
                'simple_query': 0.8,
                'complex_query': 1.2,
                'max_chunks_simple': 3,
                'max_chunks_complex': 5,
                'max_candidates': 15
            }),
            weights=analysis_config.get('weights', {
                'word_count': 0.3,
                'entity_count': 0.2,
                'dependency_depth': 0.2,
                'keyword_complexity': 0.2,
                'question_complexity': 0.1
            })
        )
          
        # Create main Config instance
        self.config = Config(
            pipeline=pipeline,
            gpt=gpt, 
            faiss_index_dir=config_data['vectors']['faiss']['faiss_index_dir'],
            reranking=reranking,
            query_analysis=query_analysis,
            additional_settings=config_data.get('additional_settings', {})
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self.config, key, default)
    
    def get_pipeline_config(self) -> PipelineConfig:
        return self.config.pipeline
    
    def get_gpt_config(self) -> GPTConfig:
        return self.config.gpt

    def get_reranking_config(self) -> RerankingConfig:
        """Get re-ranking configuration."""
        return self.config.reranking

    def get_query_analysis_config(self) -> QueryAnalysisConfig:
        """Get query analysis configuration."""
        return self.config.query_analysis
    
    def get_active_embedding_config(self):
        embedding_config = self.config.pipeline.embedding
        active_model = embedding_config.active_model
        if active_model == "azure_openai":
            return OpenAIEmbeddingConfig(**embedding_config.models["azure_openai"])
        elif active_model == "huggingface":
            return HuggingFaceEmbeddingConfig(
                model_name="sentence-transformers/all-MiniLM-L6-v2", 
                dimension=1536
            )
        else:
            raise ValueError(f"Unknown embedding model provider: {active_model}")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pipeline': {
                'embedding': {
                    'provider': self.config.pipeline.embedding.provider,
                    'active_model': self.config.pipeline.embedding.active_model,
                    'models': {
                        'azure_openai': {
                            'model_name': self.config.pipeline.embedding.models['azure_openai']['model_name'],
                            'dimension': self.config.pipeline.embedding.models['azure_openai']['dimension'],
                            'api_key': '********' if self.config.pipeline.embedding.models['azure_openai'].get('api_key') else None,
                            'api_base': self.config.pipeline.embedding.models['azure_openai'].get('api_base'),
                            'api_version': self.config.pipeline.embedding.models['azure_openai'].get('api_version'),
                        },
                        'huggingface': {
                            'model_name': self.config.pipeline.embedding.models['huggingface']['model_name'],
                            'dimension': self.config.pipeline.embedding.models['huggingface']['dimension'],
                        }
                    }
                },
                'chunk_size': self.config.pipeline.chunk_size,
                'chunk_overlap': self.config.pipeline.chunk_overlap,
                'raw_docs_dir': self.config.pipeline.raw_docs_dir,
                'processed_docs_dir': self.config.pipeline.processed_docs_dir,
            },
            'reranking': {
                'enabled': self.config.reranking.enabled,
                'provider': self.config.reranking.provider,
                'active_model': self.config.reranking.active_model,
                'thresholds': self.config.reranking.thresholds
            },
            'query_analysis': {
                'enabled': self.config.query_analysis.enabled,
                'thresholds': self.config.query_analysis.thresholds,
                'weights': self.config.query_analysis.weights
            },
            'faiss_index_dir': self.config.faiss_index_dir,
            'additional_settings': self.config.additional_settings,
        }