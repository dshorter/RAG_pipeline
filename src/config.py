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
    azure_tennant_id:  str          
    azure_client_id: str      
    azure_client_secret: str      
    deployment_name:  str
    api_key:  str 
    api_base:  str 
    api_version:  str 


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
class Config:
    pipeline: PipelineConfig
    faiss_index_dir: str
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
        
        # Create main Config instance
        self.config = Config(
            pipeline=pipeline,
            faiss_index_dir=config_data['faiss_index_dir'],
            additional_settings=config_data.get('additional_settings', {})
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self.config, key, default)
    
    def get_pipeline_config(self) -> PipelineConfig:
        return self.config.pipeline
    
    def get_active_embedding_config(self):
        embedding_config = self.config.pipeline.embedding
        active_model = embedding_config.active_model
        if active_model == "azure_openai":
            return OpenAIEmbeddingConfig(**embedding_config.models["azure_openai"])
        elif active_model == "huggingface":
            return HuggingFaceEmbeddingConfig(**embedding_config.models["huggingface"])
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
            'faiss_index_dir': self.config.faiss_index_dir,
            'additional_settings': self.config.additional_settings,
        }

# Usage example
if __name__ == "__main__":
    config = Configuration('config.yml')
    print(config.get_active_embedding_config().model_name)
    print(config.get_pipeline_config().chunk_size)
    print(config.to_dict())