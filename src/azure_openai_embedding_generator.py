
from openai import AzureOpenAI
from azure.identity import ChainedTokenCredential, ManagedIdentityCredential, EnvironmentCredential, AzureCliCredential
from typing import List, Dict, Any, Optional
import logging
import time
from .embedding_generator_base_class import EmbeddingGenerator
from .rate_limiter import AzureRateLimiter, TokenizerModel    
from .singleton_config import ConfigSingleton      
from .logging_config import get_logger  
from .metrics_collector import  MetricsCollector   

   
class AzureOpenAIEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self, azure_endpoint: str, api_version: str, deployment: str):
        self.logger = get_logger('embedding.azure')
        self.config = ConfigSingleton()
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self.deployment = deployment
        self.client = self._initialize_client()
        self.model = self.config.get_active_embedding_config().deployment_name  
        self.metrics_collector = MetricsCollector    
        
        self._dimension = 1536
        
        self.rate_limiter = AzureRateLimiter(
            model_type=TokenizerModel.EMBEDDING,
            requests_per_minute_limit=3500,
            tokens_per_minute_limit=350000
        )
        
        try:
            self.client = self._initialize_client()
            self.logger.info("Azure OpenAI client initialized successfully", 
                           extra={'operation': 'client_init'})
        except Exception as e:
            self.logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}", 
                            extra={'operation': 'client_init_error'})
            raise

    def generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        start_time = time.time()
        results = []
        failed_chunks = []

        for i, chunk in enumerate(chunks):
            try:
                embedding = self.generate_embedding(chunk)
                if embedding is not None:
                    results.append(embedding)
                    self.rate_limiter.record_request(chunk, success=True)
            except Exception as e:
                self.logger.error(f"Failed to process chunk {i}: {str(e)}")
                failed_chunks.append(i)
                self.rate_limiter.record_request(chunk, success=False, error=str(e))
                results.append(None)

        # Fire and forget metrics
        self.metrics_collector.collect(
            operation='embedding_generation',
            component='azure_embedder',
            metrics={
                'duration_ms': (time.time() - start_time) * 1000,
                'chunk_count': len(chunks),
                'success_count': len(results) - len(failed_chunks),
                'failure_count': len(failed_chunks),
                'model': {
                    'name': self.model,
                    'dimensions': self._dimension
                },
                'rate_limits': self.rate_limiter.get_current_metrics()
            }
        )

        if all(r is None for r in results):
            raise Exception("All chunks failed to process")

        return [r for r in results if r is not None]

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            return None

    def _initialize_client(self):
        """Initialize Azure OpenAI client with proper authentication."""
        try:
            credential = ChainedTokenCredential(
                ManagedIdentityCredential(),
                EnvironmentCredential(),
                AzureCliCredential()
            )
            access_token = credential.get_token("https://cognitiveservices.azure.com/.default")
            
            return AzureOpenAI(
                api_key=access_token.token,
                azure_endpoint=self.config.get_active_embedding_config().api_base,    
                api_version=self.config.get_active_embedding_config().api_version        
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}", 
                            extra={'operation': 'client_init_error'})
            raise

    @property
    def dimension(self) -> int:
        return self._dimension




