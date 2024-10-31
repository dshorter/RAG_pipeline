from openai import AzureOpenAI
from azure.identity import ChainedTokenCredential, ManagedIdentityCredential, EnvironmentCredential, AzureCliCredential
from typing import List, Dict, Any, Optional
import logging
import time
import numpy as np
from src.embedding_generator_base_class import EmbeddingGenerator
from src.rate_limiter import AzureRateLimiter, TokenizerModel    
from src.singleton_config import ConfigSingleton      
from src.logging_config import get_logger  

class AzureOpenAIEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self, azure_endpoint: str, api_version: str, deployment: str):
        """
        Initialize the Azure OpenAI Embedding Generator with rate limiting.
        """    
        self.logger = get_logger('embedding.azure')  # Only one logger definition
        self.config = ConfigSingleton()
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self.deployment = deployment
        self.model = self.config.get_active_embedding_config().deployment_name
        self._dimension = 1536  # Known dimension for this model
        
        # Initialize rate limiter
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
        """
        Generate embeddings for multiple text chunks with rate limiting.
        """
        self.logger.info(f"Generating embeddings for {len(chunks)} chunks", 
                        extra={'operation': 'embeddings_start'})
        results = []
        failed_chunks = []

        for i, chunk in enumerate(chunks, 1):
            try:
                # Get current usage metrics for logging
                usage = self.rate_limiter.get_current_metrics()
                self.logger.debug(
                    f"Processing chunk {i}/{len(chunks)}. Current usage: {usage}", 
                    extra={'operation': 'chunk_processing'}
                )
                
                # Check rate limits before processing
                should_throttle, wait_time = self.rate_limiter.should_throttle(chunk)
                if should_throttle:
                    self.logger.info(f"Rate limit prevention: waiting {wait_time:.2f}s", 
                                   extra={'operation': 'rate_limit_wait'})
                    time.sleep(wait_time)
                
                embedding = self.generate_embedding(chunk)
                if embedding is not None:
                    results.append(embedding)
                    self.rate_limiter.record_request(chunk, success=True)
                else:
                    raise ValueError("Generated embedding is None")
                
            except Exception as e:                
                self.logger.error(f"Failed to process chunk {i}: {str(e)}", 
                                extra={'operation': 'chunk_processing_error'})
                failed_chunks.append(i)
                self.rate_limiter.record_request(chunk, success=False, error=str(e))
                results.append(None)  # Maintain index alignment
        
        if failed_chunks:
            self.logger.warning(f"Failed to process chunks: {failed_chunks}", 
                              extra={'operation': 'chunks_failed'})
            if len(failed_chunks) == len(chunks):
                raise Exception("All chunks failed to process")
        
        # Filter out None values
        results = [r for r in results if r is not None]
        if not results:
            raise Exception("No valid embeddings generated")
            
        self.logger.info(f"Successfully generated {len(results)} embeddings", 
                        extra={'operation': 'embeddings_complete'})

        return results

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text chunk with rate limiting and retries.
        """
        retry_count = 0
        max_retries = 3
        
        while retry_count <= max_retries:
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model
                )
                
                return response.data[0].embedding
                
            except Exception as e:
                error_str = str(e)
                if "rate_limit" in error_str.lower():
                    retry_count += 1
                    if retry_count > max_retries:
                        self.logger.error(f"Max retries ({max_retries}) exceeded for rate limit", 
                                        extra={'operation': 'retry_limit_exceeded'})
                        raise
                    wait_time = self.rate_limiter.handle_rate_limit(retry_count)
                    self.logger.warning(f"Rate limit hit, attempt {retry_count}/{max_retries}, waiting {wait_time}s", 
                                      extra={'operation': 'rate_limit_retry'})
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Error generating embedding: {error_str}", 
                                    extra={'operation': 'embedding_error'})
                    raise
        
        return None

    def _initialize_client(self):
        """Initialize the Azure OpenAI client with proper credentials."""
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
        """Get the dimension of the embeddings."""
        return self._dimension