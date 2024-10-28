from openai import AzureOpenAI
from azure.identity import ChainedTokenCredential, ManagedIdentityCredential, EnvironmentCredential, AzureCliCredential
from typing import List, Dict, Any, Optional
import logging
import time
import numpy as np
from src.embedding_generator_base_class import EmbeddingGenerator
from src.rate_limiter import AzureRateLimiter, TokenizerModel    
from src.singleton_config import ConfigSingleton      

class AzureOpenAIEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self, azure_endpoint: str, api_version: str, deployment: str):
        """
        Initialize the Azure OpenAI Embedding Generator with rate limiting.
        
        Args:
            azure_endpoint: Azure OpenAI endpoint URL
            api_version: API version to use
            deployment: Deployment name for the embedding model
        """
        self.config =  ConfigSingleton(  ) 
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self.deployment = deployment
        self.model =  self.config.get_active_embedding_config( ).deployment_name  
        self._dimension = 1536  # Known dimension for this model
        
        # Initialize rate limiter
        self.rate_limiter = AzureRateLimiter(
            model_type=TokenizerModel.EMBEDDING,
            requests_per_minute_limit=3500,
            tokens_per_minute_limit=350000
        )
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize client
        try:
            self.client = self._initialize_client()
            self.logger.info("Azure OpenAI client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
            raise

    def _initialize_client(self) -> AzureOpenAI:
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
                azure_endpoint=self.config.get_active_embedding_config(  ).api_base,    
                api_version=self.config.get_active_embedding_config(  ).api_version        
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text chunk with rate limiting.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List[float]: Embedding vector
        """
        # Check rate limits and text size
        should_throttle, wait_time = self.rate_limiter.should_throttle(text)
        if should_throttle:
            if wait_time > 0:
                self.logger.info(f"Rate limit prevention: waiting {wait_time:.2f}s")
                time.sleep(wait_time)
            else:
                # Text is too large, needs to be handled by chunking system
                raise ValueError(f"Text too large: {self.rate_limiter.count_tokens(text)} tokens")

        retry_count = 0
        max_retries = 3
        
        while retry_count <= max_retries:
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model
                )
                
                # Record successful request
                self.rate_limiter.record_request(text, success=True)
                
                return response.data[0].embedding
                
            except Exception as e:
                error_str = str(e)
                self.rate_limiter.record_request(text, success=False, error=error_str)
                
                if "rate_limit" in error_str.lower():
                    retry_count += 1
                    if retry_count > max_retries:
                        self.logger.error(f"Max retries ({max_retries}) exceeded for rate limit")
                        raise
                    wait_time = self.rate_limiter.handle_rate_limit(retry_count)
                    self.logger.warning(f"Rate limit hit, attempt {retry_count}/{max_retries}, waiting {wait_time}s")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Error generating embedding: {error_str}")
                    raise

    def generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple text chunks with rate limiting.
        
        Args:
            chunks: List of text chunks to generate embeddings for
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        self.logger.info(f"Generating embeddings for {len(chunks)} chunks")
        results = []
        failed_chunks = []
        
        for i, chunk in enumerate(chunks, 1):
            try:
                # Get current usage metrics for logging
                usage = self.rate_limiter.get_current_metrics()
                self.logger.debug(
                    f"Processing chunk {i}/{len(chunks)}. "
                    f"Current usage: {usage['requests_per_minute']}/min, "
                    f"tokens: {usage['tokens_per_minute']}/min"
                )
                
                embedding = self.generate_embedding(chunk)
                results.append(embedding)
                
            except Exception as e:
                self.logger.error(f"Failed to process chunk {i}: {str(e)}")
                failed_chunks.append(i)
                results.append(None)  # Maintain index alignment
        
        if failed_chunks:
            self.logger.warning(f"Failed to process chunks: {failed_chunks}")
        
        return results

    def get_rate_limit_metrics(self) -> Dict[str, Any]:
        """Get current rate limit metrics."""
        return self.rate_limiter.get_current_metrics()

    @property
    def dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self._dimension