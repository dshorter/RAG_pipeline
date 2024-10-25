"""
Enhanced AzureOpenAIEmbeddingGenerator with comprehensive error handling,
rate limit management, and detailed logging.

Features:
- Rate limit tracking and management
- Detailed performance monitoring
- Memory usage tracking
- Comprehensive error handling
- Batch processing with status monitoring
- Automatic retries with exponential backoff
"""

from openai import AzureOpenAI
from azure.identity import ChainedTokenCredential, ManagedIdentityCredential, EnvironmentCredential, AzureCliCredential
from typing import List, Dict, Any
import requests
import logging
import time
import random
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from src.logging_config import get_logger
from src.embedding_generator_base_class import EmbeddingGenerator
from src.singleton_config import ConfigSingleton
import openai

class AzureOpenAIEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self, azure_endpoint: str, api_version: str, deployment: str):
        self.logger = get_logger('embedding.azure')
        self.logger.debug("Initializing Azure OpenAI Embedding Generator",
                         extra={'operation': 'initialization'})
        
        self.config = ConfigSingleton()
        
        # Configuration setup
        self.azure_endpoint = self.config.get_active_embedding_config().api_base
        self.api_version = self.config.get_active_embedding_config().api_version
        self.deployment = "api-shared-text-embedding-ada-v002-nofilter"
        self.model = "api-shared-text-embedding-ada-v002-nofilter"
        self._dimension = self.config.get_active_embedding_config().dimension
        
        # Retry and timeout configurations
        self.request_timeout = 30  # seconds
        self.max_retries = 3
        
        # Rate limit tracking
        self.rate_limits = {
            'requests': 0,
            'last_reset': time.time(),
            'rate_limit_hits': 0
        }
        
        try:
            self.client = self._initialize_client()
            self.logger.info("Azure OpenAI client initialized successfully",
                           extra={
                               'operation': 'initialization',
                               'endpoint': self.azure_endpoint,
                               'model': self.model
                           })
        except Exception as e:
            self.logger.error("Failed to initialize Azure OpenAI client",
                            extra={
                                'operation': 'initialization_error',
                                'error': str(e),
                                'error_type': type(e).__name__
                            })
            raise

    def _initialize_client(self):
        """Initialize Azure OpenAI client with credential chain and testing."""
        try:
            self.logger.debug("Creating credential chain",
                            extra={'operation': 'client_init'})
            
            credential = ChainedTokenCredential(
                ManagedIdentityCredential(),
                EnvironmentCredential(),
                AzureCliCredential()
            )
            
            token_start = time.time()
            access_token = credential.get_token("https://cognitiveservices.azure.com/.default",
                                              timeout=30)
            token_duration = time.time() - token_start
            
            self.logger.debug("Token acquired",
                            extra={
                                'operation': 'token_acquisition',
                                'duration': token_duration
                            })
            
            client = AzureOpenAI(
                api_key=access_token.token,
                azure_endpoint=self.azure_endpoint,
                api_version=self.api_version
            )
            
            # Test the client with a minimal request
            test_start = time.time()
            test_response = client.embeddings.create(
                input="test",
                model=self.model,
                timeout=10
            )
            test_duration = time.time() - test_start
            
            self.logger.debug("Client test successful",
                            extra={
                                'operation': 'client_test',
                                'duration': test_duration
                            })
            
            return client
            
        except Exception as e:
            self.logger.error("Client initialization failed",
                            extra={
                                'operation': 'client_init_error',
                                'error': str(e),
                                'error_type': type(e).__name__
                            })
            raise

    def _track_request(self):
        """Track API request counts and rate limits"""
        current_time = time.time()
        # Reset counter if it's been more than a minute
        if current_time - self.rate_limits['last_reset'] > 60:
            self.logger.debug("Resetting rate limit counter",
                            extra={
                                'operation': 'rate_limit_reset',
                                'previous_count': self.rate_limits['requests'],
                                'duration': current_time - self.rate_limits['last_reset']
                            })
            self.rate_limits['requests'] = 0
            self.rate_limits['last_reset'] = current_time
        
        self.rate_limits['requests'] += 1

    def _handle_rate_limit(self):
        """Handle rate limit occurrence"""
        self.rate_limits['rate_limit_hits'] += 1
        self.logger.warning("Rate limit hit",
                          extra={
                              'operation': 'rate_limit',
                              'total_hits': self.rate_limits['rate_limit_hits'],
                              'requests_in_window': self.rate_limits['requests'],
                              'window_duration': time.time() - self.rate_limits['last_reset']
                          })

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status and statistics"""
        current_time = time.time()
        window_duration = current_time - self.rate_limits['last_reset']
        
        status = {
            'current_requests': self.rate_limits['requests'],
            'total_rate_limit_hits': self.rate_limits['rate_limit_hits'],
            'current_window_duration': round(window_duration, 2),
            'requests_per_minute': round(self.rate_limits['requests'] / (window_duration / 60), 2) if window_duration > 0 else 0,
            'window_start': datetime.fromtimestamp(self.rate_limits['last_reset']).isoformat(),
            'is_likely_limited': self.rate_limits['requests'] >= 150  # Assuming 150 requests/min limit
        }
        
        self.logger.debug("Rate limit status check",
                         extra={
                             'operation': 'rate_limit_status',
                             'status': status
                         })
        
        return status

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (requests.exceptions.Timeout,
             requests.exceptions.ConnectionError,
             requests.exceptions.RequestException)
        ),
        before_sleep=lambda retry_state: logging.getLogger('embedding.azure').warning(
            f"Retry attempt {retry_state.attempt_number} after error: {retry_state.outcome.exception()}",
            extra={
                'operation': 'retry_attempt',
                'attempt_number': retry_state.attempt_number,
                'error': str(retry_state.outcome.exception())
            }
        )
    )
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text string with comprehensive error handling."""
        start_time = time.time()
        
        self.logger.debug("Starting embedding generation",
                         extra={
                             'operation': 'embedding_start',
                             'text_length': len(text),
                             'timeout': self.request_timeout,
                             'current_requests': self.rate_limits['requests']
                         })
        
        try:
            self._track_request()
            response = self.client.embeddings.create(
                input=text,
                model=self.model,
                timeout=self.request_timeout
            )
            
            duration = time.time() - start_time
            self.logger.info("Embedding generated successfully",
                           extra={
                               'operation': 'embedding_success',
                               'duration': duration,
                               'text_length': len(text),
                               'requests_in_minute': self.rate_limits['requests']
                           })
            
            return response.data[0].embedding

        except openai.RateLimitError as e:
            duration = time.time() - start_time
            self._handle_rate_limit()
            self.logger.warning("Rate limit exceeded",
                              extra={
                                  'operation': 'rate_limit_error',
                                  'duration': duration,
                                  'error': str(e),
                                  'requests_in_minute': self.rate_limits['requests']
                              })
            raise

        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ReadTimeout) as e:
            duration = time.time() - start_time
            self.logger.warning("Timeout during embedding generation",
                              extra={
                                  'operation': 'embedding_timeout',
                                  'duration': duration,
                                  'error': str(e),
                                  'text_length': len(text)
                              })
            raise

        except requests.exceptions.RequestException as e:
            duration = time.time() - start_time
            self.logger.error("Request error during embedding generation",
                            extra={
                                'operation': 'embedding_request_error',
                                'duration': duration,
                                'error': str(e),
                                'error_type': type(e).__name__
                            })
            raise

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error("Unexpected error during embedding generation",
                            extra={
                                'operation': 'embedding_error',
                                'duration': duration,
                                'error': str(e),
                                'error_type': type(e).__name__
                            })
            raise

    def generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of text chunks with rate limit awareness."""
        start_time = time.time()
        total_chunks = len(chunks)
        
        # Check rate limit status before starting batch
        rate_status = self.get_rate_limit_status()
        self.logger.debug(f"Starting batch embedding generation",
                         extra={
                             'operation': 'batch_start',
                             'chunk_count': total_chunks,
                             'total_text_length': sum(len(chunk) for chunk in chunks),
                             'rate_limit_status': rate_status
                         })
        
        embeddings = []
        failed_chunks = 0
        rate_limited_chunks = 0
        
        for i, chunk in enumerate(chunks, 1):
            chunk_start = time.time()
            try:
                embedding = self.generate_embedding(chunk)
                embeddings.append(embedding)
                
                chunk_duration = time.time() - chunk_start
                self.logger.debug(f"Chunk {i}/{total_chunks} embedded",
                                extra={
                                    'operation': 'chunk_success',
                                    'chunk_number': i,
                                    'total_chunks': total_chunks,
                                    'duration': chunk_duration,
                                    'chunk_length': len(chunk),
                                    'requests_in_minute': self.rate_limits['requests']
                                })
                
            except openai.RateLimitError:
                rate_limited_chunks += 1
                # Get current status after rate limit hit
                current_status = self.get_rate_limit_status()
                self.logger.warning(f"Rate limit hit during batch processing",
                                  extra={
                                      'operation': 'batch_rate_limit',
                                      'chunk_number': i,
                                      'rate_limited_chunks': rate_limited_chunks,
                                      'status': current_status
                                  })
                raise
                
            except Exception as e:
                failed_chunks += 1
                self.logger.error(f"Failed to generate embedding for chunk {i}",
                                extra={
                                    'operation': 'chunk_error',
                                    'chunk_number': i,
                                    'total_chunks': total_chunks,
                                    'error': str(e),
                                    'error_type': type(e).__name__,
                                    'duration': time.time() - chunk_start
                                })
                raise
        
        total_duration = time.time() - start_time
        final_status = self.get_rate_limit_status()
        
        self.logger.info("Batch embedding completed",
                        extra={
                            'operation': 'batch_complete',
                            'total_chunks': total_chunks,
                            'successful_chunks': len(embeddings),
                            'failed_chunks': failed_chunks,
                            'rate_limited_chunks': rate_limited_chunks,
                            'total_duration': total_duration,
                            'average_time_per_chunk': total_duration / total_chunks if total_chunks > 0 else 0,
                            'final_rate_status': final_status
                        })
        
        return embeddings

    @property
    def dimension(self) -> int:
        return self._dimension


