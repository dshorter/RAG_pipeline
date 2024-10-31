# src/rate_limiter.py

from dataclasses import dataclass
from datetime import datetime
import threading
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from src.singleton_config import ConfigSingleton

# Third-party imports
import tiktoken

def get_tokenizer_model_name(model_name: str) -> str:
    """
    Convert Azure OpenAI model names to base OpenAI model names for tiktoken.
    
    Args:
        model_name: Azure OpenAI model name
        
    Returns:
        Corresponding base OpenAI model name for tiktoken
    """
    model_mapping = {
        "api-shared-text-embedding-ada-v002-nofilter": "text-embedding-ada-002",
        "api-shared-gpt-4-turbo-nofilter": "gpt-4",
        "gpt-35-turbo": "gpt-3.5-turbo"
    }
    return model_mapping.get(model_name, model_name)

class TokenizerModel(Enum):
    """Supported models for tokenization."""
    EMBEDDING = get_tokenizer_model_name(ConfigSingleton().get_active_embedding_config().model_name)
    GPT35 = "gpt-3.5-turbo"
    GPT4 = get_tokenizer_model_name(ConfigSingleton().get_gpt_config().model_name)

@dataclass
class RateLimitMetrics:
    """Tracks rate limit metrics with thread-safe access."""
    requests_per_minute: int = 0
    tokens_per_minute: int = 0
    last_reset: float = time.time()
    rate_limit_hits: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_processed: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None

class AzureRateLimiter:
    """
    Handles rate limiting for Azure OpenAI API calls with token counting.
    
    This class provides thread-safe rate limiting functionality with:
    - Request and token counting
    - Automatic rate limit prevention
    - Token-aware throttling
    - Comprehensive metrics tracking
    """
    
    def __init__(
        self,
        model_type: TokenizerModel = TokenizerModel.EMBEDDING,
        requests_per_minute_limit: int = 3500,
        tokens_per_minute_limit: int = 350000,
        safety_factor: float = 0.95
    ):
        """
        Initialize the rate limiter.
        
        Args:
            model_type: The model type for tokenization
            requests_per_minute_limit: Maximum requests per minute
            tokens_per_minute_limit: Maximum tokens per minute
            safety_factor: Safety buffer factor (0.0 to 1.0)
        """
        self.model_type = model_type
        self.requests_per_minute_limit = requests_per_minute_limit
        self.tokens_per_minute_limit = tokens_per_minute_limit
        self.safety_factor = safety_factor
        
        # Initialize metrics with thread-safe access
        self.metrics = RateLimitMetrics()
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model_type.value)
            self.logger.info(f"Initialized tiktoken encoder for {self.model_type.value}")
        except Exception as e:
            self.logger.error(f"Failed to initialize tokenizer for {self.model_type.value}: {e}")
            # Fallback to cl100k_base for embedding models
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                self.logger.info("Initialized fallback cl100k_base encoder")
            except Exception as fallback_e:
                self.logger.error(f"Failed to initialize fallback tokenizer: {fallback_e}")
                self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """
        Get precise token count using the appropriate tokenizer.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens in the text
        """
        if not text:
            return 0
            
        try:
            if self.tokenizer is None:
                raise RuntimeError("Tokenizer not initialized")
                
            tokens = self.tokenizer.encode(text)
            return len(tokens)
        except Exception as e:
            self.logger.warning(f"Token counting failed: {e}. Using fallback estimation.")
            # Conservative fallback: assume each word might be multiple tokens
            return len(text.split()) * 2

    def should_throttle(self, text: Optional[str] = None) -> Tuple[bool, float]:
        """
        Determine if request should be throttled based on current usage.
        
        Args:
            text: Optional text to check token count for
            
        Returns:
            Tuple of (should_throttle: bool, wait_time: float)
        """
        with self._lock:
            current_time = time.time()
            
            # Check if we need to reset our minute counters
            if current_time - self.metrics.last_reset >= 60:
                self._reset_metrics(current_time)
                return False, 0
            
            # If text provided, check its token count
            if text is not None:
                token_count = self.count_tokens(text)
                if token_count + self.metrics.tokens_per_minute > self.tokens_per_minute_limit:
                    return True, self._calculate_backoff_time(1.5)  # Conservative backoff
            
            # Calculate current rates
            elapsed_time = current_time - self.metrics.last_reset
            if elapsed_time > 0:
                request_rate = (self.metrics.requests_per_minute / elapsed_time) * 60
                token_rate = (self.metrics.tokens_per_minute / elapsed_time) * 60
                
                # Check against limits with safety factor
                request_ratio = request_rate / (self.requests_per_minute_limit * self.safety_factor)
                token_ratio = token_rate / (self.tokens_per_minute_limit * self.safety_factor)
                
                max_ratio = max(request_ratio, token_ratio)
                if max_ratio > 1:
                    wait_time = self._calculate_backoff_time(max_ratio)
                    return True, wait_time
            
            return False, 0

    def record_request(self, text: str, success: bool, error: Optional[str] = None) -> None:
        """
        Record metrics for a request with thread-safe updates.
        
        Args:
            text: Text that was processed
            success: Whether the request was successful
            error: Optional error message if request failed
        """
        token_count = self.count_tokens(text)
        
        with self._lock:
            self.metrics.requests_per_minute += 1
            self.metrics.tokens_per_minute += token_count
            self.metrics.total_requests += 1
            self.metrics.total_tokens_processed += token_count
            
            if success:
                self.metrics.successful_requests += 1
            else:
                self.metrics.failed_requests += 1
                if error:
                    self.metrics.last_error = error
                    self.metrics.last_error_time = datetime.now()
                    if "rate_limit" in error.lower():
                        self.metrics.rate_limit_hits += 1

    def handle_rate_limit(self, retry_count: int) -> float:
        """
        Handle a rate limit error, returning wait time before retry.
        
        Args:
            retry_count: Current retry attempt number
            
        Returns:
            Time to wait before next attempt
        """
        base_time = 2.0  # Base exponential backoff time
        wait_time = base_time ** retry_count
        
        self.logger.warning(
            f"Rate limit hit. Retry {retry_count} "
            f"waiting {wait_time:.2f}s"
        )
        
        return wait_time

    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics with thread-safe access.
        
        Returns:
            Dictionary containing current metrics
        """
        with self._lock:
            return {
                "requests_per_minute": self.metrics.requests_per_minute,
                "tokens_per_minute": self.metrics.tokens_per_minute,
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "total_tokens_processed": self.metrics.total_tokens_processed,
                "rate_limit_hits": self.metrics.rate_limit_hits,
                "last_error": self.metrics.last_error,
                "last_error_time": self.metrics.last_error_time,
                "time_since_reset": time.time() - self.metrics.last_reset
            }

    def _reset_metrics(self, current_time: float) -> None:
        """Reset per-minute metrics."""
        self.metrics.requests_per_minute = 0
        self.metrics.tokens_per_minute = 0
        self.metrics.last_reset = current_time

    def _calculate_backoff_time(self, overuse_ratio: float) -> float:
        """
        Calculate appropriate backoff time based on overuse ratio.
        
        Args:
            overuse_ratio: Ratio of current usage to limit
            
        Returns:
            Time to wait in seconds
        """
        base_time = 1.0  # Base second
        return base_time * (overuse_ratio - 1)