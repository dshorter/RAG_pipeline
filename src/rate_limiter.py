# src/rate_limiter.py

# Standard library imports
from dataclasses import dataclass
from datetime import datetime
import threading
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from  src.singleton_config import ConfigSingleton  
# Third-party imports
import tiktoken

class TokenizerModel(Enum):
    """Supported models for tokenization."""
    EMBEDDING =  ConfigSingleton( ).get_active_embedding_config( ).model_name  
    GPT35 = "gpt-3.5-turbo"
    GPT4 = ConfigSingleton().get_gpt_config().model_name 

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
    """Handles rate limiting for Azure OpenAI API calls."""
    
    def __init__(
        self,
        model_type: TokenizerModel = TokenizerModel.EMBEDDING,
        requests_per_minute_limit: int = 3500,
        tokens_per_minute_limit: int = 350000,
        safety_factor: float = 0.95
    ):
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
            self.logger.error(f"Failed to initialize tokenizer: {e}")
            self.tokenizer = None  # Will trigger fallback behavior

    def count_tokens(self, text: str) -> int:
        """Get precise token count using the appropriate tokenizer."""
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
        """Determine if request should be throttled based on current usage."""
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
        """Record metrics for a request with thread-safe updates."""
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
            float: Time to wait before next attempt
        """
        base_time = 2.0  # Use exponential backoff
        wait_time = base_time ** retry_count
        
        self.logger.warning(
            f"Rate limit hit. Retry {retry_count} "
            f"waiting {wait_time:.2f}s"
        )
        
        return wait_time

    def _reset_metrics(self, current_time: float) -> None:
        """Reset per-minute metrics."""
        self.metrics.requests_per_minute = 0
        self.metrics.tokens_per_minute = 0
        self.metrics.last_reset = current_time

    def _calculate_backoff_time(self, overuse_ratio: float) -> float:
        """Calculate appropriate backoff time based on overuse ratio."""
        base_time = 1.0  # Base second
        return base_time * (overuse_ratio - 1)

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics with thread-safe access."""
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
                "last_error_time": self.metrics.last_error_time
            }
        