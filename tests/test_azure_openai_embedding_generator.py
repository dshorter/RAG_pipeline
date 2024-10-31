# test_azure_openai_embedding_generator.py

import pytest
import sys
import os
from datetime import datetime
import time
from unittest.mock import Mock, patch
import tiktoken
from typing import List, Dict, Any

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rate_limiter import AzureRateLimiter, TokenizerModel, RateLimitMetrics
from src.azure_openai_embedding_generator import AzureOpenAIEmbeddingGenerator

@pytest.fixture
def rate_limiter():
    """Create a rate limiter instance with test configurations."""
    return AzureRateLimiter(
        model_type=TokenizerModel.EMBEDDING,
        requests_per_minute_limit=100,  # Smaller limit for testing
        tokens_per_minute_limit=10000,
        safety_factor=0.95
    )

@pytest.fixture
def mock_generator():
    """Create a mock embedding generator with test configurations."""
    with patch('src.azure_openai_embedding_generator.AzureOpenAI'):
        generator = AzureOpenAIEmbeddingGenerator(
            azure_endpoint="https://test.openai.azure.com",
            api_version="2023-05-15",
            deployment="test-deployment"
        )
        return generator

class TestTokenCounting:
    """Tests for token counting functionality."""
    
    def test_basic_token_counting(self, rate_limiter):
        """Test basic token counting for simple text."""
        text = "This is a test"
        token_count = rate_limiter.count_tokens(text)
        assert token_count > 0
        assert isinstance(token_count, int)

    def test_token_counting_with_special_characters(self, rate_limiter):
        """Test token counting with special characters and numbers."""
        text = "Testing 123! How about some @ special # characters?"
        token_count = rate_limiter.count_tokens(text)
        assert token_count > 0
        assert isinstance(token_count, int)

    def test_empty_string_token_counting(self, rate_limiter):
        """Test token counting with empty string."""
        assert rate_limiter.count_tokens("") == 0

    def test_long_text_token_counting(self, rate_limiter):
        """Test token counting with longer text."""
        text = "This is a longer text " * 100
        token_count = rate_limiter.count_tokens(text)
        assert token_count > 100  # Should be significantly more tokens

class TestRateLimitChecking:
    """Tests for rate limit checking functionality."""

    def test_initial_state_no_throttle(self, rate_limiter):
        """Test initial state should not throttle."""
        should_throttle, wait_time = rate_limiter.should_throttle()
        assert not should_throttle
        assert wait_time == 0

    def test_approaching_request_limit(self, rate_limiter):
        """Test throttling when approaching request limit."""
        # Simulate requests approaching limit
        for _ in range(90):  # 90% of our 100 rpm limit
            rate_limiter.record_request("test text", success=True)
        
        should_throttle, wait_time = rate_limiter.should_throttle()
        assert should_throttle
        assert wait_time > 0

    def test_reset_after_minute(self, rate_limiter):
        """Test metrics reset after one minute."""
        # Record some requests
        rate_limiter.record_request("test", success=True)
        
        # Simulate time passing
        original_time = time.time()
        with patch('time.time') as mock_time:
            mock_time.return_value = original_time + 61  # Just over a minute
            
            should_throttle, wait_time = rate_limiter.should_throttle()
            assert not should_throttle
            assert wait_time == 0
            assert rate_limiter.metrics.requests_per_minute == 0

class TestMetricsTracking:
    """Tests for metrics tracking functionality."""

    def test_successful_request_metrics(self, rate_limiter):
        """Test metrics tracking for successful requests."""
        text = "This is a test request"
        rate_limiter.record_request(text, success=True)
        
        metrics = rate_limiter.get_current_metrics()
        assert metrics["total_requests"] == 1
        assert metrics["successful_requests"] == 1
        assert metrics["failed_requests"] == 0
        assert metrics["total_tokens_processed"] > 0

    def test_failed_request_metrics(self, rate_limiter):
        """Test metrics tracking for failed requests."""
        text = "This is a test request"
        rate_limiter.record_request(text, success=False, error="Test error")
        
        metrics = rate_limiter.get_current_metrics()
        assert metrics["total_requests"] == 1
        assert metrics["successful_requests"] == 0
        assert metrics["failed_requests"] == 1
        assert metrics["last_error"] == "Test error"
        assert isinstance(metrics["last_error_time"], datetime)

class TestGeneratorIntegration:
    """Tests for integration with AzureOpenAIEmbeddingGenerator."""

    @pytest.fixture
    def mock_azure_client(self):
        """Fixture to provide a consistent mock Azure client."""
        with patch('src.azure_openai_embedding_generator.AzureOpenAI') as mock_azure:
            # Create a mock client with the structure we need
            mock_client = Mock()
            mock_client.embeddings.create = Mock()
            mock_azure.return_value = mock_client
            yield mock_azure

    @pytest.fixture
    def test_generator(self, mock_azure_client):
        """Fixture to provide a generator with mocked Azure client."""
        generator = AzureOpenAIEmbeddingGenerator(
            azure_endpoint="https://test.openai.azure.com",
            api_version="2023-05-15",
            deployment="test-deployment"
        )
        # Ensure the generator is using our mocked client
        generator.client = mock_azure_client.return_value
        return generator

    def test_generator_initialization(self, test_generator):
        """Test generator initializes with rate limiter."""
        assert test_generator.rate_limiter is not None
        assert isinstance(test_generator.rate_limiter, AzureRateLimiter)

    def test_rate_limited_embedding_generation(self, test_generator, mock_azure_client):
        """Test embedding generation with rate limiting."""
        # Set up mock response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_azure_client.return_value.embeddings.create.return_value = mock_response

        # Generate embedding
        embedding = test_generator.generate_embedding("test text")
        
        # Check we got expected result
        assert embedding == [0.1, 0.2, 0.3]
        
        # Check rate limiter metrics were updated
        metrics = test_generator.get_rate_limit_metrics()
        assert metrics["total_requests"] == 1
        assert metrics["successful_requests"] == 1

    def test_rate_limit_error_handling(self, test_generator, mock_azure_client):
        """Test handling of rate limit errors."""
        # Configure mock to raise rate limit error
        mock_azure_client.return_value.embeddings.create.side_effect = Exception("rate_limit exceeded")
        
        # Verify the error is raised and caught properly
        with pytest.raises(Exception) as exc_info:
            test_generator.generate_embedding("test text")
        
        assert "rate_limit exceeded" in str(exc_info.value)
        
        # Verify rate limit hit was recorded
        metrics = test_generator.get_rate_limit_metrics()
        assert metrics["rate_limit_hits"] >= 1

if __name__ == '__main__':
    pytest.main([__file__])