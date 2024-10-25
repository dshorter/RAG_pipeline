# test_embedding_generator.py        

import os  
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.azure_openai_embedding_generator import AzureOpenAIEmbeddingGenerator
from src.logging_config import setup_rag_logging
import time

def test_embeddings():
    # Setup logging
    logger = setup_rag_logging(unified_log=True)
    logger.info("Starting embedding generator tests")

    # Initialize the embedding generator
    try:
        generator = AzureOpenAIEmbeddingGenerator(
            azure_endpoint="https://edav-dev-openai-eastus2-shared.openai.azure.com",
            api_version="2023-05-15",
            deployment="api-shared-text-embedding-ada-v002-nofilter"
        )
        logger.info("Embedding generator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize generator: {str(e)}")
        return

    # Test cases
    test_cases = [
        {
            "name": "Single Short Text",
            "input": "This is a short test text.",
            "expected_success": True
        },
        {
            "name": "Single Long Text",
            "input": "This is a longer test text. " * 50,  # About 250 words
            "expected_success": True
        },
        {
            "name": "Batch Processing",
            "input": [
                "First test text.",
                "Second test text.",
                "Third test text."
            ],
            "expected_success": True
        }
    ]

    # Run single embedding tests
    for test in test_cases:
        logger.info(f"Running test: {test['name']}")
        try:
            if isinstance(test['input'], list):
                start_time = time.time()
                embeddings = generator.generate_embeddings(test['input'])
                duration = time.time() - start_time
                
                logger.info(f"Batch processing completed",
                           extra={
                               'test_name': test['name'],
                               'chunk_count': len(test['input']),
                               'duration': duration,
                               'embeddings_generated': len(embeddings)
                           })

            else:
                start_time = time.time()
                embedding = generator.generate_embedding(test['input'])
                duration = time.time() - start_time
                
                logger.info(f"Single embedding completed",
                           extra={
                               'test_name': test['name'],
                               'text_length': len(test['input']),
                               'duration': duration,
                               'embedding_length': len(embedding)
                           })

        except Exception as e:
            logger.error(f"Test failed: {test['name']}",
                        extra={
                            'error': str(e),
                            'error_type': type(e).__name__
                        })

    # Test error handling with very large input
    logger.info("Testing large input handling")
    try:
        very_large_text = "test " * 50000  # Should trigger API limits
        embedding = generator.generate_embedding(very_large_text)
    except Exception as e:
        logger.info("Large input test failed as expected",
                   extra={
                       'error': str(e),
                       'error_type': type(e).__name__
                   })

    # Test batch processing with mixed content
    logger.info("Testing batch processing with mixed content")
    mixed_batch = [
        "Short text",
        "Medium length text with more content " * 10,
        "Very long text " * 100
    ]
    
    try:
        start_time = time.time()
        batch_embeddings = generator.generate_embeddings(mixed_batch)
        duration = time.time() - start_time
        
        logger.info("Mixed batch processing completed",
                   extra={
                       'batch_size': len(mixed_batch),
                       'duration': duration,
                       'embeddings_generated': len(batch_embeddings)
                   })
    except Exception as e:
        logger.error("Mixed batch processing failed",
                    extra={
                        'error': str(e),
                        'error_type': type(e).__name__
                    })

if __name__ == "__main__":
    test_embeddings()

