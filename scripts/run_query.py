import os
import sys
import json
import logging
from typing import List    

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_pipeline import RAGPipeline
from src.document_chunker import chunk_document
from src.config import Configuration  # Assuming you have a Configuration class
from src.singleton_config import ConfigSingleton      

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():

    logger.info("Starting RAG pipeline execution")

    # Load configuration
    config  = ConfigSingleton()          
    logger.info("Configuration loaded: %s", json.dumps(config.to_dict(), indent=2))

    # Initialize RAGPipeline
    pipeline = RAGPipeline(config.to_dict())

    # Example query (will just print a placeholder message for now)
    user_query = "What are the main safety measures for handling select agents?"
    logger.info("Executing example query: %s", user_query)
    response = pipeline.query(user_query)
    logger.info("Query response: %s", response)

    # logger.info("RAG pipeline execution completed")

if __name__ == "__main__":
    main()