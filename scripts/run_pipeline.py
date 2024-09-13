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

    # Process the Biosafety file
    biosafety_file = os.path.join(config.get_pipeline_config().raw_docs_dir, 'Biosafety_Guidance.txt')
    logger.info("Processing file: %s", biosafety_file)
    
    # Run the pipeline and get the result
    result = pipeline.run_pipeline(biosafety_file)

    # Log the summary of the pipeline result
    logger.info("Pipeline execution completed. Summary:\n%s", result.summary())

    # Save the processed document (optional)
    output_file = os.path.join(config.get_pipeline_config().processed_docs_dir, f'processed_{result.document_name}.json')
    logger.info("Saving processed document to: %s", output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'document_name': result.document_name,
            'processed_text': result.processed_text,
            'metadata': result.metadata,
            'chunks': [{'text': chunk.text, 'start_index': chunk.start_index, 'end_index': chunk.end_index} for chunk in result.chunks],
            'chunk_metrics': result.chunk_metrics.__dict__,
            'vector_metrics': result.vector_metrics.__dict__
        }, f, indent=2)

    logger.info("Processed document saved successfully")

    # Example query (will just print a placeholder message for now)
    user_query = "What are the main safety measures for handling select agents?"
    logger.info("Executing example query: %s", user_query)
    response = pipeline.query(user_query)
    logger.info("Query response: %s", response)

    logger.info("RAG pipeline execution completed")

if __name__ == "__main__":
    main()