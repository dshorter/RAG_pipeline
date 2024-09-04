import os
import sys
import json
import logging
from typing import List

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_pipeline import RAGPipeline  
from src.document_chunker import chunk_document
from src.config import RAG_CONFIG, EMBEDDING_GENERATOR_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting RAG pipeline execution")

    # config = {
    #     'raw_docs_dir': os.path.join('data', 'raw'),
    #     'processed_docs_dir': os.path.join('data', 'processed'),    
    #     'embedding_generator_type': os.getenv("EMBEDDING_GENERATOR_TYPE")
    # }

    config = {
        **RAG_CONFIG,
        'embedding_generator_config': EMBEDDING_GENERATOR_CONFIG[RAG_CONFIG['embedding_generator_type']]
    }
    

    logger.info("Configuration loaded: %s", config)

    pipeline = RAGPipeline(config)

    # Process the Biosafety file
    biosafety_file = os.path.join(config['raw_docs_dir'], 'Biosafety_Guidance.txt')
    logger.info("Processing file: %s", biosafety_file)
    processed_doc = pipeline.process_document(biosafety_file)

    # Save the processed document
    output_file = os.path.join(config['processed_docs_dir'], 'processed_Biosafety_Guidance.json')
    logger.info("Saving processed document to: %s", output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_doc, f, indent=2)

    logger.info("Processed document saved successfully")
    logger.info("Document metadata: %s", processed_doc['metadata'])
    logger.info("First 100 characters of processed content: %s", processed_doc['content'][:100])

    chunks = pipeline.chunk_document(processed_doc)
    embeddings =   pipeline.generate_embeddings(chunks)
    # Placeholder calls for future implementations
    pipeline.index_documents(chunks, embeddings)


    # Example query (will just print a placeholder message for now)
    user_query = "What are the main safety measures for handling select agents?"
    logger.info("Executing example query: %s", user_query)
    response = pipeline.query(user_query)
    logger.info("Query response: %s", response)

    logger.info("RAG pipeline execution completed")

if __name__ == "__main__":
    main()