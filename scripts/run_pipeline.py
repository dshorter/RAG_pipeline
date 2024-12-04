import os
import sys
import json
import logging
import time
from typing import List    

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_pipeline import RAGPipeline
from src.document_chunker import chunk_document
from src.config import Configuration  # Assuming you have a Configuration class
from src.singleton_config import ConfigSingleton      
from src.logging_config import setup_rag_logging, get_logger


# Set up logging
setup_rag_logging(log_dir='logs', unified_log=True)
logger = get_logger(__name__)



def main():
    start_time = time.time()
    logger.info("=== Starting RAG Pipeline Execution ===")

    config = ConfigSingleton()          
    logger.info("Configuration loaded: %s", json.dumps(config.to_dict(), indent=2))

    pipeline = RAGPipeline()
    raw_docs_dir = config.get_pipeline_config().raw_docs_dir
    
    if os.path.isdir(raw_docs_dir):
        files = [f for f in os.listdir(raw_docs_dir) if os.path.isfile(os.path.join(raw_docs_dir, f))]
        logger.info(f"Found {len(files)} files to process: {', '.join(files)}")
        
        for i, filename in enumerate(files, 1):
            file_path = os.path.join(raw_docs_dir, filename)
            logger.info(f"\n=== Processing File {i}/{len(files)}: {filename} ===")
            file_start_time = time.time()
            
            try:
                result = pipeline.run_pipeline(file_path)
                file_duration = time.time() - file_start_time
                logger.info(f"Completed processing {filename} in {file_duration:.2f} seconds")
            except Exception as e:
                logger.error(f"Failed to process {filename}: {str(e)}")
                continue
    else:
        logger.error(f"Directory not found: {raw_docs_dir}")

    total_duration = time.time() - start_time
    logger.info(f"\n=== RAG Pipeline Execution Completed in {total_duration:.2f} seconds ===")

   
if __name__ == "__main__":
    main()
    