import os
import logging
import time
from typing import List
from src.knowledge_base import process_documents
from src.document_chunker import chunk_document
from src.metrics_collector import MetricsCollector
from src.embedding import EmbeddingGenerator
from src.rag_system import RAGSystem  # New import

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, config, metrics_collector=None):
        self.config = config
        self.metrics_collector = metrics_collector if metrics_collector is not None else MetricsCollector()
        self.embedding_generator = self._initialize_embedding_generator()
        self.rag_system = RAGSystem(vector_dimension=1536)  # Adjust dimension as needed
        logger.info("RAG Pipeline initialized with config: %s", config)

    def _initialize_embedding_generator(self):
            azure_openai_endpoint = self.config.get('AZURE_OPENAI_ENDPOINT') or os.environ.get('AZURE_OPENAI_ENDPOINT')
            azure_openai_api_version = self.config.get('AZURE_OPENAI_API_VERSION') or os.environ.get('AZURE_OPENAI_API_VERSION')
            azure_openai_deployment = self.config.get('AZURE_OPENAI_DEPLOYMENT') or os.environ.get('AZURE_OPENAI_DEPLOYMENT')

            if not all([azure_openai_endpoint, azure_openai_api_version, azure_openai_deployment]):
                raise ValueError("Missing required Azure OpenAI configuration.")

            return EmbeddingGenerator(
                azure_endpoint=azure_openai_endpoint,
                api_version=azure_openai_api_version,
                deployment=azure_openai_deployment
            )
            
    def generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        start_time = time.time()
        
        embeddings = self.embedding_generator.generate_embeddings(chunks)
        freeze 
        end_time = time.time()
        embedding_time = end_time - start_time
    
        # Log metrics
        metrics = {
            'num_embeddings': len(embeddings),
            'embedding_dimension': len(embeddings[0]) if embeddings else 0,
            'embedding_generation_time': embedding_time
        }
        self.metrics_collector.log_metrics(metrics)
        
        logger.info(f"Generated {len(embeddings)} embeddings in {embedding_time:.2f} seconds")
        return embeddings

    def _get_openai_api_key(self):
        # First, try to get the API key from the config
        api_key = self.config.get('openai_api_key')
        
        # If not in config, try to get it from environment variable
        if not api_key:
            api_key = os.environ.get('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("OpenAI API key not found in config or environment variables")
        
        return api_key

    def process_document(self, file_path):
        logger.info("Processing document: %s", file_path)
        try:
            processed_doc = process_documents(file_path)
            logger.info("Document processed successfully. Word count: %d", processed_doc['metadata']['word_count'])
            return processed_doc
        except Exception as e:
            logger.error("Error processing document: %s", str(e))
            raise
    def chunk_document(self, processed_doc):
        logger.info("Chunking document")
        content = processed_doc['content']
        result = chunk_document(content)
        chunks = result['chunks']
        metrics = result['metrics']
        metrics['document'] = processed_doc['metadata']['title']
        self.metrics_collector.log_metrics(metrics)
        logger.info(f"Document chunked into {len(chunks)} parts")
        return chunks

    def generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        start_time = time.time()
        
        try:
            embeddings = self.embedding_generator.generate_embeddings(chunks)
            
            end_time = time.time()
            embedding_time = end_time - start_time
            
            # Log metrics
            metrics = {
                'num_embeddings': len(embeddings),
                'embedding_dimension': len(embeddings[0]) if embeddings else 0,
                'embedding_generation_time': embedding_time
            }
            self.metrics_collector.log_metrics(metrics)
            
            logger.info(f"Generated {len(embeddings)} embeddings in {embedding_time:.2f} seconds")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            # You might want to handle this error more gracefully depending on your use case
            raise

    def index_documents(self, chunks, embeddings):
        logger.info(f"Indexing {len(chunks)} documents")
        for chunk, embedding in zip(chunks, embeddings):
            self.rag_system.add_chunk(
                chunk=chunk['text'],
                vector=embedding,
                source=chunk['source'],
                start_index=chunk['start_index'],
                end_index=chunk['end_index']
            )
        logger.info("Indexing completed")

    def query(self, user_query: str):
        logger.info(f"Received user query: {user_query}")
        query_embedding = self.embedding_generator.generate_embedding(user_query)
        results = self.rag_system.search(query_embedding)
        # Process results and generate a response (you'll need to implement this part)
        return f"Found {len(results)} relevant chunks. Implement response generation here."

    def run_pipeline(self, file_path):
        processed_doc = self.process_document(file_path)
        chunks = self.chunk_document(processed_doc)
        embeddings = self.generate_embeddings([chunk['text'] for chunk in chunks])
        self.index_documents(chunks, embeddings)
        logger.info("Pipeline execution completed")

# ... (keep the rest of the file, including the main block if present)