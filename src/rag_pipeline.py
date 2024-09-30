import os
import logging
import sqlite3
import time
from typing import List, Dict, Any  
from src.config import Document, ChunkMetrics  
from src.knowledge_base import process_documents
from .document_chunker import chunk_document
from .metrics_collector import MetricsCollector
from .embedding_generator_factory import EmbeddingGeneratorFactory
from .rag_system import RAGSystem
from .pipeline_result import PipelineResult, ChunkInfo, ChunkMetrics, VectorMetrics
from .singleton_config import ConfigSingleton    
import faiss    
from .generation import Generator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, config: Dict):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.embedding_generator = self._initialize_embedding_generator()
        
        # Use get() method with default values
        conn =  self.config.get('conn')
        index = self.config.get('index')
        faiss_index_path = self.config.get('faiss_index_path', './data/faiss_index.bin')
        
        self.rag_system = RAGSystem(conn=conn, index=index, faiss_index_path=faiss_index_path)
        self.generator = Generator(self.config.get('gpt', {}))
        logger.info("RAG Pipeline initialized with config: %s", config)

    def _initialize_embedding_generator(self):
        return EmbeddingGeneratorFactory.create(
            generator_type=self.config['pipeline']['embedding']['provider'],
            **self.config['pipeline']
        )
        
        # Define paths relative to the project folder
        # Get the project root directory (one level up from the current file's directory)
        project_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        db_path = os.path.join(project_folder, 'data', 'metadata.db')
        faiss_path = os.path.join(project_folder, 'data', 'faiss_index.bin')

        # Initialize SQLite connection
        sql_conn = sqlite3.connect(db_path)

        # Initialize FAISS index   
        dimension = self.configS.get_active_embedding_config().dimension     # Set your vector dimension
        index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))        
        
        #  self.rag_system = RAGSystem(conn=conn, index=index, faiss_index_path=faiss_path)
        
        logger.info("RAG Pipeline initialized with config: %s", config)

    def _initialize_embedding_generator(self):
        return EmbeddingGeneratorFactory.create(
            generator_type=self.config['pipeline']['embedding']['provider'],
            **self.config['pipeline']
        )

    def process_document(self, file_path: str) -> Dict:
        logger.info("Processing document: %s", file_path)
        try:
            processed_doc = process_documents(file_path)
            logger.info("Document processed successfully. Word count: %d", processed_doc['metadata']['word_count'])
            
            return processed_doc
        
        except Exception as e:
            logger.error("Error processing document: %s", str(e))
            raise

    def chunk_document(self, processed_doc: Dict) -> List[Dict]:
        logger.info("Chunking document")
        content = processed_doc['content']
        result = chunk_document(content, 
                                chunk_size=self.config.get('chunk_size', 500), 
                                chunk_overlap=self.config.get('chunk_overlap', 50))
        chunks = result['chunks']
        metrics = result['metrics']
        metrics['document'] = processed_doc['metadata']['title']
        self.metrics_collector.log_metrics("chunks", metrics)
        logger.info(f"Document chunked into {len(chunks)} parts")
        return chunks

    def generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        start_time = time.time()
        
        try:
            embeddings = self.embedding_generator.generate_embeddings(chunks)
            
            end_time = time.time()
            embedding_time = end_time - start_time
            
            metrics = {
                'num_embeddings': len(embeddings),
                'embedding_dimension': len(embeddings[0]) if embeddings else 0,
                'embedding_generation_start_time': start_time,
                'embedding_generation_end_time': end_time,
                'embedding_generation_time': embedding_time
            }
            self.metrics_collector.log_metrics("embeddings", metrics)
            
            logger.info(f"Generated {len(embeddings)} embeddings in {embedding_time:.2f} seconds")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def index_documents(self, prepared_chunks: List[Dict]):
        logger.info(f"Indexing {len(prepared_chunks)} documents")
        for chunk_data in prepared_chunks:
            self.rag_system.add_vector(
            chunk=chunk_data['chunk'],
            vector=chunk_data['vector'],
            source=chunk_data['source'],
            start_index=chunk_data['start_index'],
            end_index=chunk_data['end_index'],
            additional_metadata=chunk_data.get('additional_metadata', {})
        )

        logger.info("Indexing completed")

    def query(self, user_query: str) -> Dict[str, Any]:
        try:
            logger.info(f"Received user query: {user_query}")
            query_embedding = self.embedding_generator.generate_embedding(user_query)
            
            # Perform the search, retrieving more results than we might typically use
            search_results = self.rag_system.search(query_embedding, k=10)
            
            # Generate response using the LLM
            response = self.generator.generate_response(user_query, search_results)
            
            return {
                "query": user_query,
                "response": response,
                "search_results": search_results  # Optionally include for transparency
            }
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise


    def run_pipeline(self, file_path: str) -> PipelineResult:
        document_name = os.path.basename(file_path)
        processed_doc = self.process_document(file_path)

        cs =  ConfigSingleton( )
        cs.document = Document("","this_doc_title","DRSC","ORR","7/1/2024","100",
                                "A summary","no tags","json_here" )
        
        
        # Use the updated chunk_document function
        chunking_result = chunk_document(processed_doc['content'], 
                chunk_size=self.config.get('chunk_size', 500),
                chunk_overlap=self.config.get('chunk_overlap', 50))
    
        chunks = chunking_result['chunks']
        chunk_metrics = chunking_result['metrics']        
        
        # Create ChunkInfo objects from the standardized chunk dictionaries
        chunk_infos = [ChunkInfo(text=chunk['text'] ,
                                start_index=chunk['start_index'],
                                end_index=chunk['end_index'])
        for chunk in chunks]
        
        chunk_metrics = ChunkMetrics(
            total_chunks=len(chunks),
            avg_chunk_size=chunk_metrics['avg_chunk_size'],
            max_chunk_size=chunk_metrics['max_chunk_size'],
            min_chunk_size=chunk_metrics['min_chunk_size'],
            # chunking_time=chunk_metrics['chunking_time']
        )
        # self.metrics_collector.log_metrics(metric_type="chunks",metrics=chunk_metrics )  
        
        embeddings = self.generate_embeddings([chunk['text'] for chunk in chunks])
        embedding_metrics = self.metrics_collector.get_metrics("embeddings")
        vector_metrics = VectorMetrics(
            num_embeddings=embedding_metrics['num_embeddings'],
            embedding_dimension=embedding_metrics['embedding_dimension'],
            embedding_generation_start_time=embedding_metrics['embedding_generation_start_time'],
            embedding_generation_end_time=embedding_metrics['embedding_generation_end_time'],
            embedding_generation_time=embedding_metrics['embedding_generation_time']
        )
        
        result = PipelineResult(
            document_name=document_name,
            processed_text=processed_doc['content'],
            metadata=processed_doc['metadata'],
            chunks=chunk_infos,
            chunk_metrics=chunk_metrics,
            embeddings=embeddings,
            vector_metrics=vector_metrics
        )
        
        self.index_documents(result.prepare_for_indexing())
        logger.info("Pipeline execution completed")
        logger.info(result.summary())
        
        return result