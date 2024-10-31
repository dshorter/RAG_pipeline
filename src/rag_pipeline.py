import os
import logging
import time
from typing import List, Dict, Any
import uuid
from .knowledge_base import process_documents
from .document_chunker import chunk_document
from .metrics_collector import MetricsCollector
from .embedding_generator_factory import EmbeddingGeneratorFactory
from .rag_system import RAGSystem
from .pipeline_result import PipelineResult, ChunkInfo, ChunkMetrics, VectorMetrics
from .singleton_config import ConfigSingleton
import faiss
from .generation import Generator
import shutil            
from  .paths import  *             

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self):    

        self.config = ConfigSingleton()  
        self.doc_metadata = {}   
        self.db_path =  get_db_path()
        self.faiss_path = get_faiss_path()
        self.raw_docs_dir = get_raw_docs_dir()
        self.processed_docs_dir = get_processed_docs_dir()
        
        self.metrics_collector = MetricsCollector()
        self.embedding_generator = self._initialize_embedding_generator()
        self.rag_system = RAGSystem()
        # self.generator = Generator()  

        logger.info("RAG Pipeline initialized with config: %s", self.config.to_dict())

        # Ensure processed_docs_dir exists
        os.makedirs(self.processed_docs_dir, exist_ok=True)

    def _initialize_embedding_generator(self):
        pipeline_config = self.config.get_pipeline_config()
        active_embedding_config = self.config.get_active_embedding_config()
    
        return EmbeddingGeneratorFactory.create(
            generator_type=pipeline_config.embedding.provider,
            endpoint=active_embedding_config
        )

    def process_document(self, file_path: str) -> List[Dict]:
        """
        Process a document and return a list of processed document dictionaries.
        
        Args:
            file_path: Path to the document to process
            
        Returns:
            List of processed document dictionaries
        """
        logger.info(f"Processing document(s): {file_path}")
        try:
            # Ensure we always have a list of dictionaries
            result = process_documents(file_path)
            if not isinstance(result, list):
                result = [result]
                
            # Add unique document ID to each processed document
            for doc in result:
                if isinstance(doc, dict):
                    doc['document_id'] = uuid.uuid4().hex
                else:
                    logger.error(f"Invalid document format: {type(doc)}")
                    raise ValueError(f"Expected dictionary but got {type(doc)}")
            
            return result
        
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise
            
    def chunk_document(self, processed_doc: Dict) -> List[Dict]:
        logger.info("Chunking document")
        content = processed_doc['content']
        result = chunk_document(content, 
                                chunk_size=self.config.get_pipeline_config().chunk_size,
                                chunk_overlap=self.config.get_pipeline_config().chunk_overlap)
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

    def index_documents(self, prepared_chunks: List[Dict], document_id: str, doc_metadata: List[Dict]    ):    

        logger.info(f"Indexing {len(prepared_chunks)} chunks for document {document_id}")
        self.doc_metadata = doc_metadata
        for chunk_data in prepared_chunks:
            self.rag_system.add_vector(
                document_id=document_id,
                chunk=chunk_data['chunk'],
                vector=chunk_data['vector'],
                source=chunk_data['source'],
                start_index=chunk_data['start_index'],
                end_index=chunk_data['end_index'],
                additional_metadata=chunk_data.get('additional_metadata', {}), 
                doc_metadata=doc_metadata 
            )
        logger.info(f"Indexing completed for document {document_id}")    

    # def query(self, user_query: str) -> Dict[str, Any]:
    #     try:
    #         logger.info(f"Received user query: {user_query}")
    #         query_embedding = self.embedding_generator.generate_embedding(user_query)
            
    #         search_results = self.rag_system.search(query_embedding, k=10)
            
    #         response = self.generator.generate_response(user_query, search_results)
            
    #         return {
    #             "query": user_query,
    #             "response": response,
    #             "search_results": search_results
    #         }
    #     except Exception as e:
    #         logger.error(f"Error processing query: {str(e)}")
    #         raise

    def cleanup_processed_documents(self, processed_docs: List[str]):
        for doc_path in processed_docs:
            if os.path.exists(doc_path):
                processed_path = os.path.join(self.processed_docs_dir, os.path.basename(doc_path))
                shutil.move(doc_path, processed_path)
                logger.info(f"Moved processed document to: {processed_path}")
            else:
                logger.warning(f"Document not found for moving: {doc_path}")
        
        remaining_files = os.listdir(self.raw_docs_dir)
        if remaining_files:
            logger.info(f"Documents remaining in raw folder: {', '.join(remaining_files)}")
        else:
            logger.info("All documents in raw folder have been processed.")

    def run_pipeline(self, file_path: str) -> List[PipelineResult]:
        processed_docs = self.process_document(file_path)
        results = []
        successfully_processed = []

        for processed_doc in processed_docs:
            document_name = processed_doc['metadata'].get('title', 'Unknown Document')    
            document_id = processed_doc['document_id']            
            logger.info(f"Processing document: {document_name} (ID: {document_id})")

            
            try:
                chunking_result = chunk_document(
                    processed_doc['content'],
                    chunk_size=self.config.get_pipeline_config().chunk_size,
                    chunk_overlap=self.config.get_pipeline_config().chunk_overlap
                )
                
                chunks = chunking_result['chunks']
                chunk_metrics = chunking_result['metrics']
                
                chunk_infos = [ChunkInfo(text=chunk['text'],
                                         start_index=chunk['start_index'],
                                         end_index=chunk['end_index'])
                               for chunk in chunks]
                
                chunk_metrics = ChunkMetrics(
                    total_chunks=len(chunks),
                    avg_chunk_size=chunk_metrics['avg_chunk_size'],
                    max_chunk_size=chunk_metrics['max_chunk_size'],
                    min_chunk_size=chunk_metrics['min_chunk_size'],
                )
                
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
                    document_id=document_id,
                    document_name=document_name,
                    processed_text=processed_doc['content'],
                    metadata=processed_doc['metadata'],
                    chunks=chunk_infos,
                    chunk_metrics=chunk_metrics,
                    embeddings=embeddings,
                    vector_metrics=vector_metrics
                )                    

                self.index_documents(result.prepare_for_indexing(), document_id, processed_doc['metadata']  )
                successfully_processed.append(processed_doc['file_path'])
                results.append(result)
                
                logger.info(f"Pipeline execution completed for document: {document_name}")
                logger.info(result.summary())
            
            except Exception as e:
                logger.error(f"Failed to process document {document_name} (ID: {document_id}): {str(e)}")

        # Cleanup after processing all documents
        #  self.cleanup_processed_documents(successfully_processed)
        
        logger.info(f"Pipeline execution completed for all documents. Total documents processed: {len(results)}")    
        
        return results   
    

