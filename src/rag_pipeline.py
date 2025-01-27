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
from .paths import *             

class RAGPipeline:
    def __init__(self):    
        self.config = ConfigSingleton()  
        self.doc_metadata = {}   
        self.db_path = get_db_path()
        self.faiss_path = get_faiss_path()
        self.raw_docs_dir = get_raw_docs_dir()
        self.processed_docs_dir = get_processed_docs_dir()
        self.metrics_collector = MetricsCollector(self.db_path)
        self.embedding_generator = self._initialize_embedding_generator()
        self.rag_system = RAGSystem()
        self.logger = logging.getLogger(__name__)
        os.makedirs(self.processed_docs_dir, exist_ok=True)

    def _initialize_embedding_generator(self):
        pipeline_config = self.config.get_pipeline_config()
        active_embedding_config = self.config.get_active_embedding_config()
        return EmbeddingGeneratorFactory.create(
            generator_type=pipeline_config.embedding.provider,
            endpoint=active_embedding_config
        )

    def process_document(self, file_path: str) -> List[Dict]:
        start_time = time.time()
        try:
            result = process_documents(file_path)
            if not isinstance(result, list):
                result = [result]
                
            for doc in result:
                doc['document_id'] = str(uuid.uuid4())

            self.metrics_collector.collect(
                'document_processing',
                'rag_pipeline',
                {
                    'duration_ms': (time.time() - start_time) * 1000,
                    'num_documents': len(result),
                    'file_path': file_path
                }
            )
            return result
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            self.metrics_collector.collect(
                'document_processing',
                'rag_pipeline',
                {
                    'duration_ms': (time.time() - start_time) * 1000,
                    'success': False,
                    'error': str(e)
                }
            )
            raise

    def chunk_document(self, processed_doc: Dict) -> List[Dict]:
        start_time = time.time()
        try:
            content = processed_doc['content']
            result = chunk_document(content, 
                                  chunk_size=self.config.get_pipeline_config().chunk_size,
                                  chunk_overlap=self.config.get_pipeline_config().chunk_overlap)
            
            self.metrics_collector.collect(
                'document_chunking',
                'rag_pipeline',
                {
                    'duration_ms': (time.time() - start_time) * 1000,
                    'document_id': processed_doc.get('document_id'),
                    'num_chunks': len(result['chunks']),
                    'chunk_metrics': result['metrics']
                }
            )
            return result['chunks']
        except Exception as e:
            self.logger.error(f"Chunking failed: {str(e)}")
            self.metrics_collector.collect(
                'document_chunking',
                'rag_pipeline',
                {
                    'duration_ms': (time.time() - start_time) * 1000,
                    'success': False,
                    'error': str(e)
                }
            )
            raise

    def generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        start_time = time.time()
        try:
            embeddings = self.embedding_generator.generate_embeddings(chunks)
            
            self.metrics_collector.collect(
                'embedding_generation',
                'rag_pipeline',
                {
                    'duration_ms': (time.time() - start_time) * 1000,
                    'num_chunks': len(chunks),
                    'num_embeddings': len(embeddings)
                }
            )
            return embeddings
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            self.metrics_collector.collect(
                'embedding_generation',
                'rag_pipeline',
                {
                    'duration_ms': (time.time() - start_time) * 1000,
                    'success': False,
                    'error': str(e)
                }
            )
            raise

    def index_documents(self, prepared_chunks: List[Dict], document_id: str, doc_metadata: List[Dict]):
        start_time = time.time()
        try:
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
            
            self.metrics_collector.collect(
                'document_indexing',
                'rag_pipeline',
                {
                    'duration_ms': (time.time() - start_time) * 1000,
                    'document_id': document_id,
                    'num_chunks': len(prepared_chunks)
                }
            )
        except Exception as e:
            self.logger.error(f"Indexing failed: {str(e)}")
            self.metrics_collector.collect(
                'document_indexing',
                'rag_pipeline',
                {
                    'duration_ms': (time.time() - start_time) * 1000,
                    'success': False,
                    'error': str(e)
                }
            )
            raise

    def run_pipeline(self, file_path: str) -> List[PipelineResult]:
        pipeline_start_time = time.time()
        try:
            processed_docs = self.process_document(file_path)
            results = []
            
            for processed_doc in processed_docs:
                doc_start_time = time.time()
                document_name = processed_doc['metadata'].get('title', 'Unknown Document')    
                document_id = processed_doc['document_id']            
                
                try:
                    chunks = self.chunk_document(processed_doc)
                    embeddings = self.generate_embeddings([chunk['text'] for chunk in chunks])
                    
                    result = PipelineResult(
                        document_id=document_id,
                        document_name=document_name,
                        processed_text=processed_doc['content'],
                        metadata=processed_doc['metadata'],
                        chunks=[ChunkInfo(text=c['text'], start_index=c['start_index'], 
                                        end_index=c['end_index']) for c in chunks],
                        embeddings=embeddings
                    )

                    self.index_documents(result.prepare_for_indexing(), document_id, processed_doc['metadata'])
                    results.append(result)
                    
                    self.metrics_collector.collect(
                        'document_pipeline',
                        'rag_pipeline',
                        {
                            'duration_ms': (time.time() - doc_start_time) * 1000,
                            'document_id': document_id,
                            'success': True,
                            'num_chunks': len(chunks),
                            'num_embeddings': len(embeddings)
                        }
                    )
                except Exception as e:
                    self.logger.error(f"Failed to process document {document_name}: {str(e)}")
                    self.metrics_collector.collect(
                        'document_pipeline',
                        'rag_pipeline',
                        {
                            'duration_ms': (time.time() - doc_start_time) * 1000,
                            'document_id': document_id,
                            'success': False,
                            'error': str(e)
                        }
                    )

            self.metrics_collector.collect(
                'pipeline_execution',
                'rag_pipeline',
                {
                    'duration_ms': (time.time() - pipeline_start_time) * 1000,
                    'total_documents': len(processed_docs),
                    'successful_documents': len(results),
                    'file_path': file_path
                }
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            self.metrics_collector.collect(
                'pipeline_execution',
                'rag_pipeline',
                {
                    'duration_ms': (time.time() - pipeline_start_time) * 1000,
                    'success': False,
                    'error': str(e)
                }
            )
            raise

