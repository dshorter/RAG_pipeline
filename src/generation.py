"""
Generation Module - 2025-01-28-GEN03

PROMPT ENGINEERING NOTICE:
The prompt engineering in this module is carefully crafted and optimized.
Do not modify the prompt structure or content during refactoring.
Changes should focus only on:
- Technical implementation
- Response handling
- Metrics collection
- Error handling
"""

import logging
import os
import time
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from datetime import datetime
from src.singleton_config import ConfigSingleton
from src.logging_config import get_logger
from src.metrics_collector import MetricsCollector
from src.logging_config import get_logger, setup_rag_logging  

class Generator:
    def __init__(self):
        self.config = ConfigSingleton()      
        # Set up logging    
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s' )
        setup_rag_logging(log_dir='logs', unified_log=True)
        logger = logging.getLogger(__name__)        
        
        self.logger = get_logger('generator')
        self.client = self._initialize_client()
        self.metrics_collector = MetricsCollector()

    def _initialize_client(self) -> AzureOpenAI:
        try:
            credential = DefaultAzureCredential()
            return AzureOpenAI(
                azure_endpoint=self.config.get_gpt_config().api_base,
                api_version=self.config.get_gpt_config().api_version, 
                azure_ad_token=credential.get_token("https://cognitiveservices.azure.com/.default").token
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize client: {str(e)}")
            raise

    def _get_prompt_instructions(self, allow_training_data: bool) -> str:
        # PRESERVED: Original prompt engineering
        base_instructions = """1. Analyze the question and provided context carefully.
2. Build your response using information from the context.
3. When citing information, indicate which chunk it came from using [1], [2], etc.
4. Provide a coherent and natural-sounding response that directly addresses the user's question.
5. If there are contradictions in the context, acknowledge them and explain the different viewpoints.
6. Make your citations precise and specific to help users track information sources.
7. Synthesize information from multiple sources when appropriate.
8. Maintain professional language and clarity throughout the response.
9. Structure your response logically, building from fundamental points to more complex details.
10. If quoting directly, use quotation marks and appropriate citation."""

        if allow_training_data:
            additional_instructions = """11. If the context doesn't fully address the question:
    - First, clearly state what aspects the context covers
    - Then, introduce additional information with: [AI Knowledge: ...]
    - Clearly distinguish between context-based information and AI knowledge
12. Always prioritize context over training data when both are available
13. Keep AI knowledge additions focused and relevant to the query
14. Ensure AI knowledge supplements rather than replaces context
15. When using AI knowledge, explain how it connects to or enhances the context
16. If there's uncertainty in the AI knowledge, acknowledge it explicitly
17. Use AI knowledge to provide broader context or explanatory framework
18. Maintain clear separation between citations and AI knowledge additions"""
        else:
            additional_instructions = """11. Use ONLY information found in the provided context
12. If the context doesn't contain enough information:
    - Clearly state what aspects you can address from the context
    - Explicitly note which parts of the question cannot be answered
    - Suggest what additional information might be helpful
13. Do not supplement with any external knowledge
14. Focus on providing the most relevant context-based information available
15. If the context is insufficient, explain what specific information is missing
16. Maintain transparency about the limitations of available information
17. Suggest related questions that could be answered with the available context
18. When appropriate, explain how additional context could enhance the answer"""

        return base_instructions + "\n\n" + additional_instructions

    def _prepare_citations(self, search_results: List[Dict[str, Any]]) -> Dict[str, str]:
        citations = {}
        for i, result in enumerate(search_results, 1):
            source_info = result.get('source_info', {})
            title = source_info.get('title', 'Unknown Document')
            if isinstance(title, str):
                title = title.strip("b'").strip('"').replace('.pdf', '')

            citation_parts = [f"'{title}'"]

            author = source_info.get('author', 'Unknown')
            if author and author != 'Unknown':
                citation_parts.append(f"by {author}")

            relevance = result.get('relevance_score', 0.0)
            citation_parts.append(f"(Relevance: {relevance:.2f})")

            citations[str(i)] = f"[{i}] {' '.join(citation_parts)}"
        return citations

    def generate_response(self, query: str, search_results: List[Dict[str, Any]], 
                         allow_training_data: bool = False) -> Dict[str, Any]:
        start_time = time.time()
        self.logger.info(f"Generating response for query: {query}")
        self.logger.info(f"Search results count: {len(search_results)}")
        self.logger.info(f"Allow training data: {allow_training_data}")
        
        # Debug logging for search results
        for i, result in enumerate(search_results):
            self.logger.debug(f"Search result {i+1}:")
            self.logger.debug(f"  Chunk text: {result.get('chunk_text', '')[:100]}...")
            self.logger.debug(f"  Score: {result.get('relevance_score', 0)}")
            self.logger.debug(f"  Source: {result.get('source_info', {})}")

        try:
            if not search_results:
                self.logger.warning("No search results found")
                return {
                    "response_text": "I apologize, but I couldn't find any relevant information in the current context to answer your question. Could you please rephrase or provide more details?",
                    "citations": {},
                    "used_training_data": False
                }

            context_chunks = [f"Chunk {i+1}: {result['chunk_text']}" 
                            for i, result in enumerate(search_results)]
            citations = self._prepare_citations(search_results)

            context = "\n\n".join(context_chunks)
            instructions = self._get_prompt_instructions(allow_training_data)

            prompt = f"""You are an AI assistant tasked with answering questions based on the provided context. 
            Your goal is to provide accurate, relevant, and well-cited answers.

            Question: {query}

            Context:
            {context}

            Instructions:
            {instructions}

            Answer:"""

            response = self.client.chat.completions.create(
                model=self.config.get_gpt_config().model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the given context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.get_gpt_config().temperature,
                max_tokens=self.config.get_gpt_config().max_tokens,
                top_p=self.config.get_gpt_config().top_p,
                frequency_penalty=self.config.get_gpt_config().frequency_penalty,
                presence_penalty=self.config.get_gpt_config().presence_penalty
            )

            generated_response = response.choices[0].message.content.strip()
            citation_text = "\n\nSources:\n" + "\n".join(citations.values())
            full_response = generated_response + "\n" + "-"*40 + citation_text

            self.logger.info("Response generated successfully")
            self.logger.debug(f"Response length: {len(generated_response)}")
            self.logger.debug(f"Citations count: {len(citations)}")

            self.metrics_collector.collect(
                operation='response_generation',
                component='generator',
                metrics={
                    'duration_ms': (time.time() - start_time) * 1000,
                    'prompt_tokens': len(prompt.split()),
                    'response_tokens': len(generated_response.split()),
                    'num_citations': len(citations),
                    'chunks_used': len(search_results),
                    'model': {
                        'name': self.config.get_gpt_config().model_name,
                        'temperature': self.config.get_gpt_config().temperature
                    },
                    'training_data_used': allow_training_data
                }
            )

            return {
                "response_text": full_response,
                "citations": citations,
                "used_training_data": allow_training_data
            }

        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            self.metrics_collector.collect(
                operation='response_generation',
                component='generator',
                metrics={
                    'duration_ms': (time.time() - start_time) * 1000,
                    'success': False,
                    'error': str(e)
                }
            )
            raise