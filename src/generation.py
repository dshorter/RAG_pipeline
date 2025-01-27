
# src/generation.py

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from typing import List, Dict, Any
import logging
import time
from src.metrics_collector import MetricsCollector
from src.singleton_config import ConfigSingleton
from src.logging_config import get_logger

class Generator:
    def __init__(self):
        self.config = ConfigSingleton()
        self.logger = get_logger('generator')
        self.client = self._initialize_client()    
        self.metrics_collector = MetricsCollector( ) 

    def generate_response(self, query: str, search_results: List[Dict[str, Any]], 
                        allow_training_data: bool = False) -> Dict[str, Any]:
        start_time = time.time()
        try:
            context = "\n\n".join([f"Chunk {i+1}: {result['chunk_text']}" 
                                 for i, result in enumerate(search_results)])
            
            prompt = self._build_prompt(query, context, allow_training_data)
            
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

            response_text = response.choices[0].message.content.strip()
            citations = self._format_citations(search_results)
            
            # Fire and forget metrics
            self.metrics_collector.collect(
                operation='response_generation',
                component='generator',
                metrics={
                    'duration_ms': (time.time() - start_time) * 1000,
                    'prompt_tokens': len(prompt.split()),
                    'response_tokens': len(response_text.split()),
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
                "response_text": response_text,
                "citations": citations,
                "used_training_data": allow_training_data
            }

        except Exception as e:
            self.logger.error(f"Generation failed: {str(e)}")
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

    def _initialize_client(self):
        credential = DefaultAzureCredential()
        return AzureOpenAI(
            azure_endpoint=self.config.get_gpt_config().api_base,
            api_version=self.config.get_gpt_config().api_version, 
            azure_ad_token=credential.get_token("https://cognitiveservices.azure.com/.default").token
        )

    def _build_prompt(self, query: str, context: str, allow_training_data: bool) -> str:
        base_prompt = f"""Answer the following question based on the provided context.

Question: {query}

Context:
{context}

Instructions:
1. Analyze the question and context carefully
2. Use information from the context to formulate your response
3. Cite sources using [1], [2], etc.
4. Make your response clear and natural"""

        if allow_training_data:
            base_prompt += "\n5. You may supplement with AI knowledge if needed"

        return base_prompt

    def _format_citations(self, search_results: List[Dict[str, Any]]) -> Dict[str, str]:
        citations = {}
        for i, result in enumerate(search_results, 1):
            source_info = result['source_info']
            citation_parts = []
            
            if title := source_info.get('title'):
                citation_parts.append(f"'{title}'")
            if author := source_info.get('author'):
                citation_parts.append(f"by {author}")
            if source := source_info.get('source'):
                citation_parts.append(f"from {source}")
            
            citations[str(i)] = f"[{i}] {' | '.join(citation_parts)}"
        
        return citations
