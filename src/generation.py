import os
from typing import List, Dict, Any
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from src.singleton_config import ConfigSingleton
from src.logging_config import get_logger

class Generator:
    def __init__(self):
        self.config = ConfigSingleton()
        self.logger = get_logger('generator')
        self.client = self._initialize_client()

    def _initialize_client(self):
        credential = DefaultAzureCredential()
        return AzureOpenAI(
            azure_endpoint=self.config.get_gpt_config().api_base,
            api_version=self.config.get_gpt_config().api_version, 
            azure_ad_token=credential.get_token("https://cognitiveservices.azure.com/.default").token
        )

    def _get_prompt_instructions(self, allow_training_data: bool) -> str:
        """Get appropriate instructions based on response mode."""
        base_instructions = """1. Analyze the question and provided context carefully.
2. Build your primary response using information from the context.
3. If directly quoting or closely paraphrasing, indicate the chunk (e.g., "According to Chunk 2...").
4. Provide a coherent and natural-sounding response.
5. If there are contradictions in the context, acknowledge them and explain the different viewpoints.
6. Do not fabricate or make up any information."""

        if allow_training_data:
            additional_instructions = """7. If the context doesn't fully address the question:
   - First, clearly state what aspects the context covers
   - Then, introduce additional information with: [AI Knowledge: ...]
8. Always prioritize context over training data
9. Keep AI knowledge additions focused and relevant
10. Ensure AI knowledge supplements rather than replaces context"""
        else:
            additional_instructions = """7. Use ONLY information found in the provided context
8. If the context doesn't contain enough information:
   - Clearly state what aspects you can address from the context
   - Explicitly note which parts of the question cannot be answered
9. Do not supplement with any external knowledge
10. Focus on providing the most relevant context-based information available"""

        return base_instructions + "\n\n" + additional_instructions

    def generate_response(self, query: str, search_results: List[Dict[str, Any]], allow_training_data: bool = False) -> str:
        try:
            self.logger.debug("Starting response generation", 
                            extra={
                                'component': 'generator',
                                'operation': 'generate_response',
                                'query_length': len(query),
                                'num_chunks': len(search_results),
                                'mode': 'context_plus_ai' if allow_training_data else 'context_only'
                            })

            # Prepare context from search results
            context = "\n\n".join([f"Chunk {i+1}: {result['chunk_text']}" for i, result in enumerate(search_results)])

            # Get appropriate instructions based on mode
            instructions = self._get_prompt_instructions(allow_training_data)
            
            prompt = f"""You are an AI assistant tasked with answering questions based on the provided context. 
            Your goal is to provide accurate, relevant, and helpful answers.

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

            self.logger.info("Response generated successfully", 
                           extra={
                               'component': 'generator',
                               'operation': 'generate_response',
                               'response_length': len(response.choices[0].message.content),
                               'mode': 'context_plus_ai' if allow_training_data else 'context_only'
                           })

            return response.choices[0].message.content.strip()

        except Exception as e:
            self.logger.error(f"Error in generate_response: {str(e)}", 
                            extra={
                                'component': 'generator',
                                'operation': 'generate_response',
                                'error': str(e),
                                'mode': 'context_plus_ai' if allow_training_data else 'context_only'
                            })
            return "I'm sorry, but I couldn't generate a response at this time."