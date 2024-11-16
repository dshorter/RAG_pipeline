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
        """Initialize Azure OpenAI client with proper authentication."""
        credential = DefaultAzureCredential()
        return AzureOpenAI(
            azure_endpoint=self.config.get_gpt_config().api_base,
            api_version=self.config.get_gpt_config().api_version, 
            azure_ad_token=credential.get_token("https://cognitiveservices.azure.com/.default").token
        )

    def _get_prompt_instructions(self, allow_training_data: bool) -> str:
        """Get appropriate instructions based on whether training data is allowed."""
        base_instructions = """1. Analyze the question and provided context carefully.
2. Build your response using information from the context.
3. When citing information, indicate which chunk it came from using [1], [2], etc.
4. Provide a coherent and natural-sounding response.
5. If there are contradictions in the context, acknowledge them and explain the different viewpoints.
6. Make your citations precise and specific to help users track information sources."""

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

 

    def generate_response(self, query: str, search_results: List[Dict[str, Any]], 
                        allow_training_data: bool = False) -> Dict[str, Any]:
        try:
            context_chunks = []
            citations = {}
            
            for i, result in enumerate(search_results, 1):
                context_chunks.append(f"Chunk {i}: {result['chunk_text']}")
                
                title = result.get('source_info', {}).get('title', 'Unknown Document')
                if isinstance(title, str):
                    title = title.strip("b'").strip('"').replace('.pdf', '')
                
                citation_parts = []
                citation_parts.append(f"'{title}'")
                
                author = result.get('source_info', {}).get('author', 'Unknown')
                if author and author != 'Unknown':
                    citation_parts.append(f"by {author}")
                
                start = result.get('source_info', {}).get('start_index')
                end = result.get('source_info', {}).get('end_index')
                if start is not None and end is not None:
                    citation_parts.append(f"section {start}-{end}")
                
                relevance = result.get('relevance_score', 0.0)
                citation_parts.append(f"(Relevance: {relevance:.2f})")
                
                if result.get('metadata'):
                    extra_meta = "; ".join(f"{k}: {v}" for k, v in result['metadata'].items() 
                                        if k not in ['title', 'author', 'source'])
                    if extra_meta:
                        citation_parts.append(f"Additional info: {extra_meta}")

                citations[str(i)] = f"[{i}] {' | '.join(citation_parts)}"

            context = "\n\n".join(context_chunks)
            instructions = self._get_prompt_instructions(allow_training_data)
            
            footnote_instruction = """
            Use numbered footnotes [1] in your response and I will add the full citations at the end.
            Each major fact or quote should have its own footnote.
            Multiple footnotes can reference the same source if needed.
            """
            
            prompt = f"""You are an AI assistant tasked with answering questions based on the provided context. 
            Your goal is to provide accurate, relevant, and well-cited answers.

            Question: {query}

            Context:
            {context}

            Instructions:
            {instructions}
            {footnote_instruction}

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
            footnotes = "\n\nSources:\n" + "\n".join(citations.values())
            full_response = generated_response + "\n" + "-"*40 + footnotes

            return {
                "response_text": full_response,
                "citations": citations,
                "used_training_data": allow_training_data
            }

        except Exception as e:
            self.logger.error(f"Error in generate_response: {str(e)}")
            return {
                "response_text": "I'm sorry, but I couldn't generate a response at this time.",
                "citations": {},
                "used_training_data": False
            }
        
#################################
    def xgenerate_response(self, query: str, search_results: List[Dict[str, Any]], 
                        allow_training_data: bool = False) -> Dict[str, Any]:
        """Generate a response with comprehensive metadata citations."""
   
        print("\nDEBUG: Generator Received Search Results: ==========================")
        for i, result in enumerate(search_results):
            print(f"\nResult {i+1}:")
            print(f"Keys: {result.keys()}")
            print(f"Source Info: {result.get('source_info', {})}")
            print(f"Metadata: {result.get('metadata', {})}")
            print(f"Title: {result.get('source_info', {}).get('title', 'No title')}")
            print(f"Author: {result.get('source_info', {}).get('author', 'No author')}")
            print(f"Relevance: {result.get('relevance_score', 'No score')}")    

        try:
            # Format context and prepare citations
            context_chunks = []
            citations = {}
            
            for i, result in enumerate(search_results, 1):
                context_chunks.append(f"Chunk {i}: {result['chunk_text']}")
                
                # Extract metadata more thoroughly
                source_info = result.get('source_info', {})
                metadata = result.get('metadata', {})
                
                # Build rich citation
                citation_parts = []
                
                # Add title and source
                if source_info.get('title'):
                    citation_parts.append(f"'{source_info['title']}'")
                if source_info.get('source'):
                    citation_parts.append(f"from {source_info['source']}")
                
                # Add author if available
                if source_info.get('author'):
                    citation_parts.append(f"by {source_info['author']}")
                
                # Add section/position information
                if source_info.get('start_index') is not None and source_info.get('end_index') is not None:
                    citation_parts.append(f"section {source_info['start_index']}-{source_info['end_index']}")
                
                # Add relevance score
                relevance = result.get('relevance_score', 0.0)
                citation_parts.append(f"(Relevance: {relevance:.2f})")
                
                # Add any additional metadata that might be useful
                if metadata:
                    extra_meta = "; ".join(f"{k}: {v}" for k, v in metadata.items() 
                                        if k not in ['title', 'author', 'source'])
                    if extra_meta:
                        citation_parts.append(f"Additional info: {extra_meta}")
                
                # Compile the full citation
                citations[str(i)] = f"[{i}] {', '.join(citation_parts)}"

            context = "\n\n".join(context_chunks)
            instructions = self._get_prompt_instructions(allow_training_data)
            
            # Modify instructions to request footnotes format
            footnote_instruction = """
            Use numbered footnotes [1] in your response and I will add the full citations at the end.
            Each major fact or quote should have its own footnote.
            Multiple footnotes can reference the same source if needed.
            """
            
            prompt = f"""You are an AI assistant tasked with answering questions based on the provided context. 
            Your goal is to provide accurate, relevant, and well-cited answers.

            Question: {query}

            Context:
            {context}

            Instructions:
            {instructions}
            {footnote_instruction}

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
            
            # Add citations as footnotes
            footnotes = "\n\nSources:\n" + "\n".join(citations.values())
            full_response = generated_response + footnotes
            
            self.logger.info("Response generated successfully", 
                        extra={
                            'component': 'generator',
                            'operation': 'generate_response',
                            'allow_training_data': allow_training_data,
                            'num_chunks_used': len(search_results)
                        })

            return {
                "response_text": full_response,
                "citations": citations,
                "used_training_data": allow_training_data
            }

        except Exception as e:
            self.logger.error(f"Error in generate_response: {str(e)}", 
                            extra={
                                'component': 'generator',
                                'operation': 'generate_response',
                                'error': str(e)
                            })
            return {
                "response_text": "I'm sorry, but I couldn't generate a response at this time.",
                "citations": {},
                "used_training_data": False
            }