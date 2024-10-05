import os
from typing import List, Dict, Any
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from src.singleton_config import ConfigSingleton


class Generator:
    def __init__(self ):
        self.config = ConfigSingleton( )
        self.client = self._initialize_client()

    def _initialize_client(self):
        credential = DefaultAzureCredential()
        return AzureOpenAI(
            azure_endpoint=self.config.get_gpt_config( ).api_base,
            api_version=self.config.get_gpt_config( ).api_version, 
            azure_ad_token=credential.get_token("https://cognitiveservices.azure.com/.default").token
        )

def generate_response(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        try:
            context = "\n\n".join([f"Chunk {i+1}: {result['text']}" for i, result in enumerate(search_results)])
            
            prompt = f"""You are an AI assistant tasked with answering questions based on the provided context. 
            Your goal is to provide accurate, relevant, and helpful answers.

            Question: {query}

            Context:
            {context}

            Instructions:
            1. Analyze the question and the provided context carefully.
            2. Synthesize information from the context to formulate your response, even if some chunks seem less relevant.
            3. If the context doesn't contain enough information to fully answer the question, say so and provide the best partial answer you can based on the available information.
            4. Do not make up information or use knowledge outside of the provided context.
            5. If you directly quote or closely paraphrase specific parts of the context, indicate which chunk it came from (e.g., "According to Chunk 2...").
            6. Provide a coherent and natural-sounding response that directly addresses the user's question.
            7. If there are contradictions in the context, acknowledge them and explain the different viewpoints.

            Answer:"""

            response = self.client.chat.completions.create(
                model=self.config['model_name'],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the given context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config['temperature'],
                max_tokens=self.config['max_tokens'],
                top_p=self.config['top_p'],
                frequency_penalty=self.config['frequency_penalty'],
                presence_penalty=self.config['presence_penalty']
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Error in generate_response: {str(e)}")
            return "I'm sorry, but I couldn't generate a  response at this time."