import os
from typing import List, Dict, Any
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from src.singleton_config import ConfigSingleton


class Generator:
    def __init__(self, config: Dict[str, Any]):
        self.config = ConfigSingleton( )
        self.client = self._initialize_client()

    def _initialize_client(self):
        credential = DefaultAzureCredential()
        return AzureOpenAI(
            azure_endpoint=self.config.get_gpt_config( ).api_base,
            api_version=self.config.get_gpt_config( ).api_version, 
            azure_ad_token=credential.get_token("https://cognitiveservices.azure.com/.default").token
        )

    def generate_response(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        context = "\n".join([chunk['text'] for chunk in retrieved_chunks])
        prompt = f"Based on the following context, answer the question: {query}\n\nContext: {context}\n\nAnswer:"

        try:
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
            print(f"Error in generate_response: {str(e)}")
            return "I'm sorry, but I couldn't generate a response at this time."