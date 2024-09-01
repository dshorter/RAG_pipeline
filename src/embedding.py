from openai import AzureOpenAI
from azure.identity import ChainedTokenCredential, ManagedIdentityCredential, EnvironmentCredential, AzureCliCredential
from typing import List
import logging
import os

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, azure_endpoint: str, api_version: str, deployment: str):
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self.deployment = deployment
        self.client = self._initialize_client()
        self.model = "text-embedding-ada-002"     #  text-embedding-ada-002      

    def _initialize_client(self):
        credential = ChainedTokenCredential(
            ManagedIdentityCredential(),
            EnvironmentCredential(),
            AzureCliCredential()
        )
        access_token = credential.get_token("https://cognitiveservices.azure.com/.default")
        
        return AzureOpenAI(
            api_key=access_token.token,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version
        )

    def generate_embedding(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model  
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        embeddings = []
        for i, chunk in enumerate(chunks):
            try:
                embedding = self.generate_embedding(chunk)
                embeddings.append(embedding)
                logger.info(f"Generated embedding for chunk {i+1}/{len(chunks)}")
            except Exception as e:
                logger.error(f"Failed to generate embedding for chunk {i+1}: {str(e)}")
        return embeddings