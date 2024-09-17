from openai import AzureOpenAI
from azure.identity import ChainedTokenCredential, ManagedIdentityCredential, EnvironmentCredential, AzureCliCredential
from typing import List
import logging
from src.embedding_generator_base_class import EmbeddingGenerator  
from src.singleton_config import ConfigSingleton

class AzureOpenAIEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self, azure_endpoint: str, api_version: str, deployment: str):
        self.config = ConfigSingleton()
        
        self.azure_endpoint = self.config.get_active_embedding_config().api_base  # "https://edav-dev-openai-eastus2-shared.openai.azure.com"   #azure_endpoint
        self.api_version =  self.config.get_active_embedding_config().api_version #  "2023-05-15"   # api_version
        self.deployment = "api-shared-text-embedding-ada-v002-nofilter" #deployment
        self.client = self._initialize_client()
        self.model = "api-shared-text-embedding-ada-v002-nofilter"  # self.config.get_active_embedding_config( ).model_name  
        self._dimension =  self.config.get_active_embedding_config().dimension    # Known dimension for this model

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
            logging.error(f"Error generating embedding: {str(e)}")
            raise

    def generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        return [self.generate_embedding(chunk) for chunk in chunks]

    @property
    def dimension(self) -> int:
        return self._dimension
