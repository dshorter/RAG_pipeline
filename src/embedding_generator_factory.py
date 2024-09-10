from .embedding_generator_base_class import EmbeddingGenerator
from .azure_openai_embedding_generator import AzureOpenAIEmbeddingGenerator
from .huggingface_embedding_generator import HuggingFaceEmbeddingGenerator

class EmbeddingGeneratorFactory:
    @staticmethod
    def create(generator_type: str, **kwargs) -> EmbeddingGenerator:
        if generator_type == "azure_openai":
            return AzureOpenAIEmbeddingGenerator(
                azure_endpoint=kwargs.get("azure_endpoint"),
                api_version=kwargs.get("api_version"),
                deployment=kwargs.get("deployment")
            )
        elif generator_type == "huggingface":
            return HuggingFaceEmbeddingGenerator(
                model_name=kwargs.get("model_name", "ada")
            )
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")