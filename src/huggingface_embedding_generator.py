from transformers import AutoTokenizer, AutoModel
import torch
from typing import List
from embedding_generator_base_class   import EmbeddingGeneratorBaseClass

class HuggingFaceEmbeddingGenerator(EmbeddingGeneratorBaseClass):
    def __init__(self, model_name="ada"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self._dimension = self.model.config.hidden_size

    def generate_embedding(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

    def generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        return [self.generate_embedding(chunk) for chunk in chunks]

    @property
    def dimension(self) -> int:
        return self._dimension
