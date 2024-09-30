# src/client/query_handling/handle_query.py

from ..preprocessing.preprocess_query import preprocess_query
from ...core.rag_system.rag_core import RAGPipeline
from typing import Dict
import asyncio

async def handle_query(query: str, rag_pipeline: RAGPipeline, config: Dict = {}) -> str:
    """
    Asynchronously handle a user query.

    Args:
    query (str): The user's query.
    rag_pipeline (RAGPipeline): An instance of the RAG pipeline.
    config (Dict): Configuration options. Default is an empty dict.

    Returns:
    str: The response to the user's query.
    """
    # Preprocess the query asynchronously
    preprocessed_query = await preprocess_query(query, config.get('preprocessing', {}))

    # Use the RAG pipeline to get a response
    # Since RAGPipeline.query() is not async, we'll run it in an executor
    response = await asyncio.to_thread(rag_pipeline.query, preprocessed_query)

    return response