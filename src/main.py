# src/main.py

import asyncio
from client.query_handling.handle_query import handle_query
from rag_pipeline import  RAGPipeline  
from config import Configuration

async def main():
    # Load configuration
    config = Configuration('config.yml')

    # Initialize RAG pipeline (this remains synchronous)
    rag_pipeline = RAGPipeline(config.to_dict())

    # Example usage
    query = "What are the main safety measures for handling select agents?"
    response = await handle_query(query, rag_pipeline, config.get('query_preprocessing', {}))
    print(f"Query: {query}")
    print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main())

