import os
import sys
import json

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_pipeline import RAGPipeline

def main():
    config = {
        'raw_docs_dir': os.path.join('data', 'raw'),
        'processed_docs_dir': os.path.join('data', 'processed'),
    }

    pipeline = RAGPipeline(config)

    # Process the Biosafety file
    biosafety_file = os.path.join(config['raw_docs_dir'], 'Biosafety_Guidance.txt')
    processed_doc = pipeline.process_document(biosafety_file)

    # Save the processed document
    output_file = os.path.join(config['processed_docs_dir'], 'processed_Biosafety_Guidance.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_doc, f, indent=2)

    print(f"Processed document saved to: {output_file}")
    print(f"Document metadata: {processed_doc['metadata']}")
    print(f"First 500 characters of processed content: {processed_doc['content'][:500]}...")

    # Placeholder calls for future implementations
    pipeline.chunk_document(processed_doc)
    pipeline.generate_embeddings([])  # This would normally take chunks as input
    pipeline.index_documents([])  # This would normally take embeddings as input

    # Example query (will just print a placeholder message for now)
    user_query = "What are the main safety measures for handling select agents?"
    response = pipeline.query(user_query)
    print(f"Query: {user_query}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()