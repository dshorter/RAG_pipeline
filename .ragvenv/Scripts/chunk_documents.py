import os
import sys
import json

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from  src.document_chunker import load_processed_document, chunk_document

def main():
    # Update this path to point to your processed document
    file_path = '../data/processed/Biosafety_Guidance.txt'
    output_path = '../data/chunked/chunked_Biosafety_Guidance.json'

    content, metadata = load_processed_document(file_path)
    chunked_document = chunk_document(content, metadata,
                                      model_token_limit=4096,  # Adjust as needed
                                      num_chunks=3,
                                      avg_question_tokens=50,
                                      max_response_tokens=500,
                                      prompt_tokens=50)

    # Print some information about the chunks
    print(f"Total number of chunks: {len(chunked_document)}")
    print(f"\nFirst chunk:")
    print(json.dumps(chunked_document[0], indent=2))
    print(f"\nLast chunk:")
    print(json.dumps(chunked_document[-1], indent=2))

    # Save the chunked document
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunked_document, f, indent=2)

    print(f"Chunked document saved to {output_path}")

if __name__ == "__main__":
    main()