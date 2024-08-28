import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import json

nltk.download('punkt', quiet=True)

def advanced_chunk(text, max_chunk_size=500, overlap=100):
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for paragraph in paragraphs:
        sentences = sent_tokenize(paragraph)
        
        for sentence in sentences:
            sentence_size = len(word_tokenize(sentence))
            
            if current_size + sentence_size > max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                overlap_size = 0
                for s in reversed(current_chunk):
                    overlap_size += len(word_tokenize(s))
                    if overlap_size >= overlap:
                        break
                current_chunk = current_chunk[-overlap_size:]
                current_size = sum(len(word_tokenize(s)) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_size += sentence_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def load_processed_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Parse the content as JSON
    doc_data = json.loads(content)
    
    return doc_data['content'], doc_data['metadata']

def chunk_document(content, metadata, max_chunk_size=500, overlap=100):
    chunks = advanced_chunk(content, max_chunk_size, overlap)
    
    chunked_doc = []
    for i, chunk in enumerate(chunks, 1):
        chunked_doc.append({
            'chunk_id': f"{metadata['title']}_chunk_{i}",
            'content': chunk,
            'metadata': {
                'source_document': metadata['title'],
                'chunk_number': i,
                'total_chunks': len(chunks),
                'word_count': len(word_tokenize(chunk))
            }
        })
    
    return chunked_doc

def calculate_max_chunk_size(model_token_limit, num_chunks, avg_question_tokens, max_response_tokens, prompt_tokens):
    available_tokens = model_token_limit - avg_question_tokens - max_response_tokens - prompt_tokens
    max_chunk_size = available_tokens // num_chunks
    return max_chunk_size

# Example calculation for GPT-3.5-turbo
model_token_limit = 4096
num_chunks = 3  # Number of chunks we want to retrieve
avg_question_tokens = 50  # Average length of user questions
max_response_tokens = 500  # Maximum length we want for the model's response
prompt_tokens = 50  # Any additional prompt text

max_chunk_size = calculate_max_chunk_size(model_token_limit, num_chunks, avg_question_tokens, max_response_tokens, prompt_tokens)

    print(f"Maximum chunk size: {max_chunk_size} tokens")


# Main execution
file_path = 'path/to/your/processed/Biosafety_Guidance.txt'  # Update this path
content, metadata = load_processed_document(file_path)

chunked_document = chunk_document(content, metadata)

# Print some information about the chunks
print(f"Total number of chunks: {len(chunked_document)}")
print(f"\nFirst chunk:")
print(json.dumps(chunked_document[0], indent=2))
print(f"\nLast chunk:")
print(json.dumps(chunked_document[-1], indent=2))

# Optionally, save the chunked document
with open('chunked_Biosafety_Guidance.json', 'w', encoding='utf-8') as f:
    json.dump(chunked_document, f, indent=2)