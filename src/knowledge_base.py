import os
import re
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize
from nltk.corpus import stopwords
import nltk   
import magic 
from tika import parser 
from .singleton_config import ConfigSingleton 
from .rag_system import RAGSystem  

import logging  


# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Set up NLTK data path (you may need to adjust this path)
NLTK_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'nltk_data')
nltk.data.path.append(NLTK_DATA_PATH)

logger = logging.getLogger(__name__)
def prepare_document(text, title=""):
    logger.info(f"Preparing document: {title}")
    logger.debug(f"Original text length: {len(text)}")

    # Clean the text
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = text.lower()  # Lowercase
    logger.debug(f"Cleaned text length: {len(text)}")

    try:
        # Use PunktSentenceTokenizer for sentence tokenization
        sentence_tokenizer = PunktSentenceTokenizer()
        sentences = sentence_tokenizer.tokenize(text)
        logger.info(f"Number of sentences: {len(sentences)}")
        
        # Tokenize words in each sentence
        tokens = [word for sentence in sentences for word in word_tokenize(sentence)]
        logger.info(f"Number of tokens: {len(tokens)}")

        # Remove stop words
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stop_words]
        logger.info(f"Number of tokens after stop word removal: {len(tokens)}")

    except Exception as e:
        logger.error(f"Error during tokenization: {str(e)}")
        logger.info("Falling back to simple tokenization")
        tokens = text.split()
        logger.info(f"Number of tokens (simple): {len(tokens)}")

    # Rejoin the text
    cleaned_text = ' '.join(tokens)
    logger.info(f"Final cleaned text length: {len(cleaned_text)}")

    return cleaned_text

    logger.info(f"Preparing document: {title}")
    logger.debug(f"Original text length: {len(text)}")

    # Clean the text
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = text.lower()  # Lowercase
    logger.debug(f"Cleaned text length: {len(text)}")

    try:
        # Use PunktSentenceTokenizer for sentence tokenization
        sentence_tokenizer = PunktSentenceTokenizer()
        sentences = sentence_tokenizer.tokenize(text)
        logger.info(f"Number of sentences: {len(sentences)}")
        logger.debug(f"First sentence: {sentences[0][:100]}...")
        
        # Tokenize words in each sentence
        tokens = [word for sentence in sentences for word in word_tokenize(sentence)]
        logger.info(f"Number of tokens: {len(tokens)}")
        logger.debug(f"First 10 tokens: {tokens[:10]}")
    except Exception as e:
        logger.error(f"Error during tokenization: {str(e)}")
        logger.info("Falling back to simple tokenization")
        tokens = text.split()
        logger.info(f"Number of tokens (simple): {len(tokens)}")

    # Remove stop words
    try:
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stop_words]
        logger.info(f"Number of tokens after stop word removal: {len(tokens)}")
    except LookupError:
        logger.warning("Stopwords not available. Skipping stopword removal.")

    # Rejoin the text
    cleaned_text = ' '.join(tokens)
    logger.info(f"Final cleaned text length: {len(cleaned_text)}")

    # Create metadata
    metadata = {
        "title": title,
        "author": author,
        "date": date,
        "word_count": len(tokens)
    }

    return cleaned_text, metadata

def read_text_file(file_path): 
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='iso-8859-1') as file:
            return file.read()

def process_single_document(file_path):
    try:
        file_type = magic.from_file(file_path, mime=True)
        parsed = parser.from_file(file_path)
        
        content = parsed.get("content", "")
        metadata = parsed.get("metadata", {})
        filename = os.path.basename(file_path)

        # Extract our standard metadata fields
        document_metadata = {
            "title": metadata.pop("title", filename),
            "author": metadata.pop("Author", "Unknown"),
            "publish_date": metadata.pop("Creation-Date", "Unknown"),
            "file_type": file_type,
            "content_type": metadata.pop("Content-Type", "Unknown"),
            "document_length": len(content.split())  # Word count
        }

        # Clean the text
        cleaned_text = prepare_document(content, title=filename)

        # Add any remaining metadata fields to an 'additional' field
        if metadata:
            document_metadata["additional"] = metadata

        return {
            "content": cleaned_text,
            "metadata": document_metadata
        }
    except Exception as e:
        logger.error(f"Error processing document {file_path}: {str(e)}")
        raise  # Re-raise the exception after logging
    
def process_documents(input_path):
    if os.path.isfile(input_path):
        return process_single_document(input_path)
    elif os.path.isdir(input_path):
        processed_docs = []
        
        for filename in os.listdir(input_path):
            file_path = os.path.join(input_path, filename)
            processed_docs.append(process_single_document(file_path))

        return processed_docs
    else:
        raise ValueError(f"Invalid input path: {input_path}")

# Example usage (if run as a script)
if __name__ == "__main__":
    # You can add test code here to process a sample document
    sample_file = "../data/raw/Biosafety_Guidance.pdf"  # Adjust this path as needed
    result = process_documents(sample_file)
    print(f"Processed document metadata: {result['metadata']}")
    print(f"First 100 characters of processed content: {result['content'][:100]}...")