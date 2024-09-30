import os
import re
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize
from nltk.corpus import stopwords
import nltk 
from .singleton_config import ConfigSingleton 

import logging  


# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Set up NLTK data path (you may need to adjust this path)
NLTK_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'nltk_data')
nltk.data.path.append(NLTK_DATA_PATH)

logger = logging.getLogger(__name__)

def prepare_document(text, title="", author="", date=""):
    logger.info(f"Preparing document: {title}")
    logger.debug(f"Original text length: {len(text)}")

    # Clean the text
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = text.lower()  # Lowercase
    logger.debug(f"Cleaned text length: {len(text)}")

    try:
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        logger.info(f"Number of sentences: {len(sentences)}")
        logger.debug(f"First sentence: {sentences[0][:100]}...")
        
        # Tokenize words
        tokens = word_tokenize(text)
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
    documnet = ConfigSingleton().document     
    documnet.title = title
    documnet.document_length = len(tokens)  

    metadata = {
        "title": title,
        "author": author,
        "date": date,
        "word_count": len(tokens)
    }

    return cleaned_text, metadata    


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
    content = read_text_file(file_path)
    filename = os.path.basename(file_path)
    cleaned_text, metadata = prepare_document(content, title=filename)
    
    return {
        "content": cleaned_text,
        "metadata": metadata
    }

def process_documents(input_path):
    if os.path.isfile(input_path):
        return process_single_document(input_path)
    elif os.path.isdir(input_path):
        processed_docs = []
        for filename in os.listdir(input_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(input_path, filename)
                processed_docs.append(process_single_document(file_path))
        return processed_docs
    else:
        raise ValueError(f"Invalid input path: {input_path}")

# Example usage (if run as a script)
if __name__ == "__main__":
    # You can add test code here to process a sample document
    sample_file = "../data/raw/Biosafety_Guidance.txt"  # Adjust this path as needed
    result = process_documents(sample_file)
    print(f"Processed document metadata: {result['metadata']}")
    print(f"First 100 characters of processed content: {result['content'][:100]}...")