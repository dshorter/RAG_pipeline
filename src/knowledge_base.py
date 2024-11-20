import os
import re
from typing import Dict, Tuple
import dateutil
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize
from nltk.corpus import stopwords
import nltk   
import magic 
from tika import parser  
import datetime

import yaml    
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


        # For text files, try to extract embedded metadata
        if file_type == "text/plain":
            embedded_metadata, content = extract_embedded_metadata(content)
            # Merge embedded metadata with any metadata from Tika
            metadata.update(embedded_metadata)    


        # Clean and organize metadata
        cleaned_metadata = clean_metadata(metadata)
        
        document_metadata = {
            "title": cleaned_metadata.pop("title", filename),
            "author": cleaned_metadata.pop("author", "Unknown"),
            "publish_date": cleaned_metadata.pop("publish_date", "Unknown"),
            "file_type": file_type,
            "content_type": cleaned_metadata.pop("content_type", file_type),
            "document_length": len(content.split())  # Word count
        }

        # Add any remaining cleaned metadata as additional
        if cleaned_metadata:
            document_metadata["additional"] = cleaned_metadata

        # Clean the text
        cleaned_text = prepare_document(content, title=filename)

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

########################################  
def clean_metadata(metadata):
    field_synonyms = {
        'title': ['title', 'name', 'documenttitle'],
        'author': ['author', 'creator', 'contributor', 'writer'],
        'publish_date': ['createdate', 'modifydate', 'creationdate', 'publicationdate', 'date'],
        'content_type': ['contenttype', 'mimetype', 'filetype'],
    }

    cleaned_metadata = {}
    for key, value in metadata.items():
        clean_key_name = clean_key(key)
        if is_useful_value(value):
            matched = False
            for field, synonyms in field_synonyms.items():
                if any(synonym in clean_key_name for synonym in synonyms):
                    if field == 'publish_date':
                        cleaned_metadata[field] = parse_date(value)
                    else:
                        cleaned_metadata[field] = value
                    matched = True
                    break
            if not matched:
                cleaned_metadata[clean_key_name] = value

    return cleaned_metadata

def clean_key(key):
    # Strip prefixes and convert to lowercase
    return re.sub(r'^.*?:', '', key).lower()

def is_useful_value(value):
    if value is None or value == '':
        return False
    if isinstance(value, str) and value.lower() in ['null', 'none', 'unknown']:
        return False
    return True

from datetime import datetime
import dateutil.parser

def parse_date(date_input):
    if not date_input:
        return None
    
    # If date_input is a list, join it into a string
    if isinstance(date_input, list):
        date_string = ' '.join(date_input)
    else:
        date_string = str(date_input)
    
    try:
        # First, try parsing with dateutil
        return dateutil.parser.parse(date_string).isoformat()
    except (ValueError, TypeError):
        # If dateutil fails, try our custom formats
        for fmt in ['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y:%m:%d %H:%M:%S']:
            try:
                return datetime.strptime(date_string, fmt).isoformat()
            except ValueError:
                continue
    # If all parsing attempts fail, return the original string
    return date_string    

def extract_embedded_metadata(content: str) -> Tuple[Dict[str, str], str]:
    """
    Extract metadata from text content using various formats.
    Returns tuple of (metadata_dict, remaining_content)
    """
    # Try YAML-style header
    yaml_match = re.match(r'^---\n(.*?)\n---\n(.*)', content, re.DOTALL)
    if yaml_match:
        try:
            metadata = yaml.safe_load(yaml_match.group(1))
            return metadata, yaml_match.group(2).strip()
        except yaml.YAMLError:
            pass

    # Try XML-style metadata
    xml_match = re.match(r'<\?xml:metadata\s+(.*?)\s*\?>\s*(.*)', content, re.DOTALL)
    if xml_match:
        metadata = {}
        for pair in re.finditer(r'(\w+)="([^"]*)"', xml_match.group(1)):
            metadata[pair.group(1)] = pair.group(2)
        return metadata, xml_match.group(2).strip()

    # Try comment block metadata
    comment_match = re.match(r'/\*@metadata\n(.*?)\n@end\*/\s*(.*)', content, re.DOTALL)
    if comment_match:
        metadata = {}
        meta_lines = comment_match.group(1).strip().split('\n')
        for line in meta_lines:
            if ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip()] = value.strip()
        return metadata, comment_match.group(2).strip()

    # No metadata found
    return {}, content    

# Example usage (if run as a script)
if __name__ == "__main__":
    # You can add test code here to process a sample document
    sample_file = "../data/raw/Biosafety_Guidance.pdf"  # Adjust this path as needed
    result = process_documents(sample_file)
    print(f"Processed document metadata: {result['metadata']}")
    print(f"First 100 characters of processed content: {result['content'][:100]}...")