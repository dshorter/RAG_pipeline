import os
import re
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize
from nltk.corpus import stopwords
import nltk

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Set up NLTK data path (you may need to adjust this path)
NLTK_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'nltk_data')
nltk.data.path.append(NLTK_DATA_PATH)

def prepare_document(text, title="", author="", date=""):
    # Clean the text
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = text.lower()  # Lowercase

    try:
        # Use PunktSentenceTokenizer for sentence tokenization
        sentence_tokenizer = PunktSentenceTokenizer()
        sentences = sentence_tokenizer.tokenize(text)
        
        # Tokenize words in each sentence
        tokens = [word for sentence in sentences for word in word_tokenize(sentence)]
    except LookupError:
        print("Using simple tokenization method.")
        tokens = text.split()

    # Remove stop words
    try:
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stop_words]
    except LookupError:
        print("Stopwords not available. Skipping stopword removal.")

    # Rejoin the text
    cleaned_text = ' '.join(tokens)

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