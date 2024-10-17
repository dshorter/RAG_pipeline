import os
from src.singleton_config import ConfigSingleton

config = ConfigSingleton()

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_data_dir():
    return os.path.join(get_project_root(), config.get('data_dir', 'data'))

def get_db_path():
    return os.path.join(get_data_dir(), config.get('db_name', 'metadata.db'))

def get_faiss_path():
    return os.path.join(get_data_dir(), config.get('faiss_index_name', 'faiss_index.bin'))

def get_raw_docs_dir():
    return os.path.join(get_data_dir(), config.get('raw_docs_dir', 'raw'))

def get_processed_docs_dir():
    return os.path.join(get_data_dir(), config.get('processed_docs_dir', 'processed'))

# Add more path-related functions as needed