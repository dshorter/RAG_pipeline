# src/client/preprocessing/preprocess_query.py

import re
from typing import Dict

async def preprocess_query(query: str, config: Dict = {}) -> str:
    """
    Asynchronously preprocess the user's query.
    
    Args:
    query (str): The original user query.
    config (Dict): Configuration dictionary for preprocessing options. Default is an empty dict.
    
    Returns:
    str: The preprocessed query.
    """
    # Convert to lowercase if specified in config
    if config.get('lowercase_query', True):
        query = query.lower()
    
    # Remove special characters if specified in config
    if config.get('remove_special_chars', True):
        query = re.sub(r'[^\w\s]', '', query)
    
    # Remove extra whitespace
    query = ' '.join(query.split())
    
    return query