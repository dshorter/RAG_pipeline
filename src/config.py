import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI Configuratio
# n
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")

# FAISS Configuration
FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "./data/faiss_index")

# Add any other configuration variables here

# Validate required environment variables
required_vars = ["OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_API_VERSION"]
for var in required_vars:
    if not globals()[var]:
        raise EnvironmentError(f"{var} is not set in the environment variables.")