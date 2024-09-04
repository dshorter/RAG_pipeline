import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")

# FAISS Configuration
FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "./data/faiss_index")

# Embedding Generator Configuration
EMBEDDING_GENERATOR_TYPE = os.getenv("EMBEDDING_GENERATOR_TYPE")


EMBEDDING_GENERATOR_CONFIG = {
    "azure_openai": {
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
        "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT")
    },
    "huggingface": {
        "model_name": os.getenv("HUGGINGFACE_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    }
}

# RAG Pipeline Configuration
RAG_CONFIG = {
    "embedding_generator_type": EMBEDDING_GENERATOR_TYPE,
    "embedding_generator_config": EMBEDDING_GENERATOR_CONFIG[EMBEDDING_GENERATOR_TYPE],
    "raw_docs_dir": os.getenv("RAW_DOCS_DIR", "./data/raw"),
    "processed_docs_dir": os.getenv("PROCESSED_DOCS_DIR", "./data/processed"),
    "chunk_size": int(os.getenv("CHUNK_SIZE", "500")),
    "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "50"))
}

# Validate required environment variables
required_vars = ["AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_ENDPOINT", "AZURE_CLIENT_SECRET", "AZURE_TENANT_ID" ]
for var in required_vars:
    if not os.getenv(var):
        raise EnvironmentError(f"{var} is not set in the environment variables.")

# Validate embedding generator configuration
if EMBEDDING_GENERATOR_TYPE not in EMBEDDING_GENERATOR_CONFIG:
    raise ValueError(f"Unknown embedding generator type: {EMBEDDING_GENERATOR_TYPE}")

if EMBEDDING_GENERATOR_TYPE == "azure_openai":
    required_azure_vars = ["azure_endpoint", "api_version", "deployment"]
    for var in required_azure_vars:
        if not EMBEDDING_GENERATOR_CONFIG["azure_openai"][var]:
            raise EnvironmentError(f"{var} is not set for Azure OpenAI configuration.")

# Add any other configuration variables here