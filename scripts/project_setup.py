import os
import subprocess

def create_directory_structure():
    directories = [
        "src",
        "src/retrieval",
        "src/optimizations",
        "tests",
        "tests/test_optimizations",
        "scripts",
        "data/raw",
        "data/processed"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "__init__.py"), "w") as f:
            pass  # Create empty __init__.py files

def create_initial_files():
    files = [
        "src/config.py",
        "src/knowledge_base.py",
        "src/embedding.py",
        "src/generation.py",
        "src/rag_system.py",
        "src/retrieval/base.py",
        "src/retrieval/faiss_search.py",
        "requirements.txt",
        "setup.py",
        "README.md"
    ]
    for file in files:
        with open(file, "w") as f:
            pass  # Create empty files

def create_env_files():
    env_example_content = """
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://your-resource-name.openai.azure.com/
OPENAI_API_VERSION=2023-05-15
FAISS_INDEX_DIR=./data/faiss_index
"""
    with open(".env.example", "w") as f:
        f.write(env_example_content.strip())
    
    print("Created .env.example file. Please create a .env file with your actual values.")

def update_gitignore():
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*.so

# Environments
.env
.venv
env/
venv/

# IDEs
.vscode/
.idea/

# Project specific
data/processed/
"""
    with open(".gitignore", "w") as f:
        f.write(gitignore_content.strip())

def main():
    create_directory_structure()
    create_initial_files()
    create_env_files()
    update_gitignore()
    
    # Initialize git repository
    subprocess.run(["git", "init"])
    
    print("Project structure created successfully!")
    print("Next steps:")
    print("1. Set up a virtual environment: python -m venv venv")
    print("2. Activate the virtual environment")
    print("3. Install dependencies: pip install -r requirements.txt")
    print("4. Create a .env file with your actual API keys and configuration")
    print("5. Start implementing your RAG system!")

if __name__ == "__main__":
    main()