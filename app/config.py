import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3:8b-instruct-q4_k_m")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
DB_NAME = os.getenv("DB_NAME", "brainbot")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents") 