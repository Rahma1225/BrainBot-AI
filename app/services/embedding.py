from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL_NAME

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def embed_texts(texts):
    """
    Embed a list of texts and return a list of vectors.
    """
    return embedding_model.encode(texts, convert_to_numpy=True) 