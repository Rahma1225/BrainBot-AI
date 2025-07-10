import os
import uuid
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from chromadb import PersistentClient

# ChromaDB n'accepte que les types simples dans les métadonnées
MetadataValue = Union[str, int, float, bool, None]
Metadata = Dict[str, MetadataValue]

# === 1. Chunking avec chevauchement ===
def chunk_text(text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
    """Divise un texte en morceaux avec chevauchement."""
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i + max_chars]
        chunks.append(chunk.strip())
        i += max_chars - overlap
    return chunks

# === 2. Recherche vectorielle dans ChromaDB ===
def retrieve_similar_chunks(query_embedding: List[float], top_n: int = 5) -> List[Tuple[str, float]]:
    """Recherche les chunks les plus similaires dans ChromaDB."""
    client = PersistentClient(path="app/chroma_store")
    collection = client.get_or_create_collection(name="documents")

    results = collection.query(query_embeddings=[query_embedding], n_results=top_n)

    documents = results.get("documents", [[]])
    distances = results.get("distances", [[]])

    if documents and distances and documents[0] and distances[0]:
        return list(zip(documents[0], distances[0]))

    return []

# === 3. Insertion dynamique dans ChromaDB ===
def store_chunks(
    chunks: List[str],
    embeddings: List[List[float]],
    metadata: Optional[Metadata] = None
):
    """
    Enregistre les chunks et leurs embeddings dans ChromaDB.

    Args:
        chunks: morceaux de texte
        embeddings: vecteurs d'embedding correspondants
        metadata: dictionnaire avec des métadonnées simples (str, int, float, bool, None)
    """
    if not chunks or len(embeddings) == 0:
        return

    if len(chunks) != len(embeddings):
        raise ValueError("Mismatch between number of chunks and embeddings.")

    client = PersistentClient(path="app/chroma_store")
    collection = client.get_or_create_collection(name="documents")

    ids = [str(uuid.uuid4()) for _ in chunks]

    # Réplique les mêmes métadonnées simples pour chaque chunk
    if metadata:
        metadatas: List[Metadata] = [
            {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool)) or v is None}
            for _ in chunks
        ]
    else:
        metadatas = [{} for _ in chunks]

    embeddings_array = np.array(embeddings, dtype=np.float32)

    collection.add(
        documents=chunks,
        embeddings=embeddings_array,
        metadatas=metadatas,  # type: ignore
        ids=ids
    )
