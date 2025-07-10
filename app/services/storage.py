import os
from pymongo import MongoClient
import numpy as np
from app.config import MONGO_URI, DB_NAME, COLLECTION_NAME

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

def store_chunks(chunks, embeddings, metadata=None):
    docs = []
    for chunk, embedding in zip(chunks, embeddings):
        doc = {
            "content": chunk,
            "embedding": embedding.tolist(),  # Convert numpy array to list
            "metadata": metadata or {}
        }
        docs.append(doc)
    if docs:
        collection.insert_many(docs)

def retrieve_similar_chunks(query_embedding, top_n=5):
    """
    Retrieve the top-N most similar chunks from MongoDB using cosine similarity.
    """
    # Fetch all embeddings and contents
    docs = list(collection.find({}, {"content": 1, "embedding": 1, "_id": 0}))
    # Filter out docs without 'embedding'
    docs = [doc for doc in docs if "embedding" in doc]
    if not docs:
        return []
    embeddings = np.array([doc["embedding"] for doc in docs])
    # Normalize for cosine similarity
    norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm_query = query_embedding / np.linalg.norm(query_embedding)
    similarities = np.dot(norm_embeddings, norm_query)
    top_indices = similarities.argsort()[-top_n:][::-1]
    return [(docs[i]["content"], similarities[i]) for i in top_indices]
