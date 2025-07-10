from app.services.embedding import embed_texts
from app.services.storage import retrieve_similar_chunks
from app.services.ollama_service import ask_ollama


def answer_with_rag(question, similarity_threshold=0.7, top_n=5):
    # 1. Embed the question
    query_embedding = embed_texts([question])[0]
    # 2. Retrieve similar chunks
    chunks = retrieve_similar_chunks(query_embedding, top_n=top_n)
    # 3. (Optional) Filter by similarity threshold if retrieve_similar_chunks returns scores
    # For now, assume chunks are already filtered or all relevant
    context = "\n".join(chunk for chunk, _ in chunks)
    # 4. Generate answer with Ollama (Llama3)
    prompt = f"""Answer the question based on the following context:\n\n{context}\n\nQuestion: {question}"""
    answer = ask_ollama(prompt)
    # 5. (Optional) Relevance check: simple version
    if not any(chunk in answer for chunk, _ in chunks):
        answer += "\n\n(Note: The answer may not be fully supported by the provided context.)"
    return answer
