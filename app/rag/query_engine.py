from typing import List, Union, Tuple
from app.services.embedding import embed_texts
from app.services.storage import retrieve_similar_chunks
from app.services.ollama_service import ask_ollama


def answer_with_rag(question: str, similarity_threshold: float = 0.7, top_n: int = 5) -> str:
    # 1. Générer l'embedding de la question
    query_embedding = embed_texts([question])[0]

    # 2. Récupérer les chunks similaires depuis ChromaDB
    retrieved = retrieve_similar_chunks(query_embedding, top_n=top_n)

    # 3. Filtrer les chunks si des scores de similarité sont présents
    if retrieved and isinstance(retrieved[0], tuple):
        filtered_chunks: List[str] = [
            chunk for chunk, score in retrieved if score >= similarity_threshold
        ]
    else:
        filtered_chunks = [chunk for chunk in retrieved]  # type: ignore

    if not filtered_chunks:
        return "Aucune information pertinente trouvée pour répondre à la question."

    # 4. Construire le prompt en français
    context = "\n".join(filtered_chunks)
    prompt = (
        "Tu es un assistant intelligent. Réponds à la question suivante en te basant UNIQUEMENT sur le contexte ci-dessous :\n\n"
        f"Contexte :\n{context}\n\n"
        f"Question : {question}"
    )

    # 5. Envoyer le prompt au modèle
    answer = ask_ollama(prompt)

    # 6. Vérification facultative de la pertinence
    if any(chunk in answer for chunk in filtered_chunks):
        return answer
    else:
        return answer + "\n\n(Note : La réponse pourrait ne pas être entièrement appuyée par le contexte fourni.)"
