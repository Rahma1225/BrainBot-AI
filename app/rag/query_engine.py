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
    "Tu es un assistant intelligent spécialisé dans la lecture et l’analyse de documents techniques et fonctionnels (guides, procédures, manuels d’implémentation, etc.).\n"
    "Tu dois répondre à la question ci-dessous UNIQUEMENT en te basant sur le CONTEXTE fourni.\n"
    "Ta réponse doit être rédigée en français, claire, structurée et toujours fidèle au contenu du contexte.\n\n"
    "📌 Règles à respecter :\n"
    "1. Si la question contient un **code, écran, identifiant ou référence (ex : IN201000)** → indique à quoi cela correspond, avec citation ou extrait du contexte si possible.\n"
    "2. Si la question concerne une **liste d’étapes, une procédure ou un plan structuré** → présente la réponse sous forme de **liste numérotée**, avec des **puces pour les sous-éléments**.\n"
    "3. Si le contexte suit une **structure ou un sommaire clair** (ex : Préparation, Configuration, Initialisation...) → respecte **l’ordre exact** et **les titres** tels qu’ils apparaissent dans le texte.\n"
    "4. Si **aucune réponse claire ne peut être déduite**, écris simplement : \"📌 Le contexte ne fournit pas cette information.\"\n"
    "5. Ne reformule pas les titres du contexte. Utilise-les tels quels si présents.\n"
    "6. Sois concis mais complet : ta réponse doit refléter **tout ce qui est pertinent dans le contexte**.\n\n"
    f"📘 CONTEXTE :\n{context}\n\n"
    f"❓ QUESTION : {question}"
)



    # 5. Envoyer le prompt au modèle
    answer = ask_ollama(prompt)

    # 6. Vérification facultative de la pertinence
    if any(chunk in answer for chunk in filtered_chunks):
        return answer
    else:
        return answer + "\n\n(Note : La réponse pourrait ne pas être entièrement appuyée par le contexte fourni.)"
