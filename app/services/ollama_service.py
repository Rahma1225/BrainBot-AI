import httpx
import json
from app.config import OLLAMA_URL, DEFAULT_MODEL

def get_relevant_context(prompt: str) -> str:
    # Placeholder: Add MongoDB retrieval logic here if needed
    return ""

def ask_ollama(prompt: str) -> str:
    context = get_relevant_context(prompt)
    if context:
        full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}"
    else:
        full_prompt = prompt

    payload = {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": full_prompt}],
        "stream": False
    }

    try:
        print(f"🔄 Sending to Ollama at {OLLAMA_URL}/api/chat with prompt: {full_prompt}")
        response = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=150)
        response.raise_for_status()
        data = response.json()
        print("✅ Ollama parsed response:", data)

        # Try both possible response formats
        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]
        if "response" in data:
            return data["response"]

        return "⚠️ Unexpected Ollama response format."

    except json.JSONDecodeError as e:
        print("❌ JSON decode error:", e)
        print("📦 Raw response:", response.text)
        return "❌ Error: Could not decode Ollama response."

    except Exception as e:
        print("❌ Unexpected error:", e)
        return f"❌ Error: {str(e)}"
