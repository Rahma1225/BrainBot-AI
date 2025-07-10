from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
import httpx
from app.config import OLLAMA_URL, DEFAULT_MODEL

app = FastAPI()

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.on_event("startup")
async def preload_ollama_model():
    print(f"🔥 Preloading model {DEFAULT_MODEL} from Ollama at {OLLAMA_URL}...")
    try:
        # Check if Ollama is running and get available models
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags")
            resp.raise_for_status()
            tags = resp.json().get("models", [])
            model_names = [m.get("name") for m in tags]
            if DEFAULT_MODEL not in model_names:
                # Pull the model if not present
                print(f"⬇️ Pulling model {DEFAULT_MODEL} from Ollama...")
                pull_resp = await client.post(f"{OLLAMA_URL}/api/pull", json={"name": DEFAULT_MODEL})
                pull_resp.raise_for_status()
                print(f"✅ Model {DEFAULT_MODEL} pulled successfully.")
            else:
                print(f"✅ Model {DEFAULT_MODEL} is already available in Ollama.")
    except Exception as e:
        print(f"❌ Failed to preload model: {e}")
