from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from app.services.embedding import embed_texts
from app.services.storage import store_chunks
from app.rag.extractor import extract_chunks_from_folder
from app.rag.query_engine import answer_with_rag  # 🔧 Make sure this is implemented
import os
import shutil
import uuid

router = APIRouter()

class PromptRequest(BaseModel):
    prompt: str

@router.post("/chat", response_model=str)
def chat(request: PromptRequest):
    try:
        response = answer_with_rag(request.prompt)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_doc(file: UploadFile = File(...)):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

        temp_dir = os.path.join("temp_uploads", str(uuid.uuid4()))
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        chunks_data = extract_chunks_from_folder(temp_dir)

        for item in chunks_data:
            text = item["text"]
            if not text.strip():
                continue
            embeddings = embed_texts([text])
            store_chunks([text], embeddings, metadata=item["metadata"])

        shutil.rmtree(temp_dir)

        return {"status": "success", "chunks_indexed": len(chunks_data)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
