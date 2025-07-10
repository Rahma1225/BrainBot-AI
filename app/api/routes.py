from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from app.rag.query_engine import answer_with_rag
from app.rag.extractor import extract_text_from_folder
from app.services.embedding import embed_texts
from app.services.storage import store_chunks
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

        # Save uploaded file to a unique subfolder
        temp_dir = os.path.join("temp_uploads", str(uuid.uuid4()))
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Extract text from file
        texts = extract_text_from_folder(temp_dir)
        for text in texts:
            if not text.strip():
                continue
            chunks = [text[j:j+512] for j in range(0, len(text), 512)]
            embeddings = embed_texts(chunks)
            store_chunks(chunks, embeddings, metadata={"filename": file.filename})

        # Cleanup
        shutil.rmtree(temp_dir)

        return {"status": "success", "chunks_indexed": len(texts)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
