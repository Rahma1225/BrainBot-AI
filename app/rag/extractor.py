import os
import fitz  # PyMuPDF
import docx
import pandas as pd
from typing import List, Dict
from transformers import AutoTokenizer
from app.services.text_splitter import split_into_sections 


# Load a sentence tokenizer (you can change the model as needed)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def read_file_text(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        with fitz.open(file_path) as doc:
            return "\n".join([page.get_text() for page in doc])  # type: ignore
    elif file_path.endswith(".docx"):
        docx_doc = docx.Document(file_path)
        return "\n".join([p.text for p in docx_doc.paragraphs])
    elif file_path.endswith(".xlsx"):
        xls = pd.read_excel(file_path, sheet_name=None)
        return "\n".join([sheet.to_string() for sheet in xls.values()])
    return ""

def chunk_text(text: str, max_tokens: int = 200, overlap: int = 30) -> List[str]:
    input_ids = tokenizer.encode(text, truncation=False)
    chunks = []

    i = 0
    while i < len(input_ids):
        chunk_ids = input_ids[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        i += max_tokens - overlap

    return chunks

def extract_chunks_from_folder(folder_path: str) -> List[Dict]:
    """
    Extract and chunk text from supported files.
    Returns a list of dicts: { 'text': chunk, 'metadata': { 'source': filename } }
    """
    data = []

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if not file_path.lower().endswith((".pdf", ".docx", ".xlsx")):
            continue

        text = read_file_text(file_path)
        if not text.strip():
            continue

        chunks = split_into_sections(text)
        for chunk in chunks:
            data.append({
                "text": chunk,
                "metadata": {
                    "source": file
                }
            })

    return data

__all__ = ["extract_chunks_from_folder"]
