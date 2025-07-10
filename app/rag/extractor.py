import os
import fitz  # PyMuPDF
import pandas as pd
import docx

def extract_text_from_folder(folder_path):
    texts = []

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        if file.endswith(".pdf"):
            with fitz.open(file_path) as doc:
                texts.append("\n".join([page.get_text() for page in doc]))  # type: ignore

        elif file.endswith(".docx"):
            docx_doc = docx.Document(file_path)
            texts.append("\n".join([p.text for p in docx_doc.paragraphs]))

        elif file.endswith(".xlsx"):
            xls = pd.read_excel(file_path, sheet_name=None)
            for sheet in xls.values():
                texts.append(sheet.to_string())

    return texts

__all__ = ["extract_text_from_folder"]
