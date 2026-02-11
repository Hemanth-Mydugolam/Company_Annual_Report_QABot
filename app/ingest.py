import os
import json
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# Folders
DATA_DIR = "pdfs"           # PDFs or pre-extracted vector text files
ARTIFACTS_DIR = "artifacts"
VECTOR_DB_DIR = os.path.join(ARTIFACTS_DIR, "chroma")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

documents = []
parsed_data = []

# Example: Either parse PDFs or load pre-extracted vector text
for file in os.listdir(DATA_DIR):
    if not file.endswith(".txt") and not file.endswith(".pdf"):
        continue

    company_name = file.split(".")[0]
    file_path = os.path.join(DATA_DIR, file)

    if file.endswith(".txt"):
        # Load pre-extracted text
        with open(file_path, "r") as f:
            text = f.read()
        parsed_data.append({"company": company_name, "text": text})
        documents.append(Document(page_content=text, metadata={"company": company_name}))

    else:
        # PDF parsing (optional if using PDFs)
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                parsed_data.append({
                    "company": company_name,
                    "page": page_num + 1,
                    "text": text,
                    "tables": page.extract_tables()
                })
                documents.append(Document(page_content=text, metadata={"company": company_name, "page": page_num + 1}))

# Save metadata for later
with open(os.path.join(ARTIFACTS_DIR, "parsed_data.json"), "w") as f:
    json.dump(parsed_data, f, indent=2)

# Create vector DB
embeddings = OpenAIEmbeddings()
Chroma.from_documents(
    documents,
    embeddings,
    persist_directory=VECTOR_DB_DIR
)

print("✅ Ingestion complete. Artifacts created in:", ARTIFACTS_DIR)
