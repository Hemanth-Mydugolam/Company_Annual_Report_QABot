from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import json
from config import VECTOR_DB_DIR, EMBEDDING_MODEL

def build_text_index():
    with open("parsed_data.json") as f:
        data = json.load(f)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    docs = []

    for company, content in data.items():
        for page in content["text"]:
            chunks = splitter.split_text(page["text"])
            for c in chunks:
                docs.append({
                    "page_content": c,
                    "metadata": {
                        "company": company,
                        "page": page["page"]
                    }
                })

    vectordb = Chroma.from_documents(
        documents=[type("D", (), d) for d in docs],
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )

    vectordb.persist()
    print("✅ Text embeddings created")
