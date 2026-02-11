from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import json
from config import VECTOR_DB_DIR, LLM_MODEL, EMBEDDING_MODEL

# LLM
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

# Vector DB
vectordb = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL)
)

# Parsed metadata
with open("artifacts/parsed_data.json") as f:
    PARSED_DATA = json.load(f)

def answer_question(question, company):
    docs = vectordb.similarity_search(
        question, k=4, filter={"company": company}
    )

    text_context = "\n".join([d.page_content for d in docs])
    tables = PARSED_DATA[company]["tables"]
    plots = PARSED_DATA[company]["plots"]

    prompt = f"""
You are an enterprise AI assistant.

TEXT CONTEXT:
{text_context}

TABLE DATA:
{tables}

PLOT INSIGHTS:
{plots}

Answer the question ONLY from the above data.
If not found, say "Not available in documents."

Question: {question}
"""

    response = llm.invoke(prompt)

    return response.content
