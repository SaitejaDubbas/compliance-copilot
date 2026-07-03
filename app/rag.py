"""Phase 4.5 -- a RAG chatbot over a single indexed contract. Retrieval runs
locally (free embeddings + FAISS); only the final answer generation calls out
to Groq.
"""

import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 750
CHUNK_OVERLAP = 100
TOP_K = 4

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a contract question-answering assistant. Answer the "
            "question using ONLY the provided contract context below. Do not "
            "use outside knowledge or assumptions. If the answer is not "
            'contained in the context, reply exactly: "I don\'t know based on '
            'this contract."',
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

print(f"Loading embedding model {EMBEDDING_MODEL} ...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

_vectorstore = None
_llm = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        if not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file (see .env.example)."
            )
        _llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    return _llm


def index_document(text: str) -> int:
    """Split, embed, and index a document. Replaces any previously indexed one."""
    global _vectorstore
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    _vectorstore = FAISS.from_documents(docs, embeddings)
    return len(chunks)


def answer_question(question: str) -> dict:
    if _vectorstore is None:
        return {
            "answer": "No document has been indexed yet. Call /rag/index first.",
            "sources": [],
        }

    retrieved = _vectorstore.similarity_search(question, k=TOP_K)
    if not retrieved:
        return {"answer": "I don't know based on this contract.", "sources": []}

    context = "\n\n".join(doc.page_content for doc in retrieved)
    chain = RAG_PROMPT | _get_llm() | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return {"answer": answer.strip(), "sources": [doc.page_content for doc in retrieved]}
