"""Phase 3/4/4.5/7 -- the front desk. A thin async HTTP layer over app/model.py,
app/agent.py, and app/rag.py, plus a landing page for non-technical visitors.
"""

import asyncio
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.agent import review_document
from app.model import classify_clause
from app.rag import answer_question, index_document

app = FastAPI(title="Compliance Copilot", version="0.1.0")

INDEX_HTML = (Path(__file__).resolve().parent / "static" / "index.html").read_text(encoding="utf-8")


class ClauseRequest(BaseModel):
    clause: str


class ClassifyResponse(BaseModel):
    label: str
    raw: str


class ReviewRequest(BaseModel):
    document: str


class ClauseResult(BaseModel):
    text: str
    label: str


class ReviewResponse(BaseModel):
    num_clauses: int
    clauses: list[ClauseResult]
    present_types: list[str]
    missing_required: list[str]
    summary: str


class IndexRequest(BaseModel):
    document: str


class IndexResponse(BaseModel):
    status: str
    num_chunks: int


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/", response_class=HTMLResponse)
async def index():
    return INDEX_HTML


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClauseRequest):
    # classify_clause() is CPU-bound and blocking; run it in a worker thread so
    # the event loop stays free to serve other requests (e.g. /health) concurrently.
    return await asyncio.to_thread(classify_clause, request.clause)


@app.post("/review", response_model=ReviewResponse)
async def review(request: ReviewRequest):
    return await asyncio.to_thread(review_document, request.document)


@app.post("/rag/index", response_model=IndexResponse)
async def rag_index(request: IndexRequest):
    num_chunks = await asyncio.to_thread(index_document, request.document)
    return {"status": "indexed", "num_chunks": num_chunks}


@app.post("/rag/ask", response_model=AskResponse)
async def rag_ask(request: AskRequest):
    return await asyncio.to_thread(answer_question, request.question)
