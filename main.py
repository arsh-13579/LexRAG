from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import fitz
from fastapi.responses import FileResponse

from rag.pipeline import RAGPipeline

limiter = Limiter(key_func=get_remote_address)

# ─────────────────────────────────────────
# 1. LIFESPAN — load pipeline once
# ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = RAGPipeline()
    yield

app = FastAPI(
    title="LexRAG",
    description="Production-grade Legal Document RAG API",
    version="1.0.0",
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ─────────────────────────────────────────
# 2. PYDANTIC MODELS
# ─────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    k: int = 10
    top_k: int = 3

class Source(BaseModel):
    content: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    cached: bool

class IngestResponse(BaseModel):
    filename: str
    document_type: str
    chunks_stored: int
    message: str


# ─────────────────────────────────────────
# 3. ROUTES
# ─────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "llama3-8b-8192",
        "embeddings": "all-MiniLM-L6-v2",
        "reranker": "ms-marco-MiniLM-L-6-v2"
    }

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

@app.post("/ingest", response_model=IngestResponse)
@limiter.limit("5/minute")
async def ingest(request: Request, file: UploadFile = File(...), document_type: str = "general"):
    # validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted"
        )

    # read file bytes
    contents = await file.read()

    if len(contents) > 10 * 1024 * 1024:  #10MB limit
        raise HTTPException(
            status_code=400,
            detail="File too large, Maximum size is 10MB"
        )

    # extract text from PDF
    try:
        pdf = fitz.open(stream=contents, filetype="pdf")
        text = ""
        for page in pdf:
            text += page.get_text()
        pdf.close()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse PDF: {str(e)}"
        )

    # validate extracted text
    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="Could not extract text from this PDF. It may be scanned or image-based."
        )

    # run ingestion pipeline
    try:
        chunks_stored = app.state.pipeline.ingest(text, document_type)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )

    return IngestResponse(
        filename=file.filename,
        document_type=document_type,
        chunks_stored=chunks_stored,
        message=f"{file.filename} ingested successfully as {document_type} document"
    )


@app.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute")
async def query(request: Request, body: QueryRequest):
    # validate question
    if not body.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    if len(body.question) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Question too long. Maximum 1000 characters."
        )

    # run query pipeline
    try:
        result = app.state.pipeline.query(
            question=body.question,
            k=body.k,
            top_k=body.top_k
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )

    # build sources
    sources = [
        Source(content=s["content"], score=s["score"])
        for s in result["sources"]
    ]

    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        cached=result["cached"]
    )


@app.post("/query-stream")
@limiter.limit("10/minute")
async def query_stream(request: Request, body: QueryRequest):
    # validate question
    if not body.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    if len(body.question) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Question too long. Maximum 1000 characters."
        )

    # streaming generator
    async def token_generator():
        try:
            async for token in app.state.pipeline.query_stream(
                question=body.question,
                k=body.k,
                top_k=body.top_k
            ):
                yield token
        except Exception as e:
            yield f"\n[Error]: {str(e)}"

    return StreamingResponse(
        token_generator(),
        media_type="text/plain"
    )