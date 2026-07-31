# LexRAG — Intelligent Legal Document Assistant

A production-grade Retrieval Augmented Generation (RAG) system built for legal document analysis. Upload court judgments, contracts, or compliance documents and ask precise legal questions — powered by hybrid search, cross-encoder reranking, and streaming LLM generation.

## Live Demo

Upload a PDF → Ask a legal question → Get a grounded, cited answer in seconds

---

## Architecture

```
PDF Upload
    ↓
Smart Legal Chunking (structure-aware for contracts, recursive for judgments)
    ↓
HuggingFace Embeddings → ChromaDB (persistent vector storage)
    ↓
Query comes in
    ↓
Hybrid Search: Vector Search (semantic) + BM25 (keyword) → RRF Fusion
    ↓
Cross-Encoder Re-ranking (top 10 → top 3)
    ↓
Grounded LLM Prompt → Groq (LLaMA 3.1)
    ↓
Streamed Answer + Sources with Relevance Scores
```

---

## Why These Design Choices

### Hybrid Search (Vector + BM25)
Legal documents contain exact citations, section numbers, and case references (e.g. "Section 4.2(b)", "Article 17 GDPR"). Pure vector search struggles with these because embedding models don't reliably capture rare exact tokens. BM25 keyword search handles this perfectly. Combining both via Reciprocal Rank Fusion (RRF) gives the best of both worlds — semantic understanding AND exact term matching.

### Two-Stage Retrieval + Cross-Encoder Reranking
Bi-encoder embedding models are fast but approximate — they compress meaning into fixed-size vectors, losing nuance. A cross-encoder reads the query and each chunk together simultaneously, producing far more precise relevance scores. We use a two-stage approach: retrieve 10 candidates fast (bi-encoder), then rerank to top 3 precisely (cross-encoder). This keeps latency low while maximizing answer quality.

### Structure-Aware Chunking for Contracts
Contracts have explicit section/clause structure. Using MarkdownHeaderTextSplitter before RecursiveCharacterTextSplitter respects this structure — each chunk stays within its section boundary, preventing context from bleeding across unrelated clauses.

### Temperature = 0.0
Legal answers must be precise and deterministic. Setting temperature to 0 ensures the LLM always picks the most probable token — no creative variation that could introduce inaccuracies in legal context.

### Grounded Prompt Design
The prompt explicitly instructs the LLM to answer using ONLY the retrieved context and say "I cannot find this information" when the answer isn't present. This directly improves faithfulness and prevents hallucination — critical for legal use cases.

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| Backend | FastAPI | Async, fast, auto-docs |
| Vector DB | ChromaDB | Persistent, local, no cost |
| Embeddings | all-MiniLM-L6-v2 | Fast, good quality, free |
| Keyword Search | BM25 (rank-bm25) | Exact term matching for legal citations |
| Re-ranker | ms-marco-MiniLM-L-6-v2 | Trained on search relevance pairs |
| LLM | Groq / LLaMA 3.1 | Fastest inference available, free tier |
| PDF Parsing | PyMuPDF | Fast, accurate text extraction |
| Evaluation | RAGAS | Standardized RAG metrics |

---

## Evaluation Results (RAGAS)

Evaluated on 5 legal questions across an NDA document:

| Metric | Score | What it measures |
|---|---|---|
| Faithfulness | 1.0000 | LLM answers grounded in context (no hallucination) |
| Answer Relevancy | 0.9846 | Answers address the question asked |
| Context Recall | 1.0000 | All necessary chunks retrieved |

> Note: Context Precision timed out due to Groq free-tier rate limits during evaluation. All other metrics measured cleanly.

---

## API Endpoints

**GET /health** — Returns server status and loaded model names.

**POST /ingest** — Upload a PDF document for ingestion.
- `file`: PDF file (multipart/form-data)
- `document_type`: `general` (default) or `contract`

**POST /query** — Ask a question and get a complete answer with sources.

```json
{
  "question": "What are the penalties for breach?",
  "k": 10,
  "top_k": 3
}
```

**POST /query-stream** — Same as /query but streams tokens word by word.

---

## Project Structure

```
lexrag/
│
├── main.py                  ← FastAPI app (routes, lifespan)
├── index.html               ← Frontend (single file, served by FastAPI)
├── rag/
│   ├── chunker.py           ← Smart legal document chunking
│   ├── embedder.py          ← HuggingFace embedding model loader
│   ├── retriever.py         ← Hybrid search (vector + BM25 + RRF)
│   ├── reranker.py          ← Cross-encoder reranking
│   ├── generator.py         ← LLM prompt + streaming generation
│   └── pipeline.py          ← Orchestrates full RAG pipeline
├── evaluation/
│   ├── ragas_eval.py        ← RAGAS evaluation suite
│   └── results.json         ← Evaluation scores
├── data/
│   └── sample_docs/         ← Sample legal PDFs
└── requirements.txt
```

---

## Local Setup

```bash
# clone the repo
git clone https://github.com/yourusername/lexrag.git
cd lexrag

# create virtual environment (Python 3.11 required)
py -3.11 -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# install dependencies
pip install -r requirements.txt

# add your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# run the server
uvicorn main:app --reload
```

Open `http://localhost:8000` in your browser.

---

## Production Upgrade Path

| Current | Production Upgrade | Why |
|---|---|---|
| ChromaDB (local) | Pinecone / Weaviate | Distributed, scalable vector search |
| BM25 (in-memory) | Elasticsearch | Scalable keyword search, persistent |
| In-memory cache | Redis | Persistent cache across restarts |
| Single FastAPI instance | Load balancer + multiple instances | Handle concurrent users |

---

## Three-Way Comparison: LexRAG vs ChatGPT vs Gemini

Tested on Sukanya Shantha v. Union of India [2024] INSC 753 — a landmark Supreme Court judgment on caste-based discrimination in prisons.

| | LexRAG | ChatGPT | Gemini |
|---|---|---|---|
| Factual accuracy | ✅ Correct | ✅ Correct | ✅ Correct |
| Response time | ✅ ~2.8s | ✅ ~2.9s | ❌ ~7s |
| Hallucination on unknown questions | ✅ Refused to answer | ❌ Gave confident wrong answer | ❌ Gave confident wrong answer |
| Sources cited with relevance scores | ✅ Yes | ❌ No | ❌ No |
| Works on private documents | ✅ Yes | ⚠️ Uploads to OpenAI servers | ⚠️ Uploads to Google servers |

### Key Finding
For questions about events not mentioned in the document (post-judgment implementation, 2025 developments), ChatGPT and Gemini generated detailed confident answers from their training data — impossible to verify what came from the document vs what was hallucinated. LexRAG correctly responded "I cannot find this information in the provided documents" — because it only answers from retrieved context.

## What I Learned Building This

- **Hybrid search matters for legal RAG** — pure vector search missed exact citations until BM25 was added
- **Re-ranking dramatically improves precision** — cross-encoder correctly re-ordered results that bi-encoder ranked wrong
- **Prompt design directly impacts faithfulness** — adding "ONLY use context" and "say I don't know" brought faithfulness to 1.0
- **Chunking strategy depends on document type** — structure-aware chunking for contracts, recursive for free-form judgments

---

Built by Felix | B.Tech Student | Aspiring RAG Developer