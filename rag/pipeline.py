import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from typing import AsyncGenerator
import chromadb

from rag.chunker import chunk_legal_document
from rag.embedder import load_embeddings
from rag.retriever import HybridRetriever
from rag.reranker import Reranker
from rag.generator import Generator
import re

load_dotenv()


class RAGPipeline:
    """
    Orchestrates the full RAG pipeline.
    Each document gets its own ChromaDB collection.
    """

    def __init__(self):
        self.embeddings = load_embeddings()
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.reranker = Reranker()
        self.generator = Generator(api_key=os.getenv("GROQ_API_KEY"))

        # document registry — maps doc_id to metadata
        self.documents: dict = {}

        # active retriever — changes when user selects a document
        self.active_retriever: HybridRetriever | None = None
        self.active_doc_id: str | None = None

        # query cache per document
        self.cache: dict[str, str] = {}

        # reload existing collections on startup
        self._reload_existing_collections()

    def _reload_existing_collections(self):
        """Load existing document collections from ChromaDB on startup."""
        try:
            collections = self.chroma_client.list_collections()
            for col in collections:
                doc_id = col.name
                # get chunk count
                collection = self.chroma_client.get_collection(col.name)
                count = collection.count()
                self.documents[doc_id] = {
                    "doc_id": doc_id,
                    "filename": doc_id,
                    "chunks": count,
                    "document_type": "general"
                }
            if self.documents:
                # set first document as active by default
                first_doc = next(iter(self.documents))
                self._set_active_document(first_doc)
        except Exception as e:
            print(f"Warning: Could not reload collections: {e}")

    def _get_vectorstore(self, doc_id: str) -> Chroma:
        """Get ChromaDB vectorstore for a specific document."""
        return Chroma(
            client=self.chroma_client,
            collection_name=doc_id,
            embedding_function=self.embeddings
        )

    def _set_active_document(self, doc_id: str):
        """Set the active document for querying."""
        if doc_id not in self.documents:
            return

        vectorstore = self._get_vectorstore(doc_id)

        # get all chunks for BM25
        existing = vectorstore.get()
        chunks = existing["documents"] if existing and existing["documents"] else []

        self.active_retriever = HybridRetriever(
            vectorstore=vectorstore,
            chunks=chunks
        )
        self.active_doc_id = doc_id

    def _get_cache_key(self, query: str) -> str:
        return f"{self.active_doc_id}:{query.strip().lower()}"

    def ingest(self, text: str, filename: str, document_type: str = "general") -> dict:
        """
        Chunks, embeds, and stores a document in its own collection.
        Returns document info.
        """
        # create clean doc_id from filename
        doc_id = re.sub(r'[^a-z0-9]', '_', filename.lower().replace('.pdf', ''))[:50]

        # chunk the document
        new_chunks = chunk_legal_document(text, document_type, self.embeddings)

        # store in its own ChromaDB collection
        vectorstore = self._get_vectorstore(doc_id)
        vectorstore.add_texts(new_chunks)

        # register document
        self.documents[doc_id] = {
            "doc_id": doc_id,
            "filename": filename,
            "chunks": len(new_chunks),
            "document_type": document_type
        }

        # set as active document
        self._set_active_document(doc_id)

        return self.documents[doc_id]

    def select_document(self, doc_id: str) -> bool:
        """Switch active document for querying."""
        if doc_id not in self.documents:
            return False
        self._set_active_document(doc_id)
        return True

    def get_documents(self) -> list:
        """Return list of all ingested documents."""
        return list(self.documents.values())

    def query(self, question: str, k: int = 20, top_k: int = 5) -> dict:
        """Runs full RAG pipeline for a question against active document."""

        cache_key = self._get_cache_key(question)
        if cache_key in self.cache:
            return {
                "answer": self.cache[cache_key],
                "sources": [],
                "cached": True,
                "active_doc": self.active_doc_id
            }

        if self.active_retriever is None:
            return {
                "answer": "No documents ingested yet. Please upload documents first.",
                "sources": [],
                "cached": False,
                "active_doc": None
            }

        # stage 1 — hybrid retrieval
        retrieved_chunks = self.active_retriever.retrieve(question, k=k)

        # stage 2 — reranking
        reranked_chunks = self.reranker.rerank(question, retrieved_chunks, top_k=top_k)

        # stage 3 — generation
        answer = self.generator.generate(question, reranked_chunks)

        # cache with size limit
        if len(self.cache) > 1000:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[cache_key] = answer

        sources = [
            {"content": chunk[:200], "score": score}
            for chunk, score in reranked_chunks
        ]

        return {
            "answer": answer,
            "sources": sources,
            "cached": False,
            "active_doc": self.active_doc_id
        }

    async def query_stream(
        self,
        question: str,
        k: int = 20,
        top_k: int = 5
    ) -> AsyncGenerator[str, None]:
        """Streaming version of query."""
        if self.active_retriever is None:
            yield "No documents ingested yet. Please upload documents first."
            return

        retrieved_chunks = self.active_retriever.retrieve(question, k=k)
        reranked_chunks = self.reranker.rerank(question, retrieved_chunks, top_k=top_k)

        async for token in self.generator.generate_stream(question, reranked_chunks):
            yield token