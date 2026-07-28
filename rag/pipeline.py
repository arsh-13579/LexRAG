import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from typing import AsyncGenerator

from rag.chunker import chunk_legal_document
from rag.embedder import load_embeddings
from rag.retriever import HybridRetriever
from rag.reranker import Reranker
from rag.generator import Generator

load_dotenv()


class RAGPipeline:
    """
    Orchestrates the full RAG pipeline:
    ingest → chunk → embed → store
    query  → retrieve → rerank → generate
    """

    def __init__(self):
        self.embeddings = load_embeddings()

        self.vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings
        )

        self.reranker = Reranker()

        self.generator = Generator(
            api_key=os.getenv("GROQ_API_KEY")
        )

        # query cache
        self.cache: dict[str, str] = {}

        # reload existing chunks from ChromaDB on startup
        existing = self.vectorstore.get()
        if existing and existing["documents"]:
            self.chunks = existing["documents"]
            self.retriever = HybridRetriever(
                vectorstore=self.vectorstore,
                chunks=self.chunks
            )
        else:
            self.chunks: list[str] = []
            self.retriever = None

    def _get_cache_key(self, query: str) -> str:
        return query.strip().lower()

    def ingest(self, text: str, document_type: str = "general") -> int:
        """
        Chunks, embeds, and stores a document.
        Returns number of chunks stored.
        """
        # chunk the document
        new_chunks = chunk_legal_document(text, document_type)

        # store chunks in memory for BM25
        self.chunks.extend(new_chunks)

        # embed and store in ChromaDB
        self.vectorstore.add_texts(new_chunks)

        # rebuild BM25 retriever with updated chunks
        self.retriever = HybridRetriever(
            vectorstore=self.vectorstore,
            chunks=self.chunks
        )

        return len(new_chunks)

    def query(self, question: str, k: int = 10, top_k: int = 3) -> dict:
        """
        Runs full RAG pipeline for a question.
        Returns answer and sources.
        """
        # check cache first
        cache_key = self._get_cache_key(question)
        if cache_key in self.cache:
            return {
                "answer": self.cache[cache_key],
                "sources": [],
                "cached": True
            }

        # check retriever is ready
        if self.retriever is None:
            return {
                "answer": "No documents ingested yet. Please upload documents first.",
                "sources": [],
                "cached": False
            }

        # stage 1 — hybrid retrieval
        retrieved_chunks = self.retriever.retrieve(question, k=k)

        # stage 2 — reranking
        reranked_chunks = self.reranker.rerank(question, retrieved_chunks, top_k=top_k)

        # stage 3 — generation
        answer = self.generator.generate(question, reranked_chunks)

        # store in cache
        # store in cache
        if len(self.cache) > 1000:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[cache_key] = answer

        # build sources list
        sources = [
            {"content": chunk[:200], "score": score}
            for chunk, score in reranked_chunks
        ]

        return {
            "answer": answer,
            "sources": sources,
            "cached": False
        }

    async def query_stream(
        self,
        question: str,
        k: int = 10,
        top_k: int = 3
    ) -> AsyncGenerator[str, None]:
        """
        Streaming version of query.
        Yields tokens one by one.
        """
        if self.retriever is None:
            yield "No documents ingested yet. Please upload documents first."
            return

        # stage 1 — hybrid retrieval
        retrieved_chunks = self.retriever.retrieve(question, k=k)

        # stage 2 — reranking
        reranked_chunks = self.reranker.rerank(question, retrieved_chunks, top_k=top_k)

        # stage 3 — streaming generation
        async for token in self.generator.generate_stream(question, reranked_chunks):
            yield token