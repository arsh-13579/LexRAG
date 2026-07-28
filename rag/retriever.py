from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from typing import Any
import numpy as np


class HybridRetriever:
    """
    Combines vector search (semantic) and BM25 (keyword) retrieval.
    Uses Reciprocal Rank Fusion (RRF) to merge results.
    """

    def __init__(self, vectorstore: Chroma, chunks: list[str], rrf_k: int = 60):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.rrf_k = rrf_k

        # build BM25 index from chunks
        tokenized_chunks = [chunk.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def _vector_search(self, query: str, k: int) -> list[tuple[str, int]]:
        """
        Runs vector search. Returns list of (chunk_text, original_index) tuples.
        """
        results = self.vectorstore.similarity_search(query, k=k)
        output = []
        for doc in results:
            content = doc.page_content
            # find index of this chunk in original chunks list
            if content in self.chunks:
                idx = self.chunks.index(content)
            else:
                idx = -1
            output.append((content, idx))
        return output

    def _bm25_search(self, query: str, k: int) -> list[tuple[str, int]]:
        """
        Runs BM25 keyword search. Returns list of (chunk_text, original_index) tuples.
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # get top k indices sorted by score
        top_k_indices = np.argsort(scores)[::-1][:k]

        output = []
        for idx in top_k_indices:
            output.append((self.chunks[idx], int(idx)))
        return output

    def _rrf_fusion(
        self,
        vector_results: list[tuple[str, int]],
        bm25_results: list[tuple[str, int]],
    ) -> list[str]:
        """
        Merges vector and BM25 results using Reciprocal Rank Fusion.
        Chunks appearing high in BOTH lists get boosted to the top.
        """
        rrf_scores = {}

        # score from vector search ranking
        for rank, (content, idx) in enumerate(vector_results):
            if content not in rrf_scores:
                rrf_scores[content] = 0
            rrf_scores[content] += 1 / (self.rrf_k + rank + 1)

        # score from BM25 ranking
        for rank, (content, idx) in enumerate(bm25_results):
            if content not in rrf_scores:
                rrf_scores[content] = 0
            rrf_scores[content] += 1 / (self.rrf_k + rank + 1)

        # sort by combined RRF score
        sorted_chunks = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [content for content, score in sorted_chunks]

    def retrieve(self, query: str, k: int = 10) -> list[str]:
        """
        Main retrieval function.
        Runs both searches, fuses results with RRF, returns top k chunks.
        """
        vector_results = self._vector_search(query, k=k)
        bm25_results = self._bm25_search(query, k=k)
        fused_results = self._rrf_fusion(vector_results, bm25_results)
        return fused_results[:k]