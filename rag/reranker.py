from sentence_transformers import CrossEncoder


class Reranker:
    """
    Re-ranks retrieved chunks using a cross-encoder model.
    More precise than embedding similarity — reads query and chunk together.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: list[str],
        top_k: int = 3
    ) -> list[tuple[str, float]]:
        """
        Takes query and list of candidate chunks.
        Returns top_k chunks sorted by relevance score, highest first.
        Each item is a tuple of (chunk_text, relevance_score).
        """

        if not chunks:
            return []

        # build query-chunk pairs for cross-encoder
        pairs = [[query, chunk] for chunk in chunks]

        # score every pair
        scores = self.model.predict(pairs)

        # zip chunks with their scores
        scored_chunks = list(zip(chunks, scores))

        # sort by score descending
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # return top_k with scores rounded for cleanliness
        return [(chunk, round(float(score), 4)) for chunk, score in scored_chunks[:top_k]]