from langchain_groq import ChatGroq
from typing import AsyncGenerator


class Generator:
    """
    Handles LLM prompt construction and response generation.
    Supports both regular and streaming responses.
    """

    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        self.llm = ChatGroq(
            api_key=api_key,
            model=model_name,
            temperature=0.0
        )

    def _build_prompt(self, query: str, chunks: list[tuple[str, float]]) -> str:
        """
        Builds a grounded prompt from query and reranked chunks.
        """
        # extract just the text from (chunk, score) tuples
        context_pieces = [chunk for chunk, score in chunks]
        context = "\n\n---\n\n".join(context_pieces)

        prompt = f"""You are a precise legal document assistant.
Answer the question using ONLY the context provided below.
If the answer is not found in the context, respond with "I cannot find this information in the provided documents."
Do not use any outside knowledge. Do not make assumptions.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

        return prompt

    def generate(self, query: str, chunks: list[tuple[str, float]]) -> str:
        """
        Generates a complete answer (non-streaming).
        Returns full answer as a string.
        """
        if not chunks:
            return "No relevant documents found. Please ingest documents first."

        prompt = self._build_prompt(query, chunks)
        response = self.llm.invoke(prompt)
        return response.content

    async def generate_stream(
        self,
        query: str,
        chunks: list[tuple[str, float]]
    ) -> AsyncGenerator[str, None]:
        """
        Generates answer token by token (streaming).
        Yields one token at a time.
        """
        if not chunks:
            yield "No relevant documents found. Please ingest documents first."
            return

        prompt = self._build_prompt(query, chunks)

        async for chunk in self.llm.astream(prompt):
            if chunk.content:
                yield chunk.content