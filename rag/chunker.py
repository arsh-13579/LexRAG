from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker


def chunk_legal_document(text: str, document_type: str = "general", embeddings=None) -> list[str]:
    """
    Chunks legal documents smartly based on document type.
    - contract: structure-aware (splits on headings first)
    - judgment: semantic chunking (splits on meaning shift)
    - general: recursive character splitting
    """

    if document_type == "contract":
        # contracts have clear headings — use structure-aware chunking first
        headers_to_split_on = [
            ("#", "Section"),
            ("##", "Subsection"),
            ("###", "Clause"),
        ]
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        header_chunks = header_splitter.split_text(text)

        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )

        final_chunks = []
        for chunk in header_chunks:
            sub_chunks = recursive_splitter.split_text(chunk.page_content)
            final_chunks.extend(sub_chunks)

        return final_chunks

    elif document_type == "judgment":
        # large legal judgments — semantic chunking for meaning-based splits
        if embeddings is None:
            # fallback to recursive if no embeddings provided
            recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                separators=["\n\n", "\n", " ", ""]
            )
            return recursive_splitter.split_text(text)

        semantic_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90
        )
        return semantic_splitter.split_text(text)

    else:
        # general documents — pure recursive splitting
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        return recursive_splitter.split_text(text)