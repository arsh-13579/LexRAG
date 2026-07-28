from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter


def chunk_legal_document(text: str, document_type: str = "general") -> list[str]:
    """
    Chunks legal documents smartly based on document type.
    - contracts: structure-aware (splits on headings first)
    - judgments/general: recursive character splitting
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

        # each header chunk may still be too large — split further recursively
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

    else:
        # court judgments and compliance docs — pure recursive splitting
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        return recursive_splitter.split_text(text)