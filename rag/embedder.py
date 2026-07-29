from langchain_huggingface import HuggingFaceEmbeddings


def load_embeddings() -> HuggingFaceEmbeddings:
    """
    Loads the sentence-transformer embedding model.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 64
        }
            
    )
    return embeddings