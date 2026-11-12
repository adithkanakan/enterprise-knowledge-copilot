
"""Document embedding, FAISS storage, and retrieval utilities."""

from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import EMBEDDING_MODEL, FAISS_INDEX_DIR


def create_vector_store(
    chunks: list[Document],
    index_dir: str | Path = FAISS_INDEX_DIR,
) -> FAISS:
    """
    Convert document chunks into embeddings, create a FAISS
    vector store, and save it locally.
    """

    if not chunks:
        raise ValueError(
            "Cannot create a vector store from an empty chunk list."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)

    vector_store.save_local(str(index_path))

    return vector_store


def load_vector_store(
    index_dir: str | Path = FAISS_INDEX_DIR,
) -> FAISS:
    """
    Load a previously saved FAISS vector store from local storage.
    """

    index_path = Path(index_dir)

    if not index_path.is_dir():
        raise FileNotFoundError(
            f"FAISS index directory does not exist: {index_path}"
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vector_store = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vector_store


def retrieve_documents(
    question: str,
    vector_store: FAISS,
    k: int = 4,
    search_type: str = "mmr",
    fetch_k: int = 12,
    lambda_mult: float = 0.7,
) -> list[Document]:
    """
    Retrieve relevant document chunks using similarity search or MMR.

    MMR balances query relevance and diversity among retrieved chunks.
    """

    # Validate the question.
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    # Validate the vector store.
    if not isinstance(vector_store, FAISS):
        raise TypeError("vector_store must be a FAISS vector store.")

    # Validate the number of results.
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")

    # Validate the search type.
    if search_type not in {"similarity", "mmr"}:
        raise ValueError(
            "search_type must be either 'similarity' or 'mmr'."
        )

    # Validate the candidate pool.
    if (
        isinstance(fetch_k, bool)
        or not isinstance(fetch_k, int)
        or fetch_k <= 0
    ):
        raise ValueError("fetch_k must be a positive integer.")

    if search_type == "mmr" and fetch_k < k:
        raise ValueError("fetch_k must be greater than or equal to k.")

    # Validate the MMR diversity parameter.
    if (
        isinstance(lambda_mult, bool)
        or not isinstance(lambda_mult, (int, float))
        or not 0 <= lambda_mult <= 1
    ):
        raise ValueError("lambda_mult must be between 0 and 1.")

    # Retrieve documents using the selected strategy.
    if search_type == "similarity":

        retrieved_documents = vector_store.similarity_search(
            query=question.strip(),
            k=k,
        )

    else:

        retrieved_documents = (
            vector_store.max_marginal_relevance_search(
                query=question.strip(),
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
            )
        )

    return retrieved_documents


def _get_chunk_identity(document: Document) -> tuple:
    """
    Create a stable identity for a document chunk.

    This helps identify the same chunk if it appears in both
    the similarity and MMR result lists.
    """

    return (
        document.metadata.get("source"),
        document.metadata.get("page_number"),
        document.metadata.get("start_index"),
        document.page_content,
    )


def retrieve_documents_rrf(
    question: str,
    vector_store: FAISS,
    k: int = 4,
    candidate_k: int = 8,
    fetch_k: int = 20,
    lambda_mult: float = 0.7,
    rrf_k: int = 60,
) -> list[Document]:
    """
    Combine similarity and MMR rankings using Reciprocal Rank Fusion.

    Both retrieval methods produce candidate rankings. RRF combines
    those rankings and returns the highest-scoring unique chunks.
    """

    # Validate the question.
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    # Validate the vector store.
    if not isinstance(vector_store, FAISS):
        raise TypeError("vector_store must be a FAISS vector store.")

    # Validate result counts.
    for name, value in {
        "k": k,
        "candidate_k": candidate_k,
        "fetch_k": fetch_k,
        "rrf_k": rrf_k,
    }.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer.")

    if candidate_k < k:
        raise ValueError("candidate_k must be greater than or equal to k.")

    if fetch_k < candidate_k:
        raise ValueError(
            "fetch_k must be greater than or equal to candidate_k."
        )

    # Validate the MMR diversity parameter.
    if (
        isinstance(lambda_mult, bool)
        or not isinstance(lambda_mult, (int, float))
        or not 0 <= lambda_mult <= 1
    ):
        raise ValueError("lambda_mult must be between 0 and 1.")

    clean_question = question.strip()

    # Retrieve the similarity ranking.
    similarity_results = vector_store.similarity_search(
        query=clean_question,
        k=candidate_k,
    )

    # Retrieve the MMR ranking.
    mmr_results = vector_store.max_marginal_relevance_search(
        query=clean_question,
        k=candidate_k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
    )

    # Store each unique chunk and its combined RRF score.
    fused_results = {}

    # Combine both rankings.
    for ranked_documents in (
        similarity_results,
        mmr_results,
    ):
        for rank, document in enumerate(
            ranked_documents,
            start=1,
        ):
            identity = _get_chunk_identity(document)

            if identity not in fused_results:
                fused_results[identity] = {
                    "document": document,
                    "score": 0.0,
                }

            fused_results[identity]["score"] += (
                1.0 / (rrf_k + rank)
            )

    # Sort chunks by their combined RRF scores.
    sorted_results = sorted(
        fused_results.values(),
        key=lambda item: item["score"],
        reverse=True,
    )

    # Return only the requested number of documents.
    return [
        item["document"]
        for item in sorted_results[:k]
    ]
