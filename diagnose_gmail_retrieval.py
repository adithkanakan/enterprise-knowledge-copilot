
from pathlib import Path

from src.retrieval import (
    load_vector_store,
    retrieve_documents,
    retrieve_documents_rrf,
)

QUESTION = "What does Gmail SMTP error 452 4.2.2 indicate?"

TARGET_FILE = "gmail_smtp_errors.pdf"

OUTPUT_DIR = Path("evaluation/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def print_results(method, documents):
    print("\n" + "=" * 90)
    print(f"{method} — TOP {len(documents)} RESULTS")
    print("=" * 90)

    target_found = False

    for rank, doc in enumerate(documents, start=1):
        source = Path(
            str(doc.metadata.get("source", "Unknown"))
        ).name

        page = doc.metadata.get("page_number", "Unknown")
        content = doc.page_content

        is_target = TARGET_FILE.lower() in source.lower()

        if is_target:
            target_found = True

        print(f"\nRANK: {rank}")
        print(f"SOURCE: {source}")
        print(f"PAGE: {page}")
        print(f"TARGET DOCUMENT: {is_target}")
        print("-" * 70)
        print(content[:2500])

    print("\nTarget document found:", target_found)


def main():
    print("Loading existing FAISS index...")

    vector_store = load_vector_store()

    print("Loaded vectors:", vector_store.index.ntotal)
    print("\nQuestion:", QUESTION)

    similarity = retrieve_documents(
        question=QUESTION,
        vector_store=vector_store,
        k=10,
        search_type="similarity",
    )

    mmr = retrieve_documents(
        question=QUESTION,
        vector_store=vector_store,
        k=10,
        search_type="mmr",
        fetch_k=30,
        lambda_mult=0.7,
    )

    rrf = retrieve_documents_rrf(
        question=QUESTION,
        vector_store=vector_store,
        k=10,
        candidate_k=15,
        fetch_k=30,
        lambda_mult=0.7,
    )

    print_results("SIMILARITY", similarity)
    print_results("MMR", mmr)
    print_results("RRF", rrf)

    print("\n" + "=" * 90)
    print("DIAGNOSTIC COMPLETE")
    print("No index modifications were performed.")
    print("=" * 90)


if __name__ == "__main__":
    main()
