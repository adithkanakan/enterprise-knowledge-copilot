
from src.retrieval import load_vector_store

vs = load_vector_store()

queries = [
    "What does Gmail SMTP error 452 4.2.2 indicate?",
    "452 4.2.2 recipient inbox out of storage space",
    "Gmail recipient inbox out of storage space",
]

for q in queries:
    print("\nQUESTION:", q)

    docs = vs.similarity_search(q, k=5)

    for i, d in enumerate(docs, 1):
        print(
            i,
            d.metadata.get("source"),
            d.metadata.get("page_number"),
            "452" in d.page_content,
            "4.2.2" in d.page_content,
        )
