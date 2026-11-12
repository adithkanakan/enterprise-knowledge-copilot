from src.retrieval import load_vector_store

vs = load_vector_store()

queries = {
    "Original": "What does Gmail SMTP error 452 4.2.2 indicate?",
    "Expanded": "Gmail SMTP 452 4.2.2 recipient inbox storage full mailbox quota exceeded"
}

for label, query in queries.items():
    print(f"\n{'='*15} {label} {'='*15}")

    results = vs.similarity_search_with_score(query, k=4)

    for i, (doc, score) in enumerate(results, 1):
        text = doc.page_content.lower()
        relevant = "452" in text and "4.2.2" in text

        print(
            f"{i}. Distance: {score:.4f} | "
            f"Relevant: {relevant} | "
            f"Source: {doc.metadata.get('source')} | "
            f"Page: {doc.metadata.get('page_number')}"
        )