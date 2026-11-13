
from src.retrieval import load_vector_store

vector_store = load_vector_store()

docstore = vector_store.docstore
mapping = vector_store.index_to_docstore_id

print("Total vectors:", vector_store.index.ntotal)
print("Mapping entries:", len(mapping))

gmail_count = 0
matching_chunks = 0

for position, doc_id in mapping.items():
    document = docstore.search(doc_id)

    if not hasattr(document, "metadata"):
        continue

    source = str(document.metadata.get("source", ""))

    if "gmail" not in source.lower():
        continue

    gmail_count += 1

    content = document.page_content
    contains_452 = (
        "452" in content and "4.2.2" in content
    )
    contains_storage = "storage" in content.lower()

    if contains_452 or contains_storage:
        matching_chunks += 1

    print("\n" + "=" * 80)
    print("FAISS POSITION:", position)
    print("DOCUMENT ID:", doc_id)
    print("SOURCE:", source)
    print("PAGE:", document.metadata.get("page_number"))
    print("START INDEX:", document.metadata.get("start_index"))
    print("CONTAINS 452:", "452" in content)
    print("CONTAINS 4.2.2:", "4.2.2" in content)
    print("CONTAINS STORAGE:", contains_storage)
    print("-" * 80)
    print(content[:2000])

print("\n" + "=" * 80)
print("TOTAL GMAIL CHUNKS:", gmail_count)
print("MATCHING 452/STORAGE CHUNKS:", matching_chunks)
print("Inspection completed. No index changes.")
