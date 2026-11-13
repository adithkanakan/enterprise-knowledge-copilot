from collections import defaultdict

from src.ingestion import load_documents


# Load documents
docs, errors = load_documents()

# Group pages by publisher
groups = defaultdict(list)

for doc in docs:
    publisher = doc.metadata["source"].split("/")[0]
    groups[publisher].append(doc)


# Display total pages
print("TOTAL EXTRACTED PAGES:", len(docs))

print("\nPUBLISHER-WISE EXTRACTION:")

for publisher, pages in sorted(groups.items()):
    print(f"{publisher}: {len(pages)} pages")


# Display sample content
print("\nSAMPLE EXTRACTED CONTENT:")

for publisher, pages in sorted(groups.items()):

    first_page = pages[0]

    print("\n" + "=" * 70)
    print(f"PUBLISHER: {publisher}")
    print(f"FILE: {first_page.metadata['file_name']}")
    print(f"PAGE: {first_page.metadata['page_number']}")
    print("=" * 70)

    print(first_page.page_content[:1200])


# Display errors
print("\nINGESTION ERRORS:", len(errors))

for error in errors:
    print(error)