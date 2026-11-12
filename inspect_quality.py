from collections import defaultdict
from pathlib import Path

from src.ingestion import load_documents


docs, errors = load_documents()

# Group extracted pages by source PDF
files = defaultdict(list)

for doc in docs:
    files[doc.metadata["source"]].append(doc)


print("=" * 80)
print("DOCUMENT QUALITY REPORT")
print("=" * 80)

for filename, pages in sorted(files.items()):

    total_characters = sum(
        len(page.page_content) for page in pages
    )

    average_characters = total_characters / len(pages)

    empty_pages = sum(
        1 for page in pages
        if not page.page_content.strip()
    )

    print(f"\nFile: {filename}")
    print(f"Extracted pages: {len(pages)}")
    print(f"Total characters: {total_characters}")
    print(f"Average characters per page: {average_characters:.0f}")
    print(f"Empty extracted pages: {empty_pages}")

print("\n" + "=" * 80)
print("TOTAL DOCUMENTS:", len(files))
print("TOTAL EXTRACTED PAGES:", len(docs))
print("INGESTION ERRORS:", len(errors))
print("=" * 80)