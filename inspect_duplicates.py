import hashlib
from collections import defaultdict

from src.ingestion import load_documents


docs, errors = load_documents()

# Group extracted pages by PDF
files = defaultdict(list)

for doc in docs:
    files[doc.metadata["source"]].append(doc)


# Create a fingerprint for each complete PDF's extracted text
fingerprints = defaultdict(list)

for filename, pages in sorted(files.items()):

    full_text = "\n".join(
        page.page_content for page in pages
    )

    normalized_text = " ".join(full_text.split())

    fingerprint = hashlib.sha256(
        normalized_text.encode("utf-8")
    ).hexdigest()

    fingerprints[fingerprint].append(filename)


# Display exact duplicate groups
print("=" * 70)
print("EXACT DUPLICATE DOCUMENT CHECK")
print("=" * 70)

duplicate_groups = 0

for fingerprint, filenames in fingerprints.items():

    if len(filenames) > 1:
        duplicate_groups += 1

        print("\nPOSSIBLE DUPLICATE GROUP:")

        for filename in filenames:
            print(" -", filename)

if duplicate_groups == 0:
    print("\nNo exact text duplicates found.")

print("\nDuplicate groups:", duplicate_groups)
print("Unique extracted documents:", len(fingerprints))
print("Total documents:", len(files))