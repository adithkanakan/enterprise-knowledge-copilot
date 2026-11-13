from src.ingestion import load_documents

docs, errors = load_documents()

target_files = [
    "cisco/broken_wireless_lan.pdf",
    "microsoft/network_troubleshooting.pdf",
]

for filename in target_files:

    matching_pages = [
        doc for doc in docs
        if doc.metadata["source"] == filename
    ]

    print("\n" + "=" * 75)
    print("DOCUMENT:", filename)
    print("=" * 75)

    for page in matching_pages:
        print(f"\n--- PAGE {page.metadata['page_number']} ---")
        print(page.page_content[:1500])

print("\nTOTAL ERRORS:", len(errors))