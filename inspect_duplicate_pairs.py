from src.ingestion import load_documents

docs, errors = load_documents()

target_files = [
    "aws/ec2_connectivity.pdf",
    "aws/ec2_troubleshooting.pdf",
    "cisco/ip_routing_troubleshooting.pdf",
    "cisco/tcp_ip_troubleshooting.pdf",
]

for filename in target_files:

    matching_pages = [
        doc for doc in docs
        if doc.metadata["source"] == filename
    ]

    print("\n" + "=" * 70)
    print("FILE:", filename)
    print("=" * 70)

    if matching_pages:
        print("PAGE COUNT:", len(matching_pages))
        print("\nFIRST PAGE CONTENT:\n")
        print(matching_pages[0].page_content[:2000])
    else:
        print("FILE NOT FOUND")