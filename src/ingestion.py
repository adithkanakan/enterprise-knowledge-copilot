from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document


def load_documents(
    data_dir: str | Path = "data/raw",
) -> tuple[list[Document], list[dict]]:
    """
    Load PDF files page by page and return LangChain documents
    along with any ingestion errors.
    """

    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Document directory does not exist: {data_path}"
        )

    if not data_path.is_dir():
        raise NotADirectoryError(
            f"Expected a directory, received: {data_path}"
        )

    pdf_files = sorted(
        file for file in data_path.rglob("*")
        if file.is_file() and file.suffix.lower() == ".pdf"
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found inside: {data_path}"
        )

    documents: list[Document] = []
    errors: list[dict] = []

    for pdf_file in pdf_files:
        try:
            reader = PdfReader(str(pdf_file))

            if reader.is_encrypted:
                raise ValueError("PDF is encrypted and requires a password.")

            extracted_pages = 0

            for page_number, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                    text = text.strip()

                    if not text:
                        continue

                    document = Document(
                        page_content=text,
                        metadata={
                            "source": pdf_file.relative_to(data_path).as_posix(),
                            "file_name": pdf_file.name,
                            "file_type": "pdf",
                            "page_number": page_number,
                        },
                    )

                    documents.append(document)
                    extracted_pages += 1

                except Exception as page_error:
                    errors.append(
                        {
                            "file": str(pdf_file),
                            "page": page_number,
                            "error": str(page_error),
                        }
                    )

            if extracted_pages == 0:
                errors.append(
                    {
                        "file": str(pdf_file),
                        "page": None,
                        "error": "No extractable text found in this PDF.",
                    }
                )

        except Exception as file_error:
            errors.append(
                {
                    "file": str(pdf_file),
                    "page": None,
                    "error": str(file_error),
                }
            )

    return documents, errors