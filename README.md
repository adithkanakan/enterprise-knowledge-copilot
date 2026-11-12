# Enterprise Knowledge Copilot

An evidence-based AI support assistant using Retrieval-Augmented Generation (RAG).

## Project status
This repository is a starter structure. The modules contain clear placeholders for the implementation phases.

## Planned workflow
1. Load PDF, TXT, DOCX and CSV documents.
2. Split extracted text into chunks while preserving source metadata.
3. Generate embeddings and save/search them with FAISS.
4. Retrieve relevant passages for a user question.
5. Ask Gemini to answer using only the retrieved evidence.
6. Save interactions and response latency.
7. Evaluate a reviewed question set with RAGAS.
8. Present the assistant and evaluation results in Streamlit.

## Folder structure

```text
enterprise-knowledge-copilot/
├── data/
│   ├── raw/                 # Add source documents here
│   └── processed/           # Optional extracted/processed data
├── indexes/                 # Persisted FAISS index
├── evaluation/
│   ├── datasets/            # Reviewed evaluation questions/reference answers
│   └── results/             # Saved RAGAS results
├── logs/                    # Interaction logs
├── notebooks/               # Experiments and prototyping
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── ingestion.py
│   ├── chunking.py
│   ├── retrieval.py
│   ├── generation.py
│   ├── logging_utils.py
│   └── evaluation.py
├── app.py                   # Streamlit entry point
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Setup

### 1. Create and activate a virtual environment

Windows PowerShell:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Gemini
Copy `.env.example` to `.env` and add your Gemini API key.

Never commit `.env` or expose your API key.

### 4. Add documents
Place permitted sample knowledge-base documents in `data/raw/`.

### 5. Run the app
```bash
streamlit run app.py
```

The starter app displays setup guidance until the pipeline is implemented.

## Important
- Use documents you have permission to use.
- Reference answers in the evaluation dataset must be reviewed independently.
- Do not claim evaluation scores or features until they have actually been implemented and measured.
