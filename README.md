# Enterprise Knowledge Copilot

Enterprise Knowledge Copilot is an evidence-grounded IT Helpdesk RAG assistant that answers technical questions using a curated corpus of documentation. It retrieves passages using FAISS, generates answers with Google Gemini using evidence-constrained prompting, provides traceable source citations with original PDF page numbers and supporting excerpts, abstains when evidence is insufficient, and logs interactions to JSON.

## Technology Stack

- **RAG & Generation:** Python, LangChain, Google Gemini, Streamlit
- **Embeddings & Vector Store:** `sentence-transformers/all-MiniLM-L6-v2`, FAISS
- **Evaluation & Local Judging:** RAGAS, Ollama (Qwen 2.5 3B)

## Repository Structure

```text
├── app.py, requirements.txt, RULES.md, PROJECT_PROGRESS.md
├── src/                 # config, ingestion, chunking, retrieval, generation, evaluation
├── data/                # raw/ (PDFs), processed/ (Markdown), source_manifest.csv
├── docs/archive/        # archived status docs
├── indexes/             # faiss_index/ (1,085 vectors), faiss_index_backup/
├── evaluation/          # datasets/, results/
├── logs/                # interaction logs
├── notebooks/           # prototyping
└── scripts/archive/     # diagnostic and investigation scripts
```

> **Note:** `data/raw`, `data/processed`, and `indexes/` are excluded from git. The repository cannot yet rebuild its own FAISS index (the index-build script is planned).

## Current Status

- **Working:** End-to-end Streamlit app with Gemini generation, FAISS MMR retrieval (`k=4, fetch_k=12`), citations, and logging. Ingestion covers 30 PDFs (268 pages, 0 errors).
- **Baseline retrieval:** Source Hit@1: 80.00%, Hit@4: 97.78% on 45 answerable questions (MMR 0.7 baseline; passage-level retrieval evaluation still needed). 5/5 unanswerable questions correctly abstained.
- **Pilot evaluation:** 20-sample pilot RAGAS results: Answer relevancy 0.7909, Context precision 0.9167, Context recall 0.8750. Experimental local HHEM claim-support pilot: 0.7142 (not official RAGAS Faithfulness).
- **In progress / Not done:** Markdown normalization under `data/processed/` has not passed quality validation; index-build script is not yet implemented; the >90% RAGAS target is an aspirational goal not yet achieved.

## Setup

Setup instructions will be added once the index build script exists.

## Planned Work

1. Validate and finalize deterministic PDF-to-Markdown normalization across all 30 PDFs.
2. Implement safe index-build script and rebuild FAISS index in a separate directory.
3. Implement passage-level retrieval evaluation and resolve known benchmark mismatches.
4. Run formal RAGAS evaluations on held-out test splits to target >90% performance.
5. Finalize portfolio documentation, architecture diagrams, and demo polish.
