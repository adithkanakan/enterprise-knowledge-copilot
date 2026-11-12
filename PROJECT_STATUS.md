
# Enterprise Knowledge Copilot — Project Status

> An evidence-based IT Helpdesk assistant using Retrieval-Augmented Generation (RAG).

**Project status:** Development in progress

---

## 1. Project Overview

The Enterprise Knowledge Copilot answers internal IT support questions using company knowledge documents. It provides evidence-based answers with source references and clearly states when the available documents do not contain enough information.

### Technology Stack

- Python
- LangChain
- Google Gemini
- Hugging Face Embeddings — `sentence-transformers/all-MiniLM-L6-v2`
- FAISS
- RAGAS
- Streamlit

---

## 2. Development Progress

### Phase 1 — Environment and Project Setup

- [x] Create project structure
- [x] Create Python virtual environment
- [x] Install required dependencies
- [x] Create `requirements.txt`
- [ ] Verify dependency compatibility using `pip check`
- [ ] Verify required package imports

### Phase 2 — Document Collection and Ingestion

- [x] Collect and organize PDF documents
- [x] Organize documents by source category
- [x] Inspect document readability and extracted text
- [x] Identify and remove exact duplicate documents
- [x] Exclude irrelevant documentation index pages
- [x] Implement `src/ingestion.py`
- [x] Test PDF text extraction
- [x] Verify source and page metadata
- [x] Verify ingestion errors and empty pages

**Current corpus:**
- 23 unique PDF documents
- 277 extracted pages
- 0 ingestion errors
- 0 empty extracted pages
- 0 exact duplicate groups

### Phase 3 — Text Chunking

- [ ] Implement `src/chunking.py`
- [ ] Understand the implementation line by line
- [ ] Preserve source metadata in chunks
- [ ] Test different chunk sizes
- [ ] Test different chunk overlaps
- [ ] Compare chunking configurations
- [ ] Select a configuration based on actual results

### Phase 4 — Embeddings and Vector Database

- [ ] Generate embeddings using Sentence Transformers
- [ ] Create the FAISS vector index
- [ ] Store and persist the vector index
- [ ] Test vector index loading
- [ ] Verify document-to-vector mapping

### Phase 5 — Retrieval

- [ ] Implement `src/retrieval.py`
- [ ] Implement similarity search
- [ ] Test different top-k values
- [ ] Retrieve relevant document chunks
- [ ] Preserve source references
- [ ] Evaluate retrieval quality

### Phase 6 — Answer Generation

- [ ] Implement `src/generation.py`
- [ ] Integrate Google Gemini
- [ ] Design the grounded prompt
- [ ] Generate answers using retrieved context
- [ ] Include source citations and supporting excerpts
- [ ] Handle insufficient evidence
- [ ] Test unsupported and out-of-scope questions

### Phase 7 — Logging and Monitoring

- [ ] Implement `src/logging_utils.py`
- [ ] Log user questions
- [ ] Log generated answers
- [ ] Log retrieved contexts
- [ ] Record timestamps and response latency
- [ ] Store logs in JSON or CSV format

### Phase 8 — Evaluation

- [ ] Prepare 30–50 human-reviewed evaluation questions
- [ ] Prepare reference answers
- [ ] Implement `src/evaluation.py`
- [ ] Evaluate faithfulness
- [ ] Evaluate answer relevancy
- [ ] Evaluate context precision
- [ ] Evaluate context recall
- [ ] Compare chunking configurations
- [ ] Compare retrieval top-k values
- [ ] Analyze at least three failure cases
- [ ] Document actual evaluation results

### Phase 9 — Streamlit Application

- [ ] Implement `app.py`
- [ ] Build the user interface
- [ ] Connect the complete RAG pipeline
- [ ] Display generated answers
- [ ] Display source documents and excerpts
- [ ] Display relevant interaction details
- [ ] Provide results or log export functionality

### Phase 10 — Finalization

- [ ] Complete `README.md`
- [ ] Document system architecture
- [ ] Document installation and execution steps
- [ ] Document experiments and evaluation results
- [ ] Document limitations and future improvements
- [ ] Perform end-to-end testing
- [ ] Prepare the GitHub repository for portfolio presentation

---

## 3. Project Structure

```text
enterprise-knowledge-copilot/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── PROJECT_STATUS.md
├── src/
│   ├── config.py
│   ├── ingestion.py
│   ├── chunking.py
│   ├── retrieval.py
│   ├── generation.py
│   ├── logging_utils.py
│   └── evaluation.py
├── data/
│   ├── raw/
│   └── processed/
├── indexes/
├── logs/
├── evaluation/
└── notebooks/
```

---

## 4. Environment Details

| Component | Version |
|---|---|
| Python | 3.13.17 |
| pip | 26.3 |
| LangChain | 1.4.9 |
| LangChain Core | 1.7.6 |
| LangChain Community | 0.4.2 |
| LangChain Text Splitters | 1.1.3 |
| LangChain Google GenAI | 4.4.3 |
| Google GenAI | 2.34.0 |
| FAISS CPU | 1.15.1 |
| Sentence Transformers | 6.2.0 |
| RAGAS | 0.4.3 |
| Streamlit | 1.66.1 |
| PyPDF | 6.21.4 |
| Python-docx | 1.2.0 |
| Pandas | 3.1.1 |
| Python-dotenv | 1.2.4 |
| NumPy | 2.5.4 |
| PyTorch | 2.15.0 |
| Transformers | 5.22.1 |
| Scikit-learn | 1.9.1 |
| SciPy | 1.18.1 |
| Pydantic | 2.14.2 |
| SQLAlchemy | 2.1.8 |

**Note:** Package versions are based on the reported `pip list` output. Dependency compatibility and import checks are still pending.

---

## 5. Engineering Decisions

- [x] Use PDF documents as the initial knowledge corpus.
- [x] Preserve source filename and page number during ingestion.
- [ ] Preserve metadata throughout the complete RAG pipeline.
- [ ] Compare multiple chunking configurations.
- [ ] Compare retrieval top-k values.
- [ ] Use retrieved evidence to generate answers.
- [ ] Include source citations in generated answers.
- [ ] Return an insufficient-evidence response when appropriate.
- [ ] Log questions, answers, retrieved contexts, and latency.
- [ ] Evaluate using human-reviewed reference answers.
- [ ] Report only actual measured performance results.

**Insufficient-evidence response:**

"The documents do not contain enough information."

---

## 6. Experiment Tracking

Record actual experiment configurations and results here.

| Experiment | Configuration | Result |
|---|---|---|
| Chunking experiment 1 | Pending | Not run |
| Chunking experiment 2 | Pending | Not run |
| Retrieval top-k comparison | Pending | Not run |
| RAGAS evaluation | Pending | Not run |

### RAGAS Evaluation Results

| Metric | Actual Score |
|---|---:|
| Faithfulness | Not evaluated |
| Answer relevancy | Not evaluated |
| Context precision | Not evaluated |
| Context recall | Not evaluated |

**Target:** Aim for scores above 0.95 where supported by the data and evaluation design. This is a target, not a measured result or guarantee.

---

## 7. Known Risks and Pending Checks

- [ ] Verify dependency compatibility.
- [ ] Inspect chunking output for PDF headers, footers, and navigation noise.
- [ ] Check for semantic overlap between different documents.
- [ ] Verify Gemini API and model compatibility before integration.
- [ ] Test insufficient-evidence handling.
- [ ] Check retrieval quality before evaluating answer generation.

---

## 8. Current Next Steps

1. Complete dependency verification.
2. Implement `src/chunking.py`.
3. Test multiple chunk sizes and overlaps.
4. Understand the chunking code line by line.
5. Proceed to embeddings and vector storage only after chunking is verified.

---

## 9. Completion Summary

| Phase | Status |
|---|---|
| Environment and setup | In progress |
| Document collection and ingestion | Complete |
| Text chunking | Pending |
| Embeddings and vector database | Pending |
| Retrieval | Pending |
| Answer generation | Pending |
| Logging and monitoring | Pending |
| Evaluation | Pending |
| Streamlit application | Pending |
| Finalization | Pending |
