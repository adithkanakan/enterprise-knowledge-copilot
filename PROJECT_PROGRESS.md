# Enterprise Knowledge Copilot --- Project Progress

**Status date:** 12 November 2026\
**Project stage:** Markdown normalization and quality validation\
**Purpose:** A portfolio-quality, evidence-grounded IT Helpdesk RAG
application.

------------------------------------------------------------------------

## 1. Project aim

Build a reliable, demonstrable Enterprise Knowledge Copilot that answers
IT Helpdesk questions using a curated technical-documentation corpus.

The project is intended to be a strong technical portfolio piece for
AI/ML, Data Science and LLM/RAG engineering roles. The quality target is
**greater than 90% on the selected RAGAS evaluation metrics**, measured
on a clearly defined, reproducible test set.

This is a target, not an achieved result or a guarantee. Metrics must be
reported honestly, with dataset size, evaluation method, judge model,
failures and limitations disclosed. Do not optimize by leaking
evaluation questions into the knowledge base or weakening the benchmark.

### Core behaviour

-   Answer only when the retrieved documents provide sufficient
    evidence.
-   Ground claims in retrieved source material.
-   Provide traceable citations with source filename and original PDF
    page number.
-   Include useful supporting excerpts.
-   Explicitly abstain when the documentation does not contain enough
    information.
-   Log interactions and evaluation results.
-   Never fabricate commands, configuration, procedures, citations or
    technical details.

## 2. Technology stack

  Component                Technology
  ------------------------ ------------------------------------------
  Language                 Python
  RAG orchestration        LangChain
  Answer generation        Google Gemini
  Embeddings               `sentence-transformers/all-MiniLM-L6-v2`
  Vector database          FAISS
  Evaluation               RAGAS
  Local evaluation model   Ollama / Qwen 2.5 3B
  User interface           Streamlit
  Logging                  JSON

**Development environment:** Windows, VS Code, Python 3.13.17, project
virtual environment `.venv`.

Project root: `enterprise-knowledge-copilot/`

## 3. Current project structure

``` text
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
│   ├── datasets/
│   └── results/
└── notebooks/
```

## 4. Knowledge base

The corpus contains 30 original PDF documents and 268 pages.

  Publisher            PDF documents     Pages
  ------------------ --------------- ---------
  AWS                             11        55
  Cisco                            3       118
  Google Workspace                 6        37
  Microsoft                        4        20
  Ubuntu                           6        38
  **Total**                   **30**   **268**

-   Original PDFs are stored under `data/raw/`, organized by publisher.
-   Original documents must remain unchanged and are the authoritative
    citation sources.
-   Initial extraction reported zero ingestion errors and zero
    empty-text pages.
-   Duplicate checks were performed. Previously identified exact
    duplicates were resolved.
-   Two Gmail documents overlap in subject matter but have not been
    established as exact duplicates.
-   AWS SSH troubleshooting documents are similar but distinct; do not
    remove them based only on text similarity.

## 5. Ingestion, chunking and vector index

Two chunking configurations were tested:

  Configuration     Chunk size   Overlap   Chunks
  --------------- ------------ --------- --------
  A                        800       150      974
  B                       1200       200      666

An existing FAISS index contains **1,085 vectors** at
`indexes/faiss_index`, with a backup at `indexes/faiss_index_backup`.

The current retrieval implementation supports: - Similarity search -
Maximal Marginal Relevance (MMR) - Reciprocal Rank Fusion (RRF)

Current application retrieval settings: - `k=4` - `fetch_k=12` -
`lambda_mult=0.7` - Search type: MMR

Do not overwrite or delete the existing index until a replacement has
been built separately and validated.

## 6. Retrieval benchmark

The original benchmark contains 50 questions: - 45 answerable
questions - 5 unanswerable questions - 9 answerable questions per
publisher

Source-file Hit@K results on the 45 answerable questions:

  -----------------------------------------------------------------------------
  Method              Hit@1        Hit@3        Hit@4  Avg. unique  Avg. unique
                                                              PDFs        pages
  ------------ ------------ ------------ ------------ ------------ ------------
  Similarity         80.00%      100.00%      100.00%         1.84         3.07

  MMR 0.7            80.00%       95.56%       97.78%         2.31         3.51

  MMR 0.85           80.00%       97.78%      100.00%         1.96         3.16

  RRF                80.00%       97.78%       97.78%         1.98         3.29
  -----------------------------------------------------------------------------

**Interpretation warning:** Hit@K checks whether the expected source PDF
appears in the retrieved results. It does not prove that the
answer-bearing passage was retrieved. A passage-level retrieval
evaluation is still needed.

Query expansion and keyword-assisted RRF were tested on the benchmark.
They did not improve the aggregate results, so query expansion was not
integrated.

### Known retrieval issue

The Gmail SMTP `452 4.2.2` answer-bearing passage exists in the index
but was ranked outside the top results for the original question. A
keyword-rich query retrieved it at rank 2. The chunk itself was intact;
this was a retrieval-ranking issue. The attempted query expansion did
not improve overall benchmark performance.

## 7. Application and generation

The Streamlit application has been implemented and smoke-tested.

Implemented capabilities: - FAISS retrieval - MMR retrieval
configuration - Gemini answer generation - Evidence-constrained
prompting - Source filename and page citations - Retrieved excerpts -
Latency display - JSON interaction logging - Abstention when evidence is
insufficient

Manual smoke tests covered: - Ubuntu EC2 default username - AWS SSH
timeout troubleshooting - EC2 status checks - Cisco IP connectivity -
Cisco router IP configuration (correctly abstained when exact procedure
was not supported) - Gmail SMTP 452 (abstained because retrieval missed
the relevant passage) - An out-of-scope SAP payroll question (correctly
abstained)

The app works end to end, but these smoke tests are not a formal
accuracy measurement.

## 8. Evaluation results to date

A 50-question evaluation run completed without RAGAS:

-   Total questions: 50
-   Answerable: 45
-   Unanswerable: 5
-   Average latency: 12.2881 seconds
-   Result CSV:
    `evaluation/results/evaluation_results_20261110T143605891204Z.csv`
-   Checkpoint: `evaluation/results/evaluation_checkpoint.json`
-   Summary:
    `evaluation/results/evaluation_summary_20261110T143605891204Z.json`

Abstention audit: - 5/5 unanswerable questions correctly detected. - One
answerable Gmail question was unnecessarily abstained on due to a
retrieval miss. - Unnecessary abstention: 1/45 answerable questions
(2.22%). - An earlier `abstention_accuracy=0.8` value came from a
detector bug and must not be reported.

RAGAS evaluation on 20 answerable samples, balanced at 4 per publisher,
seed 42:

  Metric                  Mean   Successful samples
  ------------------- -------- --------------------
  Answer relevancy      0.7909                20/20
  Context precision     0.9167                18/20
  Context recall        0.8750                18/20

Additional experimental local HHEM analysis: - Mean windowed HHEM score:
0.6921 - Experimental binary claim-support score: 0.7142 - 111 claims
assessed - 3 formatting-only claims and 1 document disclaimer excluded

**Important:** The custom windowed HHEM result is experimental and is
not official RAGAS Faithfulness. Do not label it as such.

Evaluation constraints: - Gemini quota was exhausted during earlier
runs; avoid unnecessary answer regeneration. - Saved answers should be
reused when methodologically valid. - Ollama Qwen 2.5 3B sometimes
produced unreliable judgments or metric timeouts. - Failed/time-out
metric values must remain missing/failed, not be silently converted to
zero. - A one-sample RAGAS pilot is not a meaningful final result.

## 9. Current work: PDF-to-Markdown normalization

The planned design is: - Original PDFs remain unchanged under
`data/raw/`. - Markdown becomes the primary RAG ingestion format under
`data/processed/`. - Each Markdown file keeps the original PDF filename
stem and publisher folder. - Original PDF filename and page number
remain available as metadata for citations. - Markdown is for extraction
and structure; the PDF remains the authoritative source.

Generated artifacts: - `processed_ready.zip`: first normalized corpus
archive. - `processed_cisco_fixed.zip`: revised Cisco conversion
archive. - The Cisco folder in `data/processed` was replaced with the
revised Cisco version.

Five Markdown samples were inspected: - AWS: `ec2_connectivity.md` -
Cisco: `lan_switching_troubleshooting.md` - Google Workspace:
`gmail_smtp_errors.md` - Microsoft: `network_troubleshooting.md` -
Ubuntu: `netplan_configuration.md`

Issues identified:

  -----------------------------------------------------------------------
  Publisher                           Findings
  ----------------------------------- -----------------------------------
  AWS                                 Headings need verification; some
                                      introductory framing may have been
                                      synthesized.

  Cisco                               Some ordinary paragraphs were
                                      incorrectly fenced as code; reading
                                      order and page-split fragments need
                                      review.

  Google Workspace                    Tables continue across pages and
                                      some source text appears clipped in
                                      the PDF itself. Do not invent
                                      missing cells.

  Microsoft                           Structure appears relatively clean.
                                      Its current source does not contain
                                      the `netsh` commands expected by
                                      one benchmark question.

  Ubuntu                              YAML indentation was lost and code
                                      blocks were fragmented across page
                                      boundaries.
  -----------------------------------------------------------------------

**Current status:** Markdown normalization has not passed quality
validation. Do not use the current Markdown corpus for a final index
rebuild until the extraction problems are addressed and checked against
the PDFs.

## 10. Next steps --- required order

### Phase 1: Finish Markdown normalization

1.  Compare the five inspected Markdown files with their corresponding
    original PDFs.
2.  Correct extraction without paraphrasing, summarizing or adding
    technical instructions.
3.  Preserve code formatting, YAML indentation, tables, reading order
    and original page boundaries.
4.  Ensure every Markdown file maps to exactly one original PDF.
5.  Apply the validated deterministic conversion method to all 30 PDFs.
6.  Run structural and content checks; record unresolved source-PDF
    defects separately.
7.  Finalize `data/processed/` only after validation.

### Phase 2: Rebuild ingestion and index safely

1.  Update ingestion to read Markdown.
2.  Preserve publisher, source PDF, page number and other useful
    metadata.
3.  Validate chunking and metadata before embedding.
4.  Build a new FAISS index in a separate directory.
5.  Check vector count, source traceability and retrieval quality.
6.  Keep the old index and backup intact until the new index passes
    validation.

### Phase 3: Improve retrieval and grounded generation

1.  Add passage-level retrieval evaluation.
2.  Investigate Gmail 452 retrieval failure.
3.  Audit benchmark questions whose expected answers are absent from the
    corpus.
4.  Resolve the Microsoft `netsh` source/benchmark mismatch without
    changing the original benchmark silently.
5.  Check citation correctness, answer support and abstention behaviour.
6.  Tune retrieval and prompts using a development set, not the held-out
    test set.

### Phase 4: Reach and validate the RAGAS target

1.  Define the exact RAGAS metrics and thresholds before tuning.
2.  Create reproducible development, validation and held-out test
    splits.
3.  Evaluate answer relevancy, faithfulness, context precision and
    context recall separately.
4.  Use a suitable, consistent evaluator model and document its
    limitations.
5.  Diagnose low-scoring samples and fix root causes rather than gaming
    metrics.
6.  Run final evaluation on the untouched held-out set.
7.  Report sample counts, means, failures, confidence/variation where
    appropriate, judge model and evaluation settings.

**Portfolio target:** greater than 0.90 (90%) on each agreed core RAGAS
metric on the final held-out evaluation set. This is an aspirational
acceptance target, not a current result. If it cannot be reached
honestly, report actual scores and explain the remaining limitations.

### Phase 5: Finalize portfolio deliverables

-   Reliable Streamlit demo
-   Architecture diagram and data-flow explanation
-   Reproducible setup and requirements
-   README with installation, configuration, usage and evaluation
-   Evaluation report with baseline-versus-final comparison
-   Example questions and grounded answers
-   Known limitations and future improvements
-   Clean GitHub repository with secrets excluded

## 11. Non-negotiable safeguards

-   Never modify original PDFs.
-   Never silently change the original 50-question benchmark.
-   Never overwrite the existing FAISS index before a new index is
    validated.
-   Never fabricate content, citations, metric values or successful test
    results.
-   Never report source-file Hit@K as passage-level retrieval accuracy.
-   Never call custom HHEM scores official RAGAS Faithfulness.
-   Never turn timeouts or missing metric values into zero.
-   Never claim the \>90% target has been achieved until a reproducible
    held-out evaluation proves it.
-   Keep API keys and `.env` files out of Git.

------------------------------------------------------------------------

**Handover point:** Continue from Phase 1 --- Markdown normalization
quality validation. The existing app, corpus, index, benchmark and
evaluation artifacts are valuable project assets. Do not restart the
project or discard prior results.
