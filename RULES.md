# Project Rules

1. `data/raw` (source PDFs), the original benchmark dataset, saved baseline outputs, and existing FAISS indexes are never modified, overwritten, or deleted. New indexes are built in new directories.
2. Back up before any risky change. Make small, single-purpose commits. No pushing without explicit approval.
3. Never invent technical content, citations, page numbers, benchmark labels, or evaluation values. Never claim a test passed without showing real command output.
4. Missing, failed, or timed-out evaluation results stay missing/failed. They are never replaced with zeros and never silently dropped.
5. Never relabel experimental metrics as official ones (for example, custom HHEM analysis is not RAGAS Faithfulness).
6. Do not tune on the held-out test set. Tuning uses a development set only.
7. API keys stay in `.env`, which is never committed, printed, or logged.
8. Avoid unnecessary Gemini calls; reuse cached outputs where methodologically valid.
9. Do not report existing source-level Hit@K as answer-passage accuracy.
10. Ask before deleting anything; archive instead of deleting.
