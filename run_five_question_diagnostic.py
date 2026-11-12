"""Run a focused local RAGAS/HHEM diagnostic on five saved evaluation cases.

Reuses saved answers and retrieved contexts from evaluation_checkpoint.json.
Does NOT call Gemini, regenerate answers, or modify the original benchmark/results.
Requires the same project virtual environment and local Ollama model as
run_final_evaluation.py.
"""

import asyncio
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from datetime import datetime, timezone

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics import FaithfulnesswithHHEM
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.embeddings import HuggingFaceEmbeddings as RagasHuggingFaceEmbeddings

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "evaluation" / "results" / "evaluation_checkpoint.json"
PREVIOUS_RESULT = ROOT / "evaluation" / "results" / "final_local_evaluation.json"
OUTPUT = ROOT / "evaluation" / "results" / "five_question_diagnostic.json"
CSV_OUTPUT = ROOT / "evaluation" / "results" / "five_question_diagnostic.csv"
BASE_SCRIPT = ROOT / "run_final_evaluation.py"

MODEL_NAME = "qwen2.5:3b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TIMEOUT_SECONDS = 180
STRICTNESS = 1

TARGET_QUESTIONS = [
    "What does the Google Workspace SMTP relay error 'Mail relay denied' indicate?",
    "What does Gmail SMTP error 452 4.2.2 indicate?",
    "Which commands does Microsoft recommend for resetting network settings and refreshing DNS on Windows?",
    "Which protocol does the Cisco LAN switching guide describe as a preferred way to negotiate EtherChannel automatically?",
    "What can an administrator configure with Google Workspace routing settings?",
]


def load_base_module():
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Could not find {BASE_SCRIPT}. Keep this diagnostic script beside run_final_evaluation.py."
        )
    spec = importlib.util.spec_from_file_location("base_final_evaluation", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_five_samples():
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    completed = checkpoint["completed_samples"]
    by_question = {v["question"].strip(): v for v in completed.values()}

    previous = json.loads(PREVIOUS_RESULT.read_text(encoding="utf-8"))
    metadata = {v["question"].strip(): v for v in previous["per_question_results"]}

    selected = []
    missing = []
    for question in TARGET_QUESTIONS:
        saved = by_question.get(question)
        prior = metadata.get(question)
        if saved is None or prior is None:
            missing.append(question)
            continue
        if not saved.get("answer") or not saved.get("retrieved_contexts"):
            raise ValueError(f"Saved answer/contexts missing for: {question}")
        sample = dict(saved)
        sample["publisher"] = prior.get("publisher", "")
        sample["source_file"] = prior.get("source_file", "")
        sample["topic"] = prior.get("topic", "")
        selected.append(sample)

    if missing:
        raise ValueError("Questions missing from saved files:\n- " + "\n- ".join(missing))
    if len(selected) != 5:
        raise ValueError(f"Expected 5 questions, found {len(selected)}")
    return selected


async def main():
    print("=" * 70)
    print("FIVE-QUESTION LOCAL EVALUATION DIAGNOSTIC")
    print("Saved answers/contexts only — no Gemini regeneration")
    print("=" * 70)

    base = load_base_module()
    samples = load_five_samples()

    print("Loaded five saved samples:")
    for i, sample in enumerate(samples, 1):
        print(f"{i}. [{sample['publisher']}] {sample['question']}")

    client = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        timeout=TIMEOUT_SECONDS,
        max_retries=0,
    )

    results = []
    try:
        print("\nInitializing local Qwen judge...")
        llm = llm_factory(model=MODEL_NAME, provider="openai", client=client)

        print("Loading local RAGAS embeddings...")
        embeddings = RagasHuggingFaceEmbeddings(
            model=EMBEDDING_MODEL,
            device="cpu",
            normalize_embeddings=True,
            batch_size=32,
        )

        metrics = {
            "answer_relevancy": AnswerRelevancy(
                llm=llm,
                embeddings=embeddings,
                strictness=STRICTNESS,
            ),
            "context_precision": ContextPrecision(llm=llm),
            "context_recall": ContextRecall(llm=llm),
        }

        print("Loading HHEM on CPU...")
        hhem_metric = FaithfulnesswithHHEM(llm=llm, device="cpu", batch_size=2)
        hhem_classifier = hhem_metric.nli_classifier
        print("Models ready. Starting five-question rerun.\n")

        for index, sample in enumerate(samples, 1):
            print("\n" + "#" * 70)
            print(f"QUESTION {index}/5 — {sample['publisher']}")
            print(sample["question"])
            try:
                result = await base.evaluate_sample(
                    sample=sample,
                    llm=llm,
                    metrics=metrics,
                    hhem_classifier=hhem_classifier,
                )
                result["rerun_status"] = "completed"
            except Exception as exc:
                result = {
                    "question": sample["question"],
                    "publisher": sample["publisher"],
                    "source_file": sample["source_file"],
                    "rerun_status": "sample_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print("SAMPLE ERROR:", result["error"])

            results.append(result)
            payload = {
                "evaluation_name": "Five-question local diagnostic",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "method": {
                    "judge": MODEL_NAME,
                    "embedding_model": EMBEDDING_MODEL,
                    "answer_generation": "reused saved Gemini answers; no regeneration",
                    "contexts": "reused saved retrieved contexts; no retrieval rerun",
                    "metrics": ["RAGAS AnswerRelevancy", "RAGAS ContextPrecision", "RAGAS ContextRecall", "experimental windowed HHEM claim support"],
                    "timeout_seconds_per_metric": TIMEOUT_SECONDS,
                    "answer_relevancy_strictness": STRICTNESS,
                },
                "completed_count": len(results),
                "total_questions": 5,
                "results": results,
            }
            OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Saved diagnostic progress: {len(results)}/5")

        # A compact comparison table is emitted to CSV.
        import csv
        fields = ["publisher", "question", "rerun_status", "answer_relevancy", "context_precision", "context_recall", "hhem_mean_support", "hhem_binary_support_rate", "error"]
        with CSV_OUTPUT.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for item in results:
                ragas = item.get("ragas_metrics", {})
                hhem = item.get("experimental_hhem", {})
                writer.writerow({
                    "publisher": item.get("publisher"),
                    "question": item.get("question"),
                    "rerun_status": item.get("rerun_status"),
                    "answer_relevancy": ragas.get("answer_relevancy"),
                    "context_precision": ragas.get("context_precision"),
                    "context_recall": ragas.get("context_recall"),
                    "hhem_mean_support": hhem.get("mean_best_window_support_score"),
                    "hhem_binary_support_rate": hhem.get("binary_claim_support_rate"),
                    "error": item.get("error"),
                })

        print("\n" + "=" * 70)
        print("DIAGNOSTIC COMPLETE")
        print("JSON:", OUTPUT)
        print("CSV:", CSV_OUTPUT)
        print("Original 50-question benchmark and final evaluation files were not modified.")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
