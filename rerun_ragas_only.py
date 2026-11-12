"""Re-score only RAGAS collections metrics using saved evaluation data.

This script reuses the existing 20-question HHEM evaluation progress and
checkpoint contexts. It does not regenerate answers or rerun claim extraction/HHEM.
Run from the enterprise-knowledge-copilot project root.
"""

import asyncio
import csv
import json
from pathlib import Path

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.embeddings import HuggingFaceEmbeddings as RagasHuggingFaceEmbeddings


# --------------------------------------------------------------------
# Paths and local model configuration
# --------------------------------------------------------------------

CHECKPOINT_PATH = Path("evaluation/results/evaluation_checkpoint.json")
PROGRESS_PATH = Path("evaluation/results/final_local_evaluation_progress.json")
OUTPUT_JSON = Path("evaluation/results/final_local_evaluation.json")
OUTPUT_CSV = Path("evaluation/results/final_local_evaluation.csv")

MODEL_NAME = "qwen2.5:3b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
METRIC_TIMEOUT_SECONDS = 180
ANSWER_RELEVANCY_STRICTNESS = 1

CSV_FIELDS = [
    "publisher",
    "question",
    "source_file",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "hhem_mean_support",
    "hhem_binary_support_rate",
    "factual_claims",
    "formatting_excluded",
    "documentation_disclaimers_excluded",
]


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def mean_or_none(values):
    valid = [float(value) for value in values if is_number(value)]
    return sum(valid) / len(valid) if valid else None


def build_checkpoint_lookup(checkpoint):
    completed = checkpoint.get("completed_samples")
    if not isinstance(completed, dict):
        raise ValueError(
            "The checkpoint does not contain the expected completed_samples dictionary."
        )

    lookup = {}
    for sample in completed.values():
        question = str(sample.get("question", "")).strip()
        if question:
            lookup[question] = sample
    return lookup


def build_metrics(llm, embeddings):
    return {
        "answer_relevancy": AnswerRelevancy(
            llm=llm,
            embeddings=embeddings,
            strictness=ANSWER_RELEVANCY_STRICTNESS,
        ),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }


async def score_missing_metrics(result_record, checkpoint_sample, metrics):
    question = str(checkpoint_sample.get("question", "")).strip()
    answer = checkpoint_sample.get("answer")
    contexts = checkpoint_sample.get("retrieved_contexts")
    reference = checkpoint_sample.get("reference_answer")

    if not isinstance(contexts, list) or not contexts:
        raise ValueError(f"No retrieved_contexts in checkpoint for: {question}")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"No saved answer in checkpoint for: {question}")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(f"No reference_answer in checkpoint for: {question}")

    metric_inputs = {
        "answer_relevancy": {
            "user_input": question,
            "response": answer,
        },
        "context_precision": {
            "user_input": question,
            "retrieved_contexts": contexts,
            "reference": reference,
        },
        "context_recall": {
            "user_input": question,
            "retrieved_contexts": contexts,
            "reference": reference,
        },
    }

    scores = result_record.setdefault("ragas_metrics", {})

    for name, metric in metrics.items():
        # Keep any successfully saved score; retry only missing metrics.
        if is_number(scores.get(name)):
            print(f"  {name}: already scored; keeping {scores[name]}")
            continue

        try:
            metric_result = await asyncio.wait_for(
                metric.ascore(**metric_inputs[name]),
                timeout=METRIC_TIMEOUT_SECONDS,
            )

            value = getattr(metric_result, "value", None)
            scores[name] = float(value) if value is not None else None
            scores.pop(f"{name}_error", None)
            print(f"  {name}: {scores[name]}")

        except asyncio.TimeoutError:
            scores[name] = None
            scores[f"{name}_error"] = (
                f"Metric timed out after {METRIC_TIMEOUT_SECONDS} seconds"
            )
            print(f"  TIMEOUT: {name}")

        except Exception as error:
            scores[name] = None
            scores[f"{name}_error"] = f"{type(error).__name__}: {error}"
            print(f"  ERROR: {name}: {type(error).__name__}: {error}")

    return result_record


def save_progress(results, metadata):
    write_json(
        PROGRESS_PATH,
        {
            "evaluation_method": metadata,
            "completed_samples": results,
            "completed_count": len(results),
        },
    )


def aggregate_metrics(results):
    successful_hhem = [
        item for item in results if "experimental_hhem" in item
    ]

    aggregates = {}
    for name in ("answer_relevancy", "context_precision", "context_recall"):
        values = [
            item.get("ragas_metrics", {}).get(name)
            for item in successful_hhem
        ]
        valid = [value for value in values if is_number(value)]
        aggregates[name] = {
            "mean": mean_or_none(valid),
            "successful_samples": len(valid),
            "failed_samples": len(successful_hhem) - len(valid),
            "total_completed_samples": len(successful_hhem),
        }
    return aggregates


def save_csv(results):
    successful = [
        item for item in results if "experimental_hhem" in item
    ]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for item in successful:
            ragas_scores = item.get("ragas_metrics", {})
            hhem = item["experimental_hhem"]
            claim_counts = item["claim_counts"]

            writer.writerow({
                "publisher": item.get("publisher"),
                "question": item.get("question"),
                "source_file": item.get("source_file"),
                "answer_relevancy": ragas_scores.get("answer_relevancy"),
                "context_precision": ragas_scores.get("context_precision"),
                "context_recall": ragas_scores.get("context_recall"),
                "hhem_mean_support": hhem.get("mean_best_window_support_score"),
                "hhem_binary_support_rate": hhem.get("binary_claim_support_rate"),
                "factual_claims": claim_counts.get("factual_claims_evaluated"),
                "formatting_excluded": claim_counts.get("formatting_or_heading_excluded"),
                "documentation_disclaimers_excluded": claim_counts.get(
                    "documentation_gap_disclaimer_excluded"
                ),
            })


def update_final_output(results, metadata):
    # Preserve the HHEM summary, abstention evaluation, limitations, and
    # other fields already produced by the completed evaluation.
    output = read_json(OUTPUT_JSON)
    successful = [
        item for item in results if "experimental_hhem" in item
    ]

    output.setdefault("evaluation_method", {}).update(metadata)
    output["evaluation_method"]["ragas_scoring_api"] = (
        "RAGAS collections metric.ascore(**kwargs); MetricResult.value"
    )
    output["ragas_metrics"] = aggregate_metrics(results)
    output["per_question_results"] = results
    output["completion"] = {
        "selected_samples": 20,
        "completed_samples": len(results),
        "successful_samples": len(successful),
        "failed_samples": len(results) - len(successful),
    }

    write_json(OUTPUT_JSON, output)
    save_csv(results)


async def main():
    print("=" * 65)
    print("RAGAS-ONLY RESCORING")
    print("Reusing saved HHEM results and checkpoint contexts")
    print("=" * 65)

    checkpoint = read_json(CHECKPOINT_PATH)
    progress = read_json(PROGRESS_PATH)
    checkpoint_lookup = build_checkpoint_lookup(checkpoint)

    results = progress.get("completed_samples")
    if not isinstance(results, list) or not results:
        raise ValueError("No saved evaluation samples found in progress JSON.")

    eligible = [item for item in results if "experimental_hhem" in item]
    if not eligible:
        raise ValueError("No completed HHEM results found to rescore.")

    print(f"Saved evaluation records: {len(results)}")
    print(f"Completed HHEM records to rescore: {len(eligible)}")

    client = AsyncOpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",
        timeout=METRIC_TIMEOUT_SECONDS,
        max_retries=0,
    )

    try:
        print("\nInitializing Qwen local judge...")
        llm = llm_factory(
            model=MODEL_NAME,
            provider="openai",
            client=client,
        )

        print("Loading RAGAS local embedding model...")
        embeddings = RagasHuggingFaceEmbeddings(
            model=EMBEDDING_MODEL,
            device="cpu",
            normalize_embeddings=True,
            batch_size=32,
        )

        metrics = build_metrics(llm, embeddings)
        metadata = dict(progress.get("evaluation_method", {}))
        metadata.update({
            "judge_model": MODEL_NAME,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_provider": "RAGAS native HuggingFaceEmbeddings",
            "ragas_scoring_api": "collections metric.ascore(**kwargs)",
            "answer_relevancy_strictness": ANSWER_RELEVANCY_STRICTNESS,
            "answer_generation": "Saved Gemini answers; no regeneration",
            "hhem_rerun": False,
        })

        for index, record in enumerate(results, start=1):
            if "experimental_hhem" not in record:
                print(f"\nSkipping failed HHEM record {index}/{len(results)}")
                continue

            question = str(record.get("question", "")).strip()
            original = checkpoint_lookup.get(question)
            if original is None:
                raise ValueError(
                    f"Could not match saved result to checkpoint: {question}"
                )

            print(f"\nScoring RAGAS metrics for {index}/{len(results)}")
            print(f"Publisher: {record.get('publisher')}")
            print(f"Question: {question}")

            await score_missing_metrics(record, original, metrics)
            save_progress(results, metadata)
            print("Progress saved. HHEM results remain unchanged.")

        update_final_output(results, metadata)

        print("\n" + "=" * 65)
        print("RAGAS-ONLY RESCORING SUMMARY")
        print("=" * 65)

        aggregates = aggregate_metrics(results)
        for name, values in aggregates.items():
            print(
                f"{name}: {values['mean']} "
                f"({values['successful_samples']}/"
                f"{values['total_completed_samples']} scored)"
            )

        print("\nUpdated files:")
        print(PROGRESS_PATH)
        print(OUTPUT_JSON)
        print(OUTPUT_CSV)
        print("\nRAGAS-only rescoring finished. HHEM was not rerun.")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
