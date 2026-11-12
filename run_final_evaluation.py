"""
Enterprise Knowledge Copilot - Final Local Evaluation

Uses:
- Saved Gemini answers and retrieved contexts
- Qwen 2.5 3B through local Ollama
- Local HuggingFace embeddings
- RAGAS 0.4.3
- Vectara HHEM

No Gemini answer regeneration.

IMPORTANT:
The custom windowed HHEM calculation is an experimental
claim-support measure, NOT standard RAGAS Faithfulness.
"""

import asyncio
import csv
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path

from openai import AsyncOpenAI

from ragas import SingleTurnSample
from ragas.llms import llm_factory

# Retain these legacy imports because this script uses
# Faithfulness._create_statements for claim extraction and
# the installed HHEM class is exposed through this namespace.
from ragas.metrics import Faithfulness, FaithfulnesswithHHEM

from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

# RAGAS 0.4.3 native modern embedding provider.
from ragas.embeddings import (
    HuggingFaceEmbeddings as RagasHuggingFaceEmbeddings
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

CHECKPOINT_PATH = Path(
    "evaluation/results/evaluation_checkpoint.json"
)

DATASET_PATH = Path(
    "evaluation/datasets/evaluation_dataset_test.csv"
)

OUTPUT_DIR = Path("evaluation/results")

OUTPUT_JSON = OUTPUT_DIR / "final_local_evaluation.json"
OUTPUT_CSV = OUTPUT_DIR / "final_local_evaluation.csv"
PROGRESS_JSON = OUTPUT_DIR / "final_local_evaluation_progress.json"

MODEL_NAME = "qwen2.5:3b"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

PUBLISHERS = [
    "AWS",
    "Cisco",
    "Google Workspace",
    "Microsoft",
    "Ubuntu",
]

SAMPLES_PER_PUBLISHER = 4
RANDOM_SEED = 42

# HHEM configuration
WINDOW_TOKENS = 250
OVERLAP_TOKENS = 40
MAX_INPUT_TOKENS = 512
SUPPORT_THRESHOLD = 0.5
HHEM_BATCH_SIZE = 2

# Local Qwen timeout per metric.
METRIC_TIMEOUT_SECONDS = 180

# Answer relevancy uses one generated question
# to reduce local judge workload.
ANSWER_RELEVANCY_STRICTNESS = 1


# ============================================================
# 2. UTILITIES
# ============================================================

def parse_bool(value):
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    return str(value).strip().lower() in {
        "true", "1", "yes"
    }


def safe_mean(values):
    valid = [
        float(value)
        for value in values
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
    ]

    return statistics.mean(valid) if valid else None


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            data,
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ============================================================
# 3. LOAD ORIGINAL DATA AND CHECKPOINT
# ============================================================

def load_inputs():

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    checkpoint = json.loads(
        CHECKPOINT_PATH.read_text(encoding="utf-8")
    )

    completed = checkpoint["completed_samples"]

    with DATASET_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        dataset = list(csv.DictReader(file))

    lookup = {
        str(sample["question"]).strip(): sample
        for sample in completed.values()
    }

    samples = []

    for row in dataset:

        question = str(row["question"]).strip()

        if question not in lookup:
            raise ValueError(
                f"Checkpoint question not found: {question}"
            )

        sample = dict(lookup[question])

        sample["publisher"] = row["publisher"]
        sample["source_file"] = row.get("source_file", "")
        sample["topic"] = row.get("topic", "")

        samples.append(sample)

    if len(samples) != len(dataset):
        raise ValueError("Dataset/checkpoint count mismatch.")

    print("Dataset questions:", len(dataset))
    print("Matched checkpoint samples:", len(samples))

    return samples


# ============================================================
# 4. SELECT 20 BALANCED ANSWERABLE QUESTIONS
# ============================================================

def select_balanced_samples(samples):

    rng = random.Random(RANDOM_SEED)
    selected = []

    for publisher in PUBLISHERS:

        candidates = [
            sample
            for sample in samples
            if sample["publisher"] == publisher
            and parse_bool(sample.get("answerable"))
            and sample.get("answer")
            and sample.get("reference_answer")
            and isinstance(sample.get("retrieved_contexts"), list)
            and sample["retrieved_contexts"]
        ]

        if len(candidates) < SAMPLES_PER_PUBLISHER:
            raise ValueError(
                f"Insufficient eligible samples for {publisher}"
            )

        candidates.sort(
            key=lambda item: item["question"]
        )

        selected.extend(
            rng.sample(candidates, SAMPLES_PER_PUBLISHER)
        )

    distribution = Counter(
        item["publisher"] for item in selected
    )

    if len(selected) != 20:
        raise ValueError("Expected 20 answerable samples.")

    for publisher in PUBLISHERS:
        if distribution[publisher] != 4:
            raise ValueError(
                f"Unbalanced selection: {publisher}"
            )

    print("\nSelected questions:")

    for publisher in PUBLISHERS:
        print(f"{publisher}: {distribution[publisher]}")

    return selected


# ============================================================
# 5. CLAIM CLASSIFICATION
# ============================================================

DOCUMENTATION_GAP_PATTERNS = [
    r"not detailed in the available documentation",
    r"not specified in the available documentation",
    r"not covered in the available documentation",
    r"not mentioned in the available documentation",
    r"not included in the available documentation",
    r"documents? do not contain",
    r"documentation does not contain",
    r"information is not available",
    r"insufficient information",
    r"not provided in the documents",
    r"not described in the documents",
    r"cannot be determined from the documents",
    r"not available in the provided context",
    r"not covered in the provided excerpts",
]


def clean_statement(statement):

    lines = []

    for line in statement.splitlines():

        line = line.strip()

        if not line:
            continue

        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^[*+\-]\s*", "", line)
        line = re.sub(r"^\d+[\.)]\s*", "", line)
        line = re.sub(r"\[\d+\]", "", line)

        # Remove short labels without removing the factual sentence.
        line = re.sub(
            r"^[A-Z][A-Za-z0-9 /&()_-]{1,35}:\s+(?=[A-Z])",
            "",
            line,
        )

        if line.endswith(":") and len(line) < 120:
            continue

        if line:
            lines.append(line)

    return " ".join(lines).strip()


def classify_claims(statements):

    factual = []
    excluded = []

    for statement in statements:

        parts = [
            part.strip()
            for part in statement.split(";")
            if part.strip()
        ]

        for part in parts:

            cleaned = clean_statement(part)

            if not cleaned:
                category = "formatting_or_heading"

            elif any(
                re.search(pattern, cleaned, re.IGNORECASE)
                for pattern in DOCUMENTATION_GAP_PATTERNS
            ):
                category = "documentation_gap_disclaimer"

            else:
                category = "factual_claim"

            record = {
                "original_statement": part,
                "cleaned_statement": cleaned,
                "category": category,
            }

            if category == "factual_claim":
                factual.append(record)
            else:
                excluded.append(record)

    return factual, excluded


# ============================================================
# 6. CLAIM EXTRACTION WITH QWEN
# ============================================================

async def extract_claims(question, answer, llm):

    metric = Faithfulness(llm=llm)

    row = {
        "user_input": question,
        "response": answer,
        "retrieved_contexts": [],
    }

    extracted = await metric._create_statements(row, None)

    return extracted.statements


# ============================================================
# 7. HHEM CONTEXT WINDOWS
# ============================================================

def create_context_windows(contexts, tokenizer):

    complete_context = "\n\n".join(contexts)

    tokens = tokenizer.tokenize(complete_context)

    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    windows = []

    step = WINDOW_TOKENS - OVERLAP_TOKENS

    for start in range(0, len(token_ids), step):

        end = min(
            start + WINDOW_TOKENS,
            len(token_ids),
        )

        window_ids = token_ids[start:end]

        text = tokenizer.decode(
            window_ids,
            skip_special_tokens=True,
        )

        if text.strip():
            windows.append(text)

        if end == len(token_ids):
            break

    return windows, len(token_ids)


# ============================================================
# 8. VALIDATE REAL HHEM INPUTS
# ============================================================

def validate_actual_pairs(pairs, classifier, tokenizer):

    for index, (premise, hypothesis) in enumerate(
        pairs,
        start=1,
    ):

        formatted = classifier.prompt.format(
            text1=premise,
            text2=hypothesis,
        )

        # Match the actual classifier tokenization settings.
        encoded = tokenizer(
            formatted,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            verbose=False,
        )

        token_count = len(encoded["input_ids"])

        if token_count > MAX_INPUT_TOKENS:

            raise ValueError(
                f"HHEM input {index}: {token_count} tokens. "
                f"Maximum: {MAX_INPUT_TOKENS}. "
                "Reduce WINDOW_TOKENS."
            )


# ============================================================
# 9. VERIFY CLAIMS WITH HHEM
# ============================================================

def verify_claims(claims, windows, classifier, tokenizer):

    results = []

    for index, record in enumerate(claims, start=1):

        claim = record["cleaned_statement"]

        pairs = [
            (window, claim)
            for window in windows
        ]

        validate_actual_pairs(
            pairs,
            classifier,
            tokenizer,
        )

        probabilities = []

        for start in range(
            0,
            len(pairs),
            HHEM_BATCH_SIZE,
        ):

            batch = pairs[
                start:start + HHEM_BATCH_SIZE
            ]

            scores = classifier.predict(batch)

            probabilities.extend(
                scores.cpu().detach().tolist()
            )

        if not probabilities:
            raise ValueError(
                f"No HHEM scores for claim {index}"
            )

        best_score = max(probabilities)

        results.append({
            "claim_number": index,
            "claim": claim,
            "best_window_support_score": float(best_score),
            "supported_at_threshold": (
                best_score >= SUPPORT_THRESHOLD
            ),
            "all_window_scores": [
                float(value) for value in probabilities
            ],
        })

    return results


# ============================================================
# 10. STANDARD RAGAS METRICS
# ============================================================

async def score_ragas_metrics(sample, metrics):

    metric_inputs = {
        "answer_relevancy": {
            "user_input": sample["question"],
            "response": sample["answer"],
        },

        "context_precision": {
            "user_input": sample["question"],
            "retrieved_contexts": sample["retrieved_contexts"],
            "reference": sample["reference_answer"],
        },

        "context_recall": {
            "user_input": sample["question"],
            "retrieved_contexts": sample["retrieved_contexts"],
            "reference": sample["reference_answer"],
        },
    }

    scores = {}

    for name, metric in metrics.items():

        try:
            result = await asyncio.wait_for(
                metric.ascore(**metric_inputs[name]),
                timeout=METRIC_TIMEOUT_SECONDS,
            )

            scores[name] = (
                float(result.value)
                if result.value is not None
                else None
            )

            print(f"{name}: {scores[name]}")

        except asyncio.TimeoutError:

            print(
                f"TIMEOUT: {name} exceeded "
                f"{METRIC_TIMEOUT_SECONDS} seconds."
            )

            scores[name] = None
            scores[f"{name}_error"] = "Metric timeout"

        except Exception as error:

            print(
                f"ERROR: {name}: "
                f"{type(error).__name__}: {error}"
            )

            scores[name] = None
            scores[f"{name}_error"] = (
                f"{type(error).__name__}: {error}"
            )

    return scores


# ============================================================
# 11. EVALUATE ONE SAMPLE
# ============================================================

async def evaluate_sample(
    sample,
    llm,
    metrics,
    hhem_classifier,
):

    print("\n" + "=" * 65)
    print("Publisher:", sample["publisher"])
    print("Question:", sample["question"])

    contexts = sample["retrieved_contexts"]

    if not isinstance(contexts, list) or not contexts:
        raise ValueError("Missing retrieved contexts.")

    statements = await extract_claims(
        sample["question"],
        sample["answer"],
        llm,
    )

    print("Extracted statements:", len(statements))

    factual, excluded = classify_claims(statements)

    if not factual:
        raise ValueError("No factual claims remain.")

    tokenizer = hhem_classifier.tokenzier

    windows, total_tokens = create_context_windows(
        contexts,
        tokenizer,
    )

    if not windows:
        raise ValueError("No HHEM windows generated.")

    claim_results = verify_claims(
        factual,
        windows,
        hhem_classifier,
        tokenizer,
    )

    mean_support = statistics.mean(
        item["best_window_support_score"]
        for item in claim_results
    )

    binary_support = (
        sum(
            item["supported_at_threshold"]
            for item in claim_results
        )
        / len(claim_results)
    )

    ragas_scores = await score_ragas_metrics(
        sample,
        metrics,
    )

    formatting_count = sum(
        item["category"] == "formatting_or_heading"
        for item in excluded
    )

    disclaimer_count = sum(
        item["category"] == "documentation_gap_disclaimer"
        for item in excluded
    )

    print("Factual claims:", len(factual))
    print("Formatting excluded:", formatting_count)
    print("Documentation disclaimers excluded:", disclaimer_count)
    print("Mean experimental HHEM support:", round(mean_support, 4))
    print("Binary claim support:", round(binary_support, 4))
    print("RAGAS metrics:", ragas_scores)

    return {
        "question": sample["question"],
        "publisher": sample["publisher"],
        "source_file": sample["source_file"],
        "topic": sample["topic"],

        "answer": sample["answer"],
        "reference_answer": sample["reference_answer"],

        "retrieved_context_count": len(contexts),
        "retrieved_sources": sample.get("retrieved_sources", []),

        "claim_counts": {
            "factual_claims_evaluated": len(factual),
            "formatting_or_heading_excluded": formatting_count,
            "documentation_gap_disclaimer_excluded": disclaimer_count,
            "total_excluded": len(excluded),
        },

        "experimental_hhem": {
            "mean_best_window_support_score": mean_support,
            "binary_claim_support_rate": binary_support,
            "binary_threshold": SUPPORT_THRESHOLD,
            "claims": claim_results,
            "excluded_claims": excluded,
            "context_tokens": total_tokens,
            "window_count": len(windows),
            "window_tokens": WINDOW_TOKENS,
            "overlap_tokens": OVERLAP_TOKENS,
        },

        "ragas_metrics": ragas_scores,
    }


# ============================================================
# 12. UNANSWERABLE QUESTION EVALUATION
# ============================================================

def evaluate_abstentions(samples):

    unanswerable = [
        sample
        for sample in samples
        if not parse_bool(sample.get("answerable"))
    ]

    results = []

    for sample in unanswerable:

        expected = parse_bool(
            sample.get("expected_abstention")
        )

        actual = parse_bool(
            sample.get("actual_abstention")
        )

        results.append({
            "question": sample["question"],
            "expected_abstention": expected,
            "actual_abstention": actual,
            "correct": expected == actual,
        })

    correct = sum(item["correct"] for item in results)

    return {
        "total_unanswerable": len(results),
        "correct_abstentions": correct,
        "abstention_accuracy": (
            correct / len(results) if results else None
        ),
        "results": results,
    }


# ============================================================
# 13. SAVE PROGRESS AND CSV
# ============================================================

def save_progress(results, metadata):

    save_json(
        PROGRESS_JSON,
        {
            "evaluation_method": metadata,
            "completed_samples": results,
            "completed_count": len(results),
        },
    )


def save_csv(results):

    fields = [
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

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for result in results:

            writer.writerow({
                "publisher": result["publisher"],
                "question": result["question"],
                "source_file": result["source_file"],

                "answer_relevancy": result["ragas_metrics"].get(
                    "answer_relevancy"
                ),

                "context_precision": result["ragas_metrics"].get(
                    "context_precision"
                ),

                "context_recall": result["ragas_metrics"].get(
                    "context_recall"
                ),

                "hhem_mean_support": result["experimental_hhem"][
                    "mean_best_window_support_score"
                ],

                "hhem_binary_support_rate": result[
                    "experimental_hhem"
                ]["binary_claim_support_rate"],

                "factual_claims": result["claim_counts"][
                    "factual_claims_evaluated"
                ],

                "formatting_excluded": result["claim_counts"][
                    "formatting_or_heading_excluded"
                ],

                "documentation_disclaimers_excluded": result[
                    "claim_counts"
                ]["documentation_gap_disclaimer_excluded"],
            })


# ============================================================
# 14. MAIN
# ============================================================

async def main():

    print("=" * 65)
    print("ENTERPRISE KNOWLEDGE COPILOT")
    print("FINAL LOCAL EVALUATION")
    print("=" * 65)

    all_samples = load_inputs()

    selected = select_balanced_samples(all_samples)

    client = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
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

        # ----------------------------------------------------
        # MODERN RAGAS EMBEDDING INITIALIZATION
        # ----------------------------------------------------

        print("Loading local RAGAS embedding model...")

        embeddings = RagasHuggingFaceEmbeddings(
            model=EMBEDDING_MODEL,
            device="cpu",
            normalize_embeddings=True,
            batch_size=32,
        )

        print("Local RAGAS embedding model loaded successfully.")

        # ----------------------------------------------------
        # RAGAS METRICS
        # ----------------------------------------------------

        metrics = {
            "answer_relevancy": AnswerRelevancy(
                llm=llm,
                embeddings=embeddings,
                strictness=ANSWER_RELEVANCY_STRICTNESS,
            ),

            "context_precision": ContextPrecision(
                llm=llm,
            ),

            "context_recall": ContextRecall(
                llm=llm,
            ),
        }

        # ----------------------------------------------------
        # HHEM INITIALIZATION
        # ----------------------------------------------------

        print("\nLoading HHEM on CPU...")

        hhem_metric = FaithfulnesswithHHEM(
            llm=llm,
            device="cpu",
            batch_size=HHEM_BATCH_SIZE,
        )

        hhem_classifier = hhem_metric.nli_classifier

        print("HHEM loaded.")

        # ----------------------------------------------------
        # EVALUATION METADATA
        # ----------------------------------------------------

        metadata = {
            "judge_model": MODEL_NAME,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_provider": "RAGAS native HuggingFaceEmbeddings",
            "embedding_device": "cpu",
            "embedding_interface": "modern RAGAS interface",
            "answerable_sample_count": 20,
            "random_seed": RANDOM_SEED,
            "selection": "4 questions per publisher",
            "answer_generation": "Saved Gemini answers; no regeneration",
        }

        results = []

        # ----------------------------------------------------
        # EVALUATE THE 20 SELECTED QUESTIONS
        # ----------------------------------------------------

        for index, sample in enumerate(selected, start=1):

            print(f"\n\nEvaluating sample {index}/20")

            try:

                result = await evaluate_sample(
                    sample,
                    llm,
                    metrics,
                    hhem_classifier,
                )

                results.append(result)

            except Exception as error:

                # Preserve the failure instead of silently dropping it.
                print(
                    f"\nSAMPLE FAILED: {type(error).__name__}: {error}"
                )

                results.append({
                    "question": sample["question"],
                    "publisher": sample["publisher"],
                    "source_file": sample["source_file"],
                    "evaluation_error": (
                        f"{type(error).__name__}: {error}"
                    ),
                })

            # Save after every sample.
            save_progress(results, metadata)

            # Keep the CSV synchronized with completed samples.
            successful_results = [
                item for item in results
                if "experimental_hhem" in item
            ]

            save_csv(successful_results)

            print(
                f"Progress saved: {len(results)}/20"
            )

        # ----------------------------------------------------
        # COLLECT SUCCESSFUL SAMPLES
        # ----------------------------------------------------

        successful = [
            item for item in results
            if "experimental_hhem" in item
        ]

        # ----------------------------------------------------
        # STANDARD RAGAS AGGREGATION
        # ----------------------------------------------------

        aggregates = {}

        for metric_name in [
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]:

            values = [
                item["ragas_metrics"].get(metric_name)
                for item in successful
            ]

            valid = [
                value for value in values
                if isinstance(value, (int, float))
            ]

            aggregates[metric_name] = {
                "mean": safe_mean(valid),
                "successful_samples": len(valid),
                "failed_samples": len(successful) - len(valid),
                "total_completed_samples": len(successful),
            }

        # ----------------------------------------------------
        # AGGREGATE EXPERIMENTAL HHEM RESULTS
        # ----------------------------------------------------

        hhem_means = [
            item["experimental_hhem"][
                "mean_best_window_support_score"
            ]
            for item in successful
        ]

        binary_rates = [
            item["experimental_hhem"][
                "binary_claim_support_rate"
            ]
            for item in successful
        ]

        total_claims = sum(
            item["claim_counts"]["factual_claims_evaluated"]
            for item in successful
        )

        total_formatting = sum(
            item["claim_counts"]["formatting_or_heading_excluded"]
            for item in successful
        )

        total_disclaimers = sum(
            item["claim_counts"][
                "documentation_gap_disclaimer_excluded"
            ]
            for item in successful
        )

        # ----------------------------------------------------
        # EVALUATE THE FIVE UNANSWERABLE QUESTIONS
        # ----------------------------------------------------

        abstention_results = evaluate_abstentions(all_samples)

        # ----------------------------------------------------
        # FINAL OUTPUT
        # ----------------------------------------------------

        output = {
            "evaluation_name": (
                "Enterprise Knowledge Copilot - Final Local Evaluation"
            ),

            "evaluation_method": metadata,

            "completion": {
                "selected_samples": 20,
                "completed_samples": len(results),
                "successful_samples": len(successful),
                "failed_samples": len(results) - len(successful),
            },

            "ragas_metrics": aggregates,

            "experimental_windowed_hhem": {
                "mean_of_sample_mean_support_scores": safe_mean(
                    hhem_means
                ),

                "mean_binary_claim_support_rate": safe_mean(
                    binary_rates
                ),

                "factual_claims_evaluated": total_claims,
                "formatting_or_heading_excluded": total_formatting,
                "documentation_gap_disclaimer_excluded": total_disclaimers,
                "binary_threshold": SUPPORT_THRESHOLD,
            },

            "abstention_evaluation": abstention_results,

            "per_question_results": results,

            "limitations": [
                "Windowed HHEM is a custom experimental calculation, not standard RAGAS Faithfulness.",
                "Maximum support across windows can overestimate evidence support.",
                "Qwen 2.5 3B may make evaluation errors.",
                "Claim extraction and classification require manual review.",
                "The 20-question sample is balanced but does not represent the complete 45 answerable questions.",
                "The five unanswerable questions are evaluated separately.",
                "Context precision and recall depend on reference-answer quality.",
                "Metric timeouts or failures are reported as missing scores, not zero.",
            ],
        }

        save_json(OUTPUT_JSON, output)
        save_csv(successful)

        # ----------------------------------------------------
        # PRINT FINAL SUMMARY
        # ----------------------------------------------------

        print("\n" + "=" * 65)
        print("FINAL EVALUATION SUMMARY")
        print("=" * 65)

        print(
            "Completed samples:",
            len(successful),
            "/20",
        )

        print("\nStandard RAGAS metrics:")

        for name, values in aggregates.items():

            print(
                f"{name}: {values['mean']} "
                f"({values['successful_samples']}/"
                f"{values['total_completed_samples']} scored)"
            )

        print("\nExperimental HHEM:")
        print("Mean support:", safe_mean(hhem_means))
        print("Binary claim support:", safe_mean(binary_rates))

        print("\nClaim statistics:")
        print("Factual claims:", total_claims)
        print("Formatting excluded:", total_formatting)
        print("Documentation disclaimers excluded:", total_disclaimers)

        print("\nAbstention:")
        print(
            abstention_results["correct_abstentions"],
            "/",
            abstention_results["total_unanswerable"],
        )

        print("\nSaved files:")
        print(OUTPUT_JSON)
        print(OUTPUT_CSV)
        print(PROGRESS_JSON)

        print("\nEvaluation finished.")

    finally:
        await client.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())