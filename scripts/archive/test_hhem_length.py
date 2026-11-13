
"""Final local evaluation for Enterprise Knowledge Copilot.

Uses saved Gemini answers and retrieved contexts.
Does not regenerate answers or modify the original benchmark/checkpoint.

HHEM faithfulness is a custom windowed experimental calculation.
The other three metrics use RAGAS metric implementations.
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
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics import (
    Faithfulness,
    FaithfulnesswithHHEM,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas import SingleTurnSample


# ============================================================
# CONFIGURATION
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

MODEL_NAME = "qwen2.5:3b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PUBLISHERS = [
    "AWS",
    "Cisco",
    "Google Workspace",
    "Microsoft",
    "Ubuntu",
]

SAMPLES_PER_PUBLISHER = 4
RANDOM_SEED = 42

# HHEM window configuration.
WINDOW_TOKENS = 300
OVERLAP_TOKENS = 50
MAX_INPUT_TOKENS = 512
SUPPORT_THRESHOLD = 0.5
HHEM_BATCH_SIZE = 4


# ============================================================
# DATA LOADING AND MATCHING
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

    samples = list(completed.values())

    with DATASET_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        dataset = list(csv.DictReader(file))

    checkpoint_lookup = {
        str(sample["question"]).strip(): sample
        for sample in samples
    }

    matched = []

    for row in dataset:
        question = str(row["question"]).strip()

        if question not in checkpoint_lookup:
            raise ValueError(
                f"Question missing from checkpoint: {question}"
            )

        sample = checkpoint_lookup[question]

        # Attach metadata from the original dataset.
        sample = dict(sample)
        sample["publisher"] = row["publisher"]
        sample["source_file"] = row.get("source_file", "")
        sample["topic"] = row.get("topic", "")

        matched.append(sample)

    if len(matched) != len(dataset):
        raise ValueError("Dataset/checkpoint mapping is incomplete.")

    return matched


def parse_bool(value):
    return str(value).strip().lower() in {
        "true", "1", "yes"
    }


def select_balanced_samples(samples):
    """Select four answerable questions per publisher."""

    rng = random.Random(RANDOM_SEED)
    selected = []

    for publisher in PUBLISHERS:
        candidates = [
            sample for sample in samples
            if sample["publisher"] == publisher
            and parse_bool(sample["answerable"])
            and sample.get("answer")
            and sample.get("reference_answer")
            and sample.get("retrieved_contexts")
        ]

        if len(candidates) < SAMPLES_PER_PUBLISHER:
            raise ValueError(
                f"Only {len(candidates)} eligible samples "
                f"for {publisher}."
            )

        # Stable candidate ordering before seeded selection.
        candidates.sort(key=lambda x: x["question"])

        chosen = rng.sample(
            candidates, SAMPLES_PER_PUBLISHER
        )

        selected.extend(chosen)

    if len(selected) != 20:
        raise ValueError("Expected exactly 20 answerable samples.")

    return selected


# ============================================================
# CLAIM CLASSIFICATION
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
    """Remove formatting while preserving factual content."""

    cleaned_lines = []

    for line in statement.splitlines():
        line = line.strip()

        if not line:
            continue

        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^[*+\-]\s*", "", line)
        line = re.sub(r"^\d+[\.)]\s*", "", line)
        line = re.sub(r"\[\d+\]", "", line)

        # Remove short leading labels such as "Instance Status:"
        # while preserving the statement after the label.
        line = re.sub(
            r"^[A-Z][A-Za-z0-9 /&()_-]{1,35}:\s+(?=[A-Z])",
            "",
            line,
        )

        # Standalone section headings are formatting, not claims.
        if line.endswith(":") and len(line) < 120:
            continue

        if line:
            cleaned_lines.append(line)

    return " ".join(cleaned_lines).strip()


def classify_claims(statements):
    factual = []
    excluded = []

    for statement in statements:
        # Split clauses so a disclaimer does not hide a factual claim.
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
# HHEM CONTEXT WINDOWS
# ============================================================

def create_context_windows(contexts, tokenizer):
    """Tokenize all contexts without encoding an overlong sequence."""

    complete_context = "\n\n".join(contexts)

    tokens = tokenizer.tokenize(complete_context)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    windows = []
    step = WINDOW_TOKENS - OVERLAP_TOKENS

    for start in range(0, len(token_ids), step):
        end = min(start + WINDOW_TOKENS, len(token_ids))

        window_ids = token_ids[start:end]

        window_text = tokenizer.decode(
            window_ids,
            skip_special_tokens=True,
        )

        if window_text.strip():
            windows.append(window_text)

        if end == len(token_ids):
            break

    return windows, len(token_ids)


def validate_actual_pairs(pairs, classifier, tokenizer):
    """Validate every real premise/claim pair before inference."""

    special_tokens = tokenizer.num_special_tokens_to_add(
        pair=False
    )

    for index, (premise, hypothesis) in enumerate(
        pairs, start=1
    ):
        formatted = classifier.prompt.format(
            text1=premise,
            text2=hypothesis,
        )

        token_count = (
            len(tokenizer.tokenize(formatted))
            + special_tokens
        )

        if token_count > MAX_INPUT_TOKENS:
            raise ValueError(
                f"HHEM input {index} is {token_count} tokens. "
                f"Maximum allowed: {MAX_INPUT_TOKENS}. "
                "Reduce WINDOW_TOKENS before continuing."
            )


def verify_claims(claims, windows, classifier, tokenizer):
    """Verify each factual claim against every evidence window."""

    results = []

    for index, claim_record in enumerate(claims, start=1):
        claim = claim_record["cleaned_statement"]

        pairs = [
            (window, claim)
            for window in windows
        ]

        # Validate actual claim/window combinations, not a placeholder.
        validate_actual_pairs(
            pairs, classifier, tokenizer
        )

        probabilities = []

        for start in range(0, len(pairs), HHEM_BATCH_SIZE):
            batch = pairs[
                start:start + HHEM_BATCH_SIZE
            ]

            scores = classifier.predict(batch)

            probabilities.extend(
                scores.cpu().detach().tolist()
            )

        if not probabilities:
            raise ValueError(
                f"No HHEM scores produced for claim {index}."
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
                float(score) for score in probabilities
            ],
        })

    return results


# ============================================================
# RAGAS METRICS
# ============================================================

async def score_ragas_metrics(
    sample, llm, embeddings, metrics
):
    """Run the three RAGAS metrics using local models."""

    ragas_sample = SingleTurnSample(
        user_input=sample["question"],
        response=sample["answer"],
        retrieved_contexts=sample["retrieved_contexts"],
        reference=sample["reference_answer"],
    )

    scores = {}

    for metric_name, metric in metrics.items():
        try:
            score = await metric.single_turn_ascore(
                ragas_sample
            )

            scores[metric_name] = (
                float(score)
                if score is not None
                else None
            )

        except Exception as error:
            scores[metric_name] = None
            scores[f"{metric_name}_error"] = str(error)

    return scores


# ============================================================
# EVALUATE ONE ANSWERABLE SAMPLE
# ============================================================

async def evaluate_sample(
    sample, llm, embeddings, metrics, hhem_classifier
):
    print("\n" + "-" * 65)
    print("Publisher:", sample["publisher"])
    print("Question:", sample["question"])

    contexts = sample["retrieved_contexts"]

    if not isinstance(contexts, list) or not contexts:
        raise ValueError("Missing retrieved contexts.")

    # Extract claims with RAGAS's statement-generation prompt.
    extractor = Faithfulness(llm=llm)

    extracted = await extractor._create_statements(
        {
            "user_input": sample["question"],
            "response": sample["answer"],
            "retrieved_contexts": contexts,
        },
        None,
    )

    factual_claims, excluded_claims = classify_claims(
        extracted.statements
    )

    if not factual_claims:
        raise ValueError("No factual claims remain.")

    tokenizer = hhem_classifier.tokenzier

    windows, total_context_tokens = create_context_windows(
        contexts, tokenizer
    )

    if not windows:
        raise ValueError("No evidence windows created.")

    claim_results = verify_claims(
        factual_claims,
        windows,
        hhem_classifier,
        tokenizer,
    )

    mean_hhem_score = statistics.mean(
        item["best_window_support_score"]
        for item in claim_results
    )

    supported_count = sum(
        item["supported_at_threshold"]
        for item in claim_results
    )

    binary_support_rate = (
        supported_count / len(claim_results)
    )

    # Run the three standard RAGAS metrics.
    ragas_scores = await score_ragas_metrics(
        sample, llm, embeddings, metrics
    )

    formatting_count = sum(
        item["category"] == "formatting_or_heading"
        for item in excluded_claims
    )

    disclaimer_count = sum(
        item["category"] == "documentation_gap_disclaimer"
        for item in excluded_claims
    )

    print("Factual claims:", len(factual_claims))
    print("Formatting/headings excluded:", formatting_count)
    print("Documentation disclaimers excluded:", disclaimer_count)
    print("Mean windowed HHEM support:", round(mean_hhem_score, 4))
    print("Binary claim support:", round(binary_support_rate, 4))
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
            "factual_claims_evaluated": len(factual_claims),
            "formatting_or_heading_excluded": formatting_count,
            "documentation_gap_disclaimer_excluded": disclaimer_count,
            "total_excluded": len(excluded_claims),
        },

        "experimental_hhem": {
            "mean_best_window_support_score": mean_hhem_score,
            "binary_claim_support_rate": binary_support_rate,
            "binary_threshold": SUPPORT_THRESHOLD,
            "claims": claim_results,
            "excluded_claims": excluded_claims,
            "context_tokens": total_context_tokens,
            "window_count": len(windows),
            "window_tokens": WINDOW_TOKENS,
            "overlap_tokens": OVERLAP_TOKENS,
        },

        "ragas_metrics": ragas_scores,
    }


# ============================================================
# UNANSWERABLE QUESTION EVALUATION
# ============================================================

def evaluate_abstentions(samples):
    """Evaluate saved abstention decisions without regenerating answers."""

    unanswerable = [
        sample for sample in samples
        if not parse_bool(sample["answerable"])
    ]

    results = []

    for sample in unanswerable:
        expected = bool(sample.get("expected_abstention"))
        actual = bool(sample.get("actual_abstention"))

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
# MAIN
# ============================================================

async def main():
    print("=" * 65)
    print("ENTERPRISE KNOWLEDGE COPILOT")
    print("FINAL LOCAL EVALUATION")
    print("=" * 65)

    all_samples = load_inputs()

    print("Matched checkpoint samples:", len(all_samples))

    selected = select_balanced_samples(all_samples)

    print("\nSelected answerable questions:")
    print(Counter(item["publisher"] for item in selected))

    print("\nInitializing local Qwen judge...")

    client = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )

    try:
        llm = llm_factory(
            model=MODEL_NAME,
            provider="openai",
            client=client,
        )

        print("Loading local embedding model...")

        embeddings = HuggingFaceEmbeddings(
            model=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        metrics = {
            "answer_relevancy": AnswerRelevancy(
                llm=llm,
                embeddings=embeddings,
            ),
            "context_precision": ContextPrecision(
                llm=llm,
            ),
            "context_recall": ContextRecall(
                llm=llm,
            ),
        }

        print("Loading HHEM on CPU...")

        hhem_metric = FaithfulnesswithHHEM(
            llm=llm,
            device="cpu",
            batch_size=HHEM_BATCH_SIZE,
        )

        hhem_classifier = hhem_metric.nli_classifier

        results = []

        for index, sample in enumerate(selected, start=1):
            print(
                f"\nEvaluating sample {index}/{len(selected)}"
            )

            result = await evaluate_sample(
                sample,
                llm,
                embeddings,
                metrics,
                hhem_classifier,
            )

            results.append(result)

        # Aggregate the 20 answerable samples.
        metric_names = [
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]

        aggregates = {}

        for metric_name in metric_names:
            values = [
                result["ragas_metrics"].get(metric_name)
                for result in results
                if isinstance(
                    result["ragas_metrics"].get(metric_name),
                    (int, float),
                )
            ]

            aggregates[metric_name] = {
                "mean": (
                    statistics.mean(values) if values else None
                ),
                "successful_samples": len(values),
                "total_samples": len(results),
            }

        hhem_values = [
            result["experimental_hhem"][
                "mean_best_window_support_score"
            ]
            for result in results
        ]

        binary_values = [
            result["experimental_hhem"][
                "binary_claim_support_rate"
            ]
            for result in results
        ]

        total_claims = sum(
            result["claim_counts"]["factual_claims_evaluated"]
            for result in results
        )

        total_formatting_excluded = sum(
            result["claim_counts"]["formatting_or_heading_excluded"]
            for result in results
        )

        total_disclaimers_excluded = sum(
            result["claim_counts"][
                "documentation_gap_disclaimer_excluded"
            ]
            for result in results
        )

        abstention_results = evaluate_abstentions(all_samples)

        output = {
            "evaluation_name": (
                "Enterprise Knowledge Copilot - Final Local Evaluation"
            ),
            "evaluation_method": {
                "answer_generation": (
                    "Previously saved Gemini answers; no regeneration"
                ),
                "judge_model": MODEL_NAME,
                "embedding_model": EMBEDDING_MODEL,
                "hhem_model": (
                    "vectara/hallucination_evaluation_model"
                ),
                "selection_method": (
                    "Four randomly selected questions per publisher"
                ),
                "random_seed": RANDOM_SEED,
                "answerable_sample_count": len(results),
                "unanswerable_sample_count": (
                    abstention_results["total_unanswerable"]
                ),
            },

            "ragas_metrics": aggregates,

            "experimental_windowed_hhem": {
                "mean_of_sample_mean_support_scores": (
                    statistics.mean(hhem_values)
                ),
                "mean_binary_claim_support_rate": (
                    statistics.mean(binary_values)
                ),
                "factual_claims_evaluated": total_claims,
                "formatting_or_heading_excluded": (
                    total_formatting_excluded
                ),
                "documentation_gap_disclaimer_excluded": (
                    total_disclaimers_excluded
                ),
                "binary_threshold": SUPPORT_THRESHOLD,
            },

            "abstention_evaluation": abstention_results,

            "per_question_results": results,

            "limitations": [
                "Windowed HHEM is a custom experimental calculation, not standard RAGAS Faithfulness.",
                "The maximum support score across windows can overestimate evidence support.",
                "Qwen 2.5 3B is a local evaluator and may make judging errors.",
                "Claim extraction and exclusion decisions require manual review.",
                "The 20-question balanced sample is not the complete 45-question answerable benchmark.",
                "Five unanswerable questions are evaluated separately.",
                "Context precision and recall depend on the quality of the reference answers.",
            ],
        }

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        OUTPUT_JSON.write_text(
            json.dumps(
                output,
                indent=4,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        # Flat CSV for easy inspection and reporting.
        with OUTPUT_CSV.open(
            "w", encoding="utf-8", newline=""
        ) as file:
            fieldnames = [
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

            writer = csv.DictWriter(
                file, fieldnames=fieldnames
            )

            writer.writeheader()

            for result in results:
                writer.writerow({
                    "publisher": result["publisher"],
                    "question": result["question"],
                    "source_file": result["source_file"],
                    "answer_relevancy": result[
                        "ragas_metrics"
                    ].get("answer_relevancy"),
                    "context_precision": result[
                        "ragas_metrics"
                    ].get("context_precision"),
                    "context_recall": result[
                        "ragas_metrics"
                    ].get("context_recall"),
                    "hhem_mean_support": result[
                        "experimental_hhem"
                    ]["mean_best_window_support_score"],
                    "hhem_binary_support_rate": result[
                        "experimental_hhem"
                    ]["binary_claim_support_rate"],
                    "factual_claims": result[
                        "claim_counts"
                    ]["factual_claims_evaluated"],
                    "formatting_excluded": result[
                        "claim_counts"
                    ]["formatting_or_heading_excluded"],
                    "documentation_disclaimers_excluded": result[
                        "claim_counts"
                    ]["documentation_gap_disclaimer_excluded"],
                })

        print("\n" + "=" * 65)
        print("FINAL EVALUATION SUMMARY")
        print("=" * 65)

        print("\nRAGAS metrics:")
        for name, values in aggregates.items():
            print(
                name,
                values["mean"],
                f"({values['successful_samples']}/"
                f"{values['total_samples']} scored)",
            )

        print("\nExperimental HHEM:")
        print(
            "Mean support:",
            round(statistics.mean(hhem_values), 4),
        )
        print(
            "Binary claim support:",
            round(statistics.mean(binary_values), 4),
        )

        print("\nClaim counts:")
        print("Factual claims:", total_claims)
        print("Formatting excluded:", total_formatting_excluded)
        print("Documentation disclaimers excluded:", total_disclaimers_excluded)

        print("\nAbstention:")
        print(
            abstention_results["correct_abstentions"],
            "/",
            abstention_results["total_unanswerable"],
        )

        print("\nSaved JSON:", OUTPUT_JSON)
        print("Saved CSV:", OUTPUT_CSV)

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
