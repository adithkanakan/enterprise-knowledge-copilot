"""Utilities for evaluating and resuming the RAG pipeline."""

import asyncio
import hashlib
import json
import math
import re
import time

from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
# from ragas.metrics import 
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    FaithfulnesswithHHEM
)

from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    EMBEDDING_MODEL,
    FAISS_INDEX_DIR,
)

from src.generation import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    generate_answer,
)

from src.retrieval import (
    load_vector_store,
    retrieve_documents,
    retrieve_documents_rrf,
)


RAGAS_VERSION_REQUIRED = "0.4.3"
RETRIEVAL_K = 4


# --------------------------------------------------
# Dataset helpers
# --------------------------------------------------

def _parse_answerable(value) -> bool:
    """Convert a CSV answerable value into a boolean."""

    if isinstance(value, bool):
        return value

    normalized_value = str(value).strip().lower()

    if normalized_value in {"true", "1", "yes"}:
        return True

    if normalized_value in {"false", "0", "no"}:
        return False

    raise ValueError(
        f"Invalid answerable value: {value}. "
        "Use true/false, yes/no, or 1/0."
    )


# --------------------------------------------------
# Abstention detection
# --------------------------------------------------

def _is_abstention(answer: str) -> bool:
    """
    Detect whether the response primarily abstains because
    the requested information is unavailable in the evidence.
    """

    if not isinstance(answer, str):
        return False

    normalized = " ".join(answer.strip().lower().split())

    if not normalized:
        return False

    # Examine the opening statement rather than the entire answer.
    opening = re.split(r"(?<=[.!?])\s+", normalized, maxsplit=1)[0]

    if len(opening) > 400:
        opening = opening[:400]

    abstention_patterns = [
        r"\b(?:information|details|answer)\b.{0,200}"
        r"\b(?:is|are)\s+(?:unavailable|not available|not provided|not found)\b",

        r"\b(?:the requested information|the requested details|"
        r"the specific information|the specific option)\b.{0,200}"
        r"\b(?:is|are)\s+(?:unavailable|not available|not specified|"
        r"not explicitly detailed|not provided)\b",

        r"\b(?:cannot determine|cannot answer|unable to determine)\b"
        r".{0,150}\b(?:provided|supplied|available)\b",

        r"\b(?:there is|there are)\s+insufficient information\b",
        r"\bnot enough information\b",
        r"\binsufficient evidence\b",
        r"^based on\b.{0,250}\b.{1,120}\b(?:is|are)\s+"
        r"(?:unavailable|not available|not provided|not specified|not found)\b",
    ]

    return any(
        re.search(pattern, opening)
        for pattern in abstention_patterns
    )


# --------------------------------------------------
# Evaluation sample selection
# --------------------------------------------------

def _select_evaluation_samples(
    dataset: pd.DataFrame,
    max_questions: int | None,
) -> pd.DataFrame:
    """
    Select a balanced evaluation sample across publishers.

    When max_questions is an integer:
    - Answerable questions are selected in round-robin order.
    - One unanswerable question is included when available.

    When max_questions is None:
    - The complete dataset is selected.
    """

    if max_questions is None:
        return dataset.reset_index(drop=True).copy()

    if (
        isinstance(max_questions, bool)
        or not isinstance(max_questions, int)
        or max_questions <= 0
    ):
        raise ValueError(
            "max_questions must be a positive integer or None."
        )

    answerable_mask = dataset["answerable"].apply(
        _parse_answerable
    )

    answerable_df = dataset[answerable_mask].copy()
    unanswerable_df = dataset[~answerable_mask].copy()

    include_unanswerable = not unanswerable_df.empty

    answerable_limit = (
        max_questions - 1
        if include_unanswerable
        else max_questions
    )

    publisher_groups = [
        group.reset_index(drop=True)
        for _, group in answerable_df.groupby(
            "publisher",
            sort=False,
        )
    ]

    selected_answerable = []
    row_index = 0

    while len(selected_answerable) < answerable_limit:

        added_question = False

        for group in publisher_groups:

            if row_index < len(group):

                selected_answerable.append(
                    group.iloc[row_index]
                )

                added_question = True

                if len(selected_answerable) >= answerable_limit:
                    break

        if not added_question:
            break

        row_index += 1

    if selected_answerable:

        selected_df = pd.DataFrame(
            selected_answerable
        )

    else:

        selected_df = pd.DataFrame(
            columns=dataset.columns
        )

    if include_unanswerable and len(selected_df) < max_questions:

        selected_df = pd.concat(
            [
                selected_df,
                unanswerable_df.iloc[[0]],
            ],
            ignore_index=True,
        )

    return selected_df.reset_index(drop=True)


# --------------------------------------------------
# Checkpoint management
# --------------------------------------------------

def _create_checkpoint_fingerprint(
    dataset: pd.DataFrame,
    gemini_model: str,
    retrieval_k: int,
) -> str:
    """Create a unique fingerprint for the current evaluation setup."""

    dataset_content = dataset.to_csv(
        index=False,
        lineterminator="\n",
    )

    fingerprint_data = {
        "dataset": dataset_content,
        "gemini_model": gemini_model,
        "retrieval_k": retrieval_k,
    }

    serialized_data = json.dumps(
        fingerprint_data,
        sort_keys=True,
        ensure_ascii=False,
    )

    return hashlib.sha256(
        serialized_data.encode("utf-8")
    ).hexdigest()


def _save_checkpoint(
    checkpoint_file: Path,
    fingerprint: str,
    completed_samples: dict,
) -> None:
    """Save completed evaluation samples to a JSON checkpoint."""

    checkpoint_data = {
        "fingerprint": fingerprint,
        "completed_samples": completed_samples,
    }

    temporary_file = checkpoint_file.with_suffix(
        checkpoint_file.suffix + ".tmp"
    )

    try:

        with temporary_file.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                checkpoint_data,
                file,
                indent=4,
                ensure_ascii=False,
            )

            file.write("\n")

        temporary_file.replace(checkpoint_file)

    finally:

        if temporary_file.exists():
            temporary_file.unlink()


def _load_checkpoint(
    checkpoint_file: Path,
    fingerprint: str,
) -> dict:
    """Load completed samples from a compatible checkpoint."""

    if not checkpoint_file.is_file():
        return {}

    try:

        with checkpoint_file.open(
            "r",
            encoding="utf-8",
        ) as file:

            checkpoint_data = json.load(file)

    except (json.JSONDecodeError, OSError) as error:

        raise ValueError(
            f"Could not read evaluation checkpoint: {checkpoint_file}"
        ) from error

    if not isinstance(checkpoint_data, dict):

        raise ValueError(
            "Evaluation checkpoint must contain a JSON object."
        )

    if checkpoint_data.get("fingerprint") != fingerprint:

        print(
            "Existing checkpoint does not match this evaluation. "
            "Starting a new checkpoint.",
            flush=True,
        )

        return {}

    completed_samples = checkpoint_data.get(
        "completed_samples"
    )

    if not isinstance(completed_samples, dict):

        raise ValueError(
            "Checkpoint contains an invalid completed_samples field."
        )

    return completed_samples


# --------------------------------------------------
# RAGAS scoring
# --------------------------------------------------

def _is_retryable_ragas_error(error: Exception) -> bool:
    """Return True only for errors that are likely temporary service failures."""

    message = str(error).upper()

    # Do not retry quota/rate-limit errors.
    if "429" in message or "RESOURCE_EXHAUSTED" in message:
        return False

    has_temporary_status_code = bool(
        re.search(r"\b(?:500|502|503|504)\b", message)
    )

    temporary_status_names = (
        "UNAVAILABLE",
        "INTERNAL SERVER ERROR",
        "BAD GATEWAY",
        "DEADLINE_EXCEEDED",
        "SERVICE_UNAVAILABLE",
        "GATEWAY_TIMEOUT",
    )

    has_temporary_status_name = any(
        marker in message
        for marker in temporary_status_names
    )

    return (
        has_temporary_status_code
        or has_temporary_status_name
    )


def _is_ragas_quota_error(error: Exception) -> bool:
    """Identify API quota and rate-limit errors that should stop RAGAS scoring."""

    message = str(error).upper()

    quota_markers = (
        "429",
        "RESOURCE_EXHAUSTED",
        "QUOTA EXCEEDED",
        "QUOTA_EXCEEDED",
        "RATE LIMIT",
        "RATE_LIMIT",
        "GENERATE_CONTENT_FREE_TIER",
    )

    return any(
        marker in message
        for marker in quota_markers
    )


async def _score_metric_with_retries(
    metric,
    metric_arguments: dict,
    metric_name: str,
):
    """Run one metric and retry temporary provider errors up to two times."""

    max_attempts = 3

    for attempt in range(1, max_attempts + 1):

        try:

            return await metric.ascore(
                **metric_arguments
            )

        except Exception as error:

            if (
                not _is_retryable_ragas_error(error)
                or attempt == max_attempts
            ):
                raise

            delay_seconds = 5 * (2 ** (attempt - 1))

            print(
                f"Temporary error while scoring {metric_name}: "
                f"{error}. Retrying in {delay_seconds} seconds "
                f"(attempt {attempt + 1}/{max_attempts})...",
                flush=True,
            )

            await asyncio.sleep(delay_seconds)

    raise RuntimeError(
        f"RAGAS scoring ended without a result for metric: {metric_name}"
    )


def _score_answerable_samples(
    samples: list[dict],
    judge_llm,
    judge_embeddings,
) -> list[dict]:
    """Calculate RAGAS metrics for answerable questions asynchronously."""

    metrics = {
        "faithfulness": Faithfulness(
            llm=judge_llm
        ),

        "answer_relevancy": AnswerRelevancy(
            llm=judge_llm,
            embeddings=judge_embeddings,
        ),

        "context_precision": ContextPrecision(
            llm=judge_llm
        ),

        "context_recall": ContextRecall(
            llm=judge_llm
        ),
    }

    async def score_all_metrics():
        """Score each metric with its own documented input fields."""

        for sample_number, sample in enumerate(
            samples,
            start=1,
        ):

            sample.setdefault("evaluation_errors", {})

            for metric_name, metric in metrics.items():

                print(
                    f"Scoring sample {sample_number}/{len(samples)}: "
                    f"{metric_name}",
                    flush=True,
                )

                # Each RAGAS collection metric has its own required inputs.

                if metric_name == "faithfulness":

                    metric_arguments = {
                        "user_input": sample["question"],
                        "response": sample["answer"],
                        "retrieved_contexts": sample["retrieved_contexts"],
                    }

                elif metric_name == "answer_relevancy":

                    metric_arguments = {
                        "user_input": sample["question"],
                        "response": sample["answer"],
                    }

                elif metric_name in {
                    "context_precision",
                    "context_recall",
                }:

                    metric_arguments = {
                        "user_input": sample["question"],
                        "retrieved_contexts": sample["retrieved_contexts"],
                        "reference": sample["reference_answer"],
                    }

                else:

                    raise ValueError(
                        f"Unsupported RAGAS metric: {metric_name}"
                    )

                try:

                    result = await _score_metric_with_retries(
                        metric=metric,
                        metric_arguments=metric_arguments,
                        metric_name=metric_name,
                    )

                    score = float(result.value)

                    if not math.isfinite(score):

                        raise ValueError(
                            "RAGAS returned a non-finite score."
                        )

                    sample[metric_name] = score

                    # Remove an earlier error if this metric succeeds.
                    sample["evaluation_errors"].pop(
                        metric_name,
                        None,
                    )

                except Exception as error:

                    sample[metric_name] = None

                    sample["evaluation_errors"][
                        metric_name
                    ] = str(error)

                    print(
                        f"RAGAS metric failed ({metric_name}): {error}",
                        flush=True,
                    )

                    # Stop all remaining RAGAS requests if quota is exhausted.
                    if _is_ragas_quota_error(error):

                        print(
                            "RAGAS scoring stopped because the API "
                            "quota or rate limit was reached. "
                            "Remaining metrics and samples will be skipped.",
                            flush=True,
                        )

                        return

    asyncio.run(
        score_all_metrics()
    )

    return samples


# --------------------------------------------------
# Main RAG evaluation
# --------------------------------------------------

def run_evaluation(
    dataset_path: str,
    output_dir: str,
    max_questions: int | None = 3,
    enable_ragas: bool = False,
    ragas_max_samples: int = 1,
    resume: bool = True,
) -> dict:
    """
    Evaluate the RAG pipeline using a reviewed CSV dataset.

    Development defaults:
    - Process at most three questions.
    - Include answerable and unanswerable questions.
    - Skip RAGAS scoring.
    - Resume previously completed questions.

    Full evaluation:
    Set max_questions=None and enable_ragas=True.
    """

    installed_ragas_version = version("ragas")

    if installed_ragas_version != RAGAS_VERSION_REQUIRED:

        raise RuntimeError(
            f"This evaluation implementation requires RAGAS "
            f"{RAGAS_VERSION_REQUIRED}, but found "
            f"{installed_ragas_version}."
        )

    if (
        isinstance(ragas_max_samples, bool)
        or not isinstance(ragas_max_samples, int)
        or ragas_max_samples <= 0
    ):

        raise ValueError(
            "ragas_max_samples must be a positive integer."
        )

    if not isinstance(resume, bool):

        raise TypeError(
            "resume must be True or False."
        )

    if (
        max_questions is not None
        and (
            isinstance(max_questions, bool)
            or not isinstance(max_questions, int)
            or max_questions <= 0
        )
    ):

        raise ValueError(
            "max_questions must be a positive integer or None."
        )

    dataset_file = Path(dataset_path)
    results_dir = Path(output_dir)

    if not dataset_file.is_file():

        raise FileNotFoundError(
            f"Evaluation dataset not found: {dataset_file}"
        )

    print(
        "Loading evaluation dataset...",
        flush=True,
    )

    dataset = pd.read_csv(dataset_file)

    required_columns = {
        "question",
        "reference_answer",
        "answerable",
    }

    missing_columns = (
        required_columns - set(dataset.columns)
    )

    if missing_columns:

        raise ValueError(
            f"Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if dataset.empty:

        raise ValueError(
            "Evaluation dataset is empty."
        )

    dataset["answerable"] = dataset[
        "answerable"
    ].apply(_parse_answerable)

    if dataset["question"].isna().any():

        raise ValueError(
            "Dataset contains an empty question."
        )

    dataset["question"] = (
        dataset["question"]
        .astype(str)
        .str.strip()
    )

    if dataset["question"].eq("").any():

        raise ValueError(
            "Dataset contains an empty question."
        )

    answerable_rows = dataset[
        dataset["answerable"]
    ]

    if answerable_rows["reference_answer"].isna().any():

        raise ValueError(
            "Every answerable question must have a reference answer."
        )

    reference_answers = (
        answerable_rows["reference_answer"]
        .astype(str)
        .str.strip()
    )

    if reference_answers.eq("").any():

        raise ValueError(
            "Every answerable question must have a non-empty "
            "reference answer."
        )

    selected_dataset = _select_evaluation_samples(
        dataset,
        max_questions,
    )

    print(
        f"Selected {len(selected_dataset)} of {len(dataset)} questions.",
        flush=True,
    )

    print(
        f"RAGAS scoring: {'enabled' if enable_ragas else 'disabled'}",
        flush=True,
    )

    if not GEMINI_API_KEY:

        raise ValueError(
            "GEMINI_API_KEY is missing. Add it to your .env file."
        )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_file = (
        results_dir / "evaluation_checkpoint.json"
    )

    fingerprint = _create_checkpoint_fingerprint(
        selected_dataset,
        GEMINI_MODEL,
        RETRIEVAL_K,
    )

    if resume:

        completed_samples = _load_checkpoint(
            checkpoint_file,
            fingerprint,
        )

    else:

        completed_samples = {}

        if checkpoint_file.exists():
            checkpoint_file.unlink()

    previously_completed_count = len(
        completed_samples
    )

    valid_indices = {
        str(index)
        for index in range(len(selected_dataset))
    }

    if not set(completed_samples).issubset(valid_indices):

        raise ValueError(
            "Checkpoint contains question indices that do not "
            "match the selected dataset."
        )

    print(
        f"Previously completed questions: "
        f"{len(completed_samples)}/{len(selected_dataset)}",
        flush=True,
    )

    pending_indices = [
        index
        for index in range(len(selected_dataset))
        if str(index) not in completed_samples
    ]

    if pending_indices:

        print(
            "Loading FAISS vector store...",
            flush=True,
        )

        vector_store = load_vector_store(
            FAISS_INDEX_DIR
        )

    else:

        vector_store = None

        print(
            "All selected questions are already completed.",
            flush=True,
        )

    total_questions = len(selected_dataset)

    for question_index in pending_indices:

        row = selected_dataset.iloc[
            question_index
        ]

        question = row["question"]

        expected_answerable = row["answerable"]

        reference_answer = (
            str(row["reference_answer"]).strip()
            if pd.notna(row["reference_answer"])
            else ""
        )

        print(
            f"Processing question {question_index + 1}/"
            f"{total_questions}...",
            flush=True,
        )

        start_counter = time.perf_counter()

        try:

            # retrieve_documents expects question first.
            # Named arguments make the argument order explicit.
            retrieved_documents = retrieve_documents(
                question=question,
                vector_store=vector_store,
                k=RETRIEVAL_K,
            )

            generated_result = generate_answer(
                question=question,
                retrieved_documents=retrieved_documents,
            )

        except Exception as error:

            print(
                f"Question {question_index + 1} failed: {error}",
                flush=True,
            )

            if completed_samples:

                print(
                    f"{len(completed_samples)} completed question(s) "
                    f"remain saved in: {checkpoint_file}",
                    flush=True,
                )

            raise

        latency_seconds = (
            time.perf_counter() - start_counter
        )

        answer = generated_result["answer"]

        retrieved_contexts = [
            document.page_content
            for document in retrieved_documents
        ]

        actual_abstention = _is_abstention(
            answer
        )

        sample = {
            "question": question,
            "answer": answer,
            "reference_answer": reference_answer,
            "answerable": bool(expected_answerable),

            "retrieved_contexts": retrieved_contexts,

            "retrieved_sources": [
                document.metadata
                for document in retrieved_documents
            ],

            "latency_seconds": round(
                latency_seconds,
                4,
            ),

            "expected_abstention": not expected_answerable,
            "actual_abstention": actual_abstention,

            "abstention_correct": (
                actual_abstention
                == (not expected_answerable)
            ),

            "evaluation_errors": {},
        }

        completed_samples[
            str(question_index)
        ] = sample

        _save_checkpoint(
            checkpoint_file,
            fingerprint,
            completed_samples,
        )

        print(
            f"Question {question_index + 1} completed and checkpointed.",
            flush=True,
        )

    evaluation_samples = [
        completed_samples[str(index)]
        for index in range(len(selected_dataset))
    ]

    # Recalculate abstention results for all samples.
    for sample in evaluation_samples:

        sample["actual_abstention"] = _is_abstention(
            sample["answer"]
        )

        sample["expected_abstention"] = (
            not sample["answerable"]
        )

        sample["abstention_correct"] = (
            sample["actual_abstention"]
            == sample["expected_abstention"]
        )

    answerable_samples = [
        sample
        for sample in evaluation_samples
        if sample["answerable"]
    ]

    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    for sample in evaluation_samples:

        for metric_name in metric_names:
            sample.setdefault(metric_name, None)

    ragas_samples_scored = 0

    if enable_ragas and answerable_samples:

        samples_to_score = answerable_samples[
            :ragas_max_samples
        ]

        print(
            f"Starting RAGAS scoring for "
            f"{len(samples_to_score)} answerable question(s)...",
            flush=True,
        )

        # Initialize the local Ollama model through its OpenAI-compatible API.
        # AsyncOpenAI is required because RAGAS collection metrics use ascore().
        ollama_client = AsyncOpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )

        judge_llm = llm_factory(
            model="qwen2.5:3b",
            provider="openai",
            client=ollama_client,
        )

        # Use RAGAS's native Hugging Face embeddings implementation.
        judge_embeddings = HuggingFaceEmbeddings(
            model=EMBEDDING_MODEL,
            device="cpu",
            normalize_embeddings=True,
        )

        _score_answerable_samples(
            samples=samples_to_score,
            judge_llm=judge_llm,
            judge_embeddings=judge_embeddings,
        )

        ragas_samples_scored = sum(
            1
            for sample in samples_to_score
            if all(
                sample.get(metric_name) is not None
                for metric_name in metric_names
            )
        )

        # Update the checkpoint with RAGAS results too.
        _save_checkpoint(
            checkpoint_file,
            fingerprint,
            completed_samples,
        )

    elif enable_ragas:

        print(
            "RAGAS scoring skipped: no answerable questions selected.",
            flush=True,
        )

    else:

        print(
            "Skipping RAGAS scoring.",
            flush=True,
        )

    run_id = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    results_file = (
        results_dir
        / f"evaluation_results_{run_id}.csv"
    )

    summary_file = (
        results_dir
        / f"evaluation_summary_{run_id}.json"
    )

    results_dataframe = pd.DataFrame(
        evaluation_samples
    )

    for column in [
        "retrieved_contexts",
        "retrieved_sources",
        "evaluation_errors",
    ]:

        results_dataframe[column] = (
            results_dataframe[column].apply(
                lambda value: json.dumps(
                    value,
                    ensure_ascii=False,
                )
            )
        )

    results_dataframe.to_csv(
        results_file,
        index=False,
        encoding="utf-8-sig",
    )

    unanswerable_samples = [
        sample
        for sample in evaluation_samples
        if not sample["answerable"]
    ]

    summary = {
        "run_id": run_id,

        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "ragas_version": installed_ragas_version,
        "gemini_model": GEMINI_MODEL,
        "retrieval_k": RETRIEVAL_K,

        "dataset_questions_total": len(dataset),
        "questions_evaluated": len(evaluation_samples),

        "answerable_questions": len(answerable_samples),
        "unanswerable_questions": len(unanswerable_samples),

        "ragas_enabled": enable_ragas,
        "ragas_samples_selected": (
            len(samples_to_score)
            if enable_ragas and answerable_samples
            else 0
        ),
        "ragas_samples_scored": ragas_samples_scored,

        "resumed_from_checkpoint": (
            previously_completed_count > 0
        ),

        "average_latency_seconds": round(
            sum(
                sample["latency_seconds"]
                for sample in evaluation_samples
            ) / len(evaluation_samples),
            4,
        ),

        "abstention_accuracy": (
            round(
                sum(
                    sample["abstention_correct"]
                    for sample in unanswerable_samples
                ) / len(unanswerable_samples),
                4,
            )
            if unanswerable_samples
            else None
        ),

        "ragas_average_scores": {
            metric_name: (
                round(
                    results_dataframe[metric_name].mean(),
                    4,
                )
                if results_dataframe[metric_name].notna().any()
                else None
            )
            for metric_name in metric_names
        },

        "results_file": str(results_file),
        "checkpoint_file": str(checkpoint_file),
    }

    with summary_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

        file.write("\n")

    summary["summary_file"] = str(
        summary_file
    )

    print(
        "Evaluation completed successfully.",
        flush=True,
    )

    print(
        f"Results saved to: {results_file}",
        flush=True,
    )

    print(
        f"Summary saved to: {summary_file}",
        flush=True,
    )

    print(
        f"Checkpoint retained at: {checkpoint_file}",
        flush=True,
    )

    return summary


# --------------------------------------------------
# Retrieval comparison experiment
# --------------------------------------------------

def run_retrieval_comparison(
    dataset_path: str,
    output_dir: str,
) -> dict:
    """
    Compare Similarity, MMR, and RRF retrieval methods.

    This is a retrieval-only experiment.
    It does not call Gemini to generate answers or run RAGAS.
    """

    dataset_file = Path(dataset_path)

    if not dataset_file.is_file():

        raise FileNotFoundError(
            f"Evaluation dataset not found: {dataset_file}"
        )

    dataset = pd.read_csv(dataset_file)

    required_columns = {
        "question",
        "answerable",
        "source_file",
    }

    missing_columns = (
        required_columns - set(dataset.columns)
    )

    if missing_columns:

        raise ValueError(
            f"Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    dataset["answerable"] = dataset[
        "answerable"
    ].apply(_parse_answerable)

    answerable_dataset = dataset[
        dataset["answerable"]
    ].copy()

    if answerable_dataset.empty:

        raise ValueError(
            "No answerable questions were found in the dataset."
        )

    if answerable_dataset["source_file"].isna().any():

        raise ValueError(
            "An answerable question is missing its expected source file."
        )

    answerable_dataset["question"] = (
        answerable_dataset["question"]
        .astype(str)
        .str.strip()
    )

    if answerable_dataset["question"].eq("").any():

        raise ValueError(
            "An answerable question is empty."
        )

    answerable_dataset["source_file"] = (
        answerable_dataset["source_file"]
        .astype(str)
        .str.strip()
    )

    if answerable_dataset["source_file"].eq("").any():

        raise ValueError(
            "An answerable question has an empty source file."
        )

    print(
        f"Answerable questions selected: {len(answerable_dataset)}",
        flush=True,
    )

    print(
        "Loading FAISS vector store...",
        flush=True,
    )

    vector_store = load_vector_store(
        FAISS_INDEX_DIR
    )

    retrieval_methods = {
        "similarity": {
            "type": "similarity",
            "lambda_mult": None,
        },

        "mmr_07": {
            "type": "mmr",
            "lambda_mult": 0.7,
        },

        "mmr_085": {
            "type": "mmr",
            "lambda_mult": 0.85,
        },

        "rrf": {
            "type": "rrf",
            "lambda_mult": 0.7,
        },
    }

    retrieval_k = RETRIEVAL_K

    question_results = []

    for question_number, (_, row) in enumerate(
        answerable_dataset.iterrows(),
        start=1,
    ):

        question = row["question"]

        expected_source = Path(
            row["source_file"]
        ).name.casefold()

        print(
            f"Question {question_number}/"
            f"{len(answerable_dataset)}: {question}",
            flush=True,
        )

        for method_name, configuration in retrieval_methods.items():

            if configuration["type"] == "similarity":

                documents = retrieve_documents(
                    question=question,
                    vector_store=vector_store,
                    k=retrieval_k,
                    search_type="similarity",
                )

            elif configuration["type"] == "mmr":

                documents = retrieve_documents(
                    question=question,
                    vector_store=vector_store,
                    k=retrieval_k,
                    search_type="mmr",
                    fetch_k=12,
                    lambda_mult=configuration["lambda_mult"],
                )

            else:

                documents = retrieve_documents_rrf(
                    question=question,
                    vector_store=vector_store,
                    k=retrieval_k,
                    candidate_k=8,
                    fetch_k=20,
                    lambda_mult=configuration["lambda_mult"],
                    rrf_k=60,
                )

            retrieved_sources = [
                document.metadata.get("source", "")
                for document in documents
            ]

            retrieved_filenames = [
                Path(source).name.casefold()
                for source in retrieved_sources
                if source
            ]

            retrieved_pages = [
                (
                    document.metadata.get("source"),
                    document.metadata.get("page_number"),
                )
                for document in documents
            ]

            hit_at_1 = (
                expected_source in retrieved_filenames[:1]
            )

            hit_at_3 = (
                expected_source in retrieved_filenames[:3]
            )

            hit_at_4 = (
                expected_source in retrieved_filenames[:4]
            )

            unique_pdfs = len(
                set(retrieved_sources)
            )

            unique_pages = len(
                set(retrieved_pages)
            )

            question_results.append(
                {
                    "question": question,
                    "method": method_name,
                    "expected_source": expected_source,

                    "retrieved_sources": json.dumps(
                        retrieved_sources,
                        ensure_ascii=False,
                    ),

                    "hit_at_1": hit_at_1,
                    "hit_at_3": hit_at_3,
                    "hit_at_4": hit_at_4,

                    "unique_pdfs": unique_pdfs,
                    "unique_pages": unique_pages,
                }
            )

    results_dataframe = pd.DataFrame(
        question_results
    )

    summary_rows = []

    for method_name in retrieval_methods:

        method_results = results_dataframe[
            results_dataframe["method"] == method_name
        ]

        summary_rows.append(
            {
                "method": method_name,

                "questions_evaluated": len(
                    method_results
                ),

                "hit_at_1_percent": round(
                    method_results["hit_at_1"].mean() * 100,
                    2,
                ),

                "hit_at_3_percent": round(
                    method_results["hit_at_3"].mean() * 100,
                    2,
                ),

                "hit_at_4_percent": round(
                    method_results["hit_at_4"].mean() * 100,
                    2,
                ),

                "average_unique_pdfs": round(
                    method_results["unique_pdfs"].mean(),
                    2,
                ),

                "average_unique_pages": round(
                    method_results["unique_pages"].mean(),
                    2,
                ),
            }
        )

    summary_dataframe = pd.DataFrame(
        summary_rows
    )

    results_dir = Path(output_dir)

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_id = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    details_file = (
        results_dir
        / f"retrieval_comparison_details_{run_id}.csv"
    )

    summary_file = (
        results_dir
        / f"retrieval_comparison_summary_{run_id}.csv"
    )

    results_dataframe.to_csv(
        details_file,
        index=False,
        encoding="utf-8-sig",
    )

    summary_dataframe.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nRETRIEVAL COMPARISON RESULTS"
    )

    print(
        summary_dataframe.to_string(index=False)
    )

    print(
        f"\nDetailed results saved to: {details_file}"
    )

    print(
        f"Summary saved to: {summary_file}"
    )

    return {
        "questions_evaluated": len(answerable_dataset),
        "summary": summary_rows,
        "details_file": str(details_file),
        "summary_file": str(summary_file),
    }