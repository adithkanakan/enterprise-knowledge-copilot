"""Utilities for logging RAG interactions."""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document


def log_interaction(
    question: str,
    answer: str,
    retrieved_contexts: list[Document],
    latency_seconds: float,
    log_path: str | Path,
) -> None:
    """
    Save one user interaction to a JSON log file.

    Each record contains:
    - User question
    - Generated answer
    - Retrieved document contexts and metadata
    - UTC timestamp
    - Response latency
    """

    # Validate the question.
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    # Validate the answer.
    if not isinstance(answer, str):
        raise TypeError("Answer must be a string.")

    # Validate retrieved contexts.
    if not isinstance(retrieved_contexts, list):
        raise TypeError("Retrieved contexts must be a list.")

    if not all(
        isinstance(context, Document)
        for context in retrieved_contexts
    ):
        raise TypeError(
            "Every retrieved context must be a LangChain Document."
        )

    # Validate response latency.
    if (
        isinstance(latency_seconds, bool)
        or not isinstance(latency_seconds, (int, float))
        or not math.isfinite(latency_seconds)
        or latency_seconds < 0
    ):
        raise ValueError(
            "Latency must be a finite, non-negative number."
        )

    # Prepare the log file path.
    log_file = Path(log_path)

    # Create the parent directory if it does not exist.
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Create the interaction record.
    record = {
        "question": question.strip(),
        "answer": answer,
        "retrieved_contexts": [
            {
                "page_content": context.page_content,
                "metadata": context.metadata,
            }
            for context in retrieved_contexts
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latency_seconds": float(latency_seconds),
    }

    # Load existing interaction records.
    if log_file.exists() and log_file.stat().st_size > 0:

        try:
            with log_file.open("r", encoding="utf-8") as file:
                records = json.load(file)

        except json.JSONDecodeError as error:
            raise ValueError(
                f"Existing log file contains invalid JSON: {log_file}"
            ) from error

        if not isinstance(records, list):
            raise ValueError(
                f"Expected a JSON list in the log file: {log_file}"
            )

    else:
        records = []

    # Add the new interaction.
    records.append(record)

    # Save all records back to the JSON file.
    with log_file.open("w", encoding="utf-8") as file:
        json.dump(
            records,
            file,
            indent=4,
            ensure_ascii=False,
        )

        file.write("\n")