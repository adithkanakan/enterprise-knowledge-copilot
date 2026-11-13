
import csv
import re
from pathlib import Path
from collections import defaultdict

from src.retrieval import load_vector_store

DATASET = "evaluation/datasets/evaluation_dataset_test.csv"
OUTPUT = "evaluation/results/query_expansion_comparison.csv"

STOP_WORDS = {
    "what", "does", "is", "are", "the", "a", "an", "of",
    "for", "to", "in", "on", "how", "why", "which",
    "and", "or", "can", "do", "should", "explain",
    "describe", "tell", "me", "about", "please"
}


def normalize(value):
    return str(value or "").replace("\\", "/").strip("/").lower()


def keyword_query(question):
    words = re.findall(r"[a-zA-Z0-9_.-]+", question.lower())
    result = " ".join(w for w in words if w not in STOP_WORDS)
    return result or question


def matching_rank(documents, expected_source):
    expected = normalize(expected_source)

    for rank, doc in enumerate(documents, 1):
        source = normalize(doc.metadata.get("source", ""))

        if (
            source == expected
            or source.endswith("/" + expected)
            or expected.endswith("/" + source)
        ):
            return rank

    return None


# Load dataset
with open(DATASET, encoding="utf-8-sig", newline="") as file:
    dataset = list(csv.DictReader(file))

questions = [
    row for row in dataset
    if row.get("answerable", "").strip().lower()
    in {"true", "1", "yes"}
]

if not questions:
    raise ValueError("No answerable questions found in the dataset.")

if any(not row.get("source_file", "").strip() for row in questions):
    raise ValueError("An answerable question has no expected source_file.")

print(f"Loading FAISS index... {len(questions)} questions")

vs = load_vector_store()
results = []

for number, item in enumerate(questions, 1):
    question = item["question"]
    expected = item["source_file"]

    original = vs.similarity_search(question, k=10)
    keywords = vs.similarity_search(keyword_query(question), k=10)

    # Reciprocal Rank Fusion
    scores = defaultdict(float)
    documents = {}

    for result_set in (original, keywords):
        for rank, doc in enumerate(result_set, 1):
            key = (
                normalize(doc.metadata.get("source", "")),
                str(doc.metadata.get("page_number", "")),
                str(doc.metadata.get("start_index", "")),
                doc.page_content
            )

            scores[key] += 1 / (60 + rank)
            documents[key] = doc

    fused_keys = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    fused_documents = [documents[key] for key in fused_keys]

    baseline_rank = matching_rank(original, expected)
    rrf_rank = matching_rank(fused_documents, expected)

    results.append({
        "question": question,
        "expected_source": expected,
        "baseline_rank": baseline_rank or "",
        "rrf_rank": rrf_rank or "",
        "baseline_hit1": baseline_rank is not None and baseline_rank <= 1,
        "baseline_hit3": baseline_rank is not None and baseline_rank <= 3,
        "baseline_hit4": baseline_rank is not None and baseline_rank <= 4,
        "rrf_hit1": rrf_rank is not None and rrf_rank <= 1,
        "rrf_hit3": rrf_rank is not None and rrf_rank <= 3,
        "rrf_hit4": rrf_rank is not None and rrf_rank <= 4,
    })

    print(f"Completed {number}/{len(questions)}", end="\r")


# Save separate experimental results
Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)


# Print comparison
print(f"\n\nQuestions evaluated: {len(results)}")

for method in ("baseline", "rrf"):
    print(f"\n{method.upper()}")

    for k in (1, 3, 4):
        hits = sum(row[f"{method}_hit{k}"] for row in results)
        print(f"Hit@{k}: {hits}/{len(results)} = {hits / len(results):.2%}")

print(f"\nDetailed results: {OUTPUT}")
