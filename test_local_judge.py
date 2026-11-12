import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness


def main():
    checkpoint_path = Path(
        "evaluation/results/evaluation_checkpoint.json"
    )

    checkpoint = json.loads(
        checkpoint_path.read_text(encoding="utf-8")
    )

    samples = [
        sample
        for sample in checkpoint["completed_samples"].values()
        if (
            sample.get("answerable")
            and sample.get("answer")
            and sample.get("retrieved_contexts")
        )
    ][:5]

    print(f"Testing {len(samples)} existing answers.")

    client = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )

    judge_llm = llm_factory(
        model="qwen2.5:3b",
        provider="openai",
        client=client,
    )

    metric = Faithfulness(llm=judge_llm)

    async def evaluate():
        for index, sample in enumerate(samples, start=1):
            print(f"\nEvaluating sample {index}/5...")

            try:
                result = await metric.ascore(
                    user_input=sample["question"],
                    response=sample["answer"],
                    retrieved_contexts=sample["retrieved_contexts"],
                )

                print(f"Faithfulness: {result.value}")

            except Exception as error:
                print(f"Evaluation error: {error}")

    asyncio.run(evaluate())


if __name__ == "__main__":
    main()
    