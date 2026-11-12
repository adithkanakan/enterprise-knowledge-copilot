"""Evidence-based answer generation using Google Gemini."""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import GEMINI_API_KEY, GEMINI_MODEL


INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I could not find sufficient information in the available "
    "technical documents to answer this question reliably."
)

INCOMPLETE_RESPONSE_MESSAGE = (
    "The AI response was interrupted before completion. "
    "Please try the question again."
)


def _extract_response_text(response) -> str:
    """
    Extract readable text from a Gemini response.

    LangChain may return response content as either a string
    or a list of content blocks.
    """

    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []

        for part in content:

            if isinstance(part, str):
                text_parts.append(part)

            elif isinstance(part, dict):
                text = part.get("text")

                if isinstance(text, str):
                    text_parts.append(text)

        return "\n".join(text_parts).strip()

    return ""


def _response_was_truncated(response) -> bool:
    """
    Check whether Gemini stopped because it reached its
    maximum output token limit.

    A normal STOP finish reason is not considered truncation.
    """

    metadata = getattr(response, "response_metadata", {}) or {}

    additional_kwargs = getattr(
        response,
        "additional_kwargs",
        {},
    ) or {}

    finish_reason = metadata.get("finish_reason")

    if finish_reason is None:
        finish_reason = additional_kwargs.get("finish_reason")

    if finish_reason is None:
        finish_reason = additional_kwargs.get("finishReason")

    if finish_reason is None:
        return False

    # Google Gemini uses numeric value 2 for MAX_TOKENS.
    if isinstance(finish_reason, int) and not isinstance(
        finish_reason,
        bool,
    ):
        return finish_reason == 2

    # Handle string and enum-like finish reasons.
    normalized_reason = str(
        getattr(finish_reason, "name", finish_reason)
    ).upper()

    return normalized_reason in {
        "MAX_TOKENS",
        "FINISHREASON.MAX_TOKENS",
        "LENGTH",
    }


def generate_answer(
    question: str,
    retrieved_documents: list[Document],
) -> dict:
    """
    Generate an evidence-based answer using Gemini and retrieved
    document chunks.

    Return the answer and its source references.
    """

    # Validate the question.
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    # Return the fallback when retrieval finds no documents.
    if not retrieved_documents:
        return {
            "answer": INSUFFICIENT_EVIDENCE_MESSAGE,
            "sources": [],
        }

    # Validate Gemini API configuration.
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is missing. Add it to your .env file."
        )

    context_parts = []
    sources = []

    # Prepare retrieved document excerpts and source references.
    for index, document in enumerate(
        retrieved_documents,
        start=1,
    ):

        if not isinstance(document, Document):
            raise TypeError(
                "Every retrieved context must be a LangChain Document."
            )

        source = document.metadata.get(
            "source",
            "Unknown source",
        )

        page_number = document.metadata.get(
            "page_number",
            "Unknown",
        )

        context_parts.append(
            f"[Source {index}]\n"
            f"File: {source}\n"
            f"Page: {page_number}\n"
            f"Content:\n{document.page_content}"
        )

        sources.append(
            {
                "id": index,
                "source": source,
                "page_number": page_number,
            }
        )

    context = "\n\n".join(context_parts)

    # Instructions that control Gemini's answer behaviour.
    system_prompt = """
You are an enterprise IT helpdesk assistant for TechNova Solutions.

Your task is to answer employee IT support questions using only
the supplied technical document excerpts.

STRICT EVIDENCE RULES:

1. Use only information explicitly supported by the supplied excerpts.
2. Do not invent commands, procedures, configuration values,
   troubleshooting steps, or technical explanations.
3. Do not use outside knowledge to fill gaps in the evidence.
4. If the excerpts do not contain enough information, clearly
   state which information is unavailable.
5. Do not assume that a procedure from one product or operating
   system applies to another.

CITATION RULES:

6. Cite supporting excerpts using their assigned source numbers,
   such as [1] or [2].
7. Place citations immediately after the claims they support.
8. Use only source numbers provided in the retrieved excerpts.
9. Do not cite a source unless it actually supports the claim.
10. Do not invent source numbers, filenames, or page numbers.

TECHNICAL INSTRUCTION RULES:

11. Give clear, practical troubleshooting steps when supported
    by the excerpts.
12. Present steps in a logical order, starting with checks that
    do not unnecessarily change system configuration.
13. Do not recommend destructive or security-weakening actions
    as routine troubleshooting steps.
14. If a source describes a potentially disruptive or insecure
    procedure, identify that risk instead of presenting it as
    an ordinary recommended action.
15. Never provide an incomplete command or configuration example.
16. If the excerpts provide only part of a configuration,
    explain the limitation instead of inventing the missing fields.
17. Preserve important technical details and configuration values
    exactly as provided by the source.

ANSWER COMPLETENESS:

18. Provide a complete answer, not an unfinished list or sentence.
19. Prefer concise, clearly numbered steps when appropriate.
20. Avoid unnecessary repetition and unrelated troubleshooting.
21. If only part of the question can be answered from the evidence,
    answer that part and explicitly identify what remains unknown.

Your final response must be technically clear, evidence-based,
and complete.
"""

    human_prompt = f"""
Employee question:
{question.strip()}

Retrieved technical document excerpts:

{context}

Answer the employee's question using only these excerpts.

Before finishing, check that:
- Every technical recommendation is supported by the excerpts.
- Every important factual claim has an appropriate citation.
- No command or configuration example is incomplete.
- No unsupported troubleshooting step has been added.
- The response ends with a complete sentence.
"""

    # Initialize the Gemini model.
    model = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        api_key=GEMINI_API_KEY,
        temperature=0,
    )

    # Generate the answer.
    response = model.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
    )

    # Detect provider-reported output truncation.
    if _response_was_truncated(response):
        return {
            "answer": INCOMPLETE_RESPONSE_MESSAGE,
            "sources": sources,
        }

    # Extract readable response text.
    answer = _extract_response_text(response)

    # Handle empty model responses.
    if not answer:
        answer = INSUFFICIENT_EVIDENCE_MESSAGE

    return {
        "answer": answer,
        "sources": sources,
    }