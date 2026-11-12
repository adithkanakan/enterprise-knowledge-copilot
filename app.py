
"""Streamlit application for Enterprise Knowledge Copilot."""

import time
from pathlib import Path

import streamlit as st

from src.config import (
    FAISS_INDEX_DIR,
    INTERACTION_LOG_PATH,
)
from src.retrieval import (
    load_vector_store,
    retrieve_documents,
)
from src.generation import generate_answer
from src.logging_utils import log_interaction


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Enterprise Knowledge Copilot",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Enterprise Knowledge Copilot")

st.caption(
    "An evidence-based IT Helpdesk assistant powered by "
    "Retrieval-Augmented Generation."
)

st.divider()


# --------------------------------------------------
# Load FAISS vector store
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading knowledge base...")
def get_vector_store():
    """Load and cache the existing FAISS index."""
    return load_vector_store(FAISS_INDEX_DIR)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.header("About")

    st.write(
        "This assistant answers IT support questions using "
        "the technical documents available in its knowledge base."
    )

    st.divider()

    st.subheader("System components")

    st.markdown(
        """
        - **Knowledge base:** FAISS
        - **Embeddings:** MiniLM
        - **Retrieval:** Maximum Marginal Relevance
        - **LLM:** Google Gemini
        - **Logging:** JSON
        """
    )

    st.divider()

    st.caption(
        "Answers are generated using retrieved technical "
        "documentation. Always verify critical changes "
        "against the cited sources."
    )


# --------------------------------------------------
# Question input
# --------------------------------------------------

st.subheader("Ask the knowledge base")

st.write(
    "Enter an IT-related question to retrieve relevant "
    "documentation and generate an evidence-based answer."
)

with st.form("question_form"):
    question = st.text_input(
        "Your question",
        placeholder=(
            "e.g. How can I troubleshoot an SSH connection "
            "timeout in an Amazon EC2 instance?"
        ),
    )

    submitted = st.form_submit_button(
    "Ask Copilot",
    type="primary",
)

if submitted and not question.strip():
    st.warning("Please enter a question before submitting.")
# --------------------------------------------------
# Process question
# --------------------------------------------------

if submitted:

    # Clear the previous result before processing a new question.
    st.session_state.pop("rag_result", None)

    try:
        start_time = time.perf_counter()

        with st.spinner("Searching the knowledge base..."):

            # Load the existing FAISS index.
            vector_store = get_vector_store()

            # Retrieve the four most relevant document chunks.
            retrieved_documents = retrieve_documents(
                question=question,
                vector_store=vector_store,
                k=4,
                search_type="mmr",
                fetch_k=12,
                lambda_mult=0.7,
            )

        if not retrieved_documents:

            st.warning(
                "No relevant technical documents were found."
            )

        else:

            with st.spinner("Generating an evidence-based answer..."):

                # Generate the answer using Gemini.
                result = generate_answer(
                    question=question,
                    retrieved_documents=retrieved_documents,
                )

            answer = result["answer"]

            latency = time.perf_counter() - start_time

            # Log the interaction.
            log_interaction(
                question=question,
                answer=answer,
                retrieved_contexts=retrieved_documents,
                latency_seconds=latency,
                log_path=INTERACTION_LOG_PATH,
            )

            # Save the result in session state.
            st.session_state["rag_result"] = {
                "question": question,
                "answer": answer,
                "sources": result["sources"],
                "documents": retrieved_documents,
                "latency": latency,
            }

    except FileNotFoundError:
        st.error(
            "The FAISS knowledge base could not be found. "
            "Please build the vector index before using the application."
        )

    except ValueError as error:
        st.error(f"Configuration or input error: {error}")

    except Exception as error:
        st.error(
            "Something went wrong while processing your question. "
            "Please check the application configuration and logs."
        )
        st.exception(error)


# --------------------------------------------------
# Display answer
# --------------------------------------------------

if "rag_result" in st.session_state:

    result = st.session_state["rag_result"]

    st.divider()

    st.subheader("Your question")
    st.write(result["question"])

    st.subheader("Copilot's answer")

    st.markdown(result["answer"])

    # Response information.
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Response time",
            f"{result['latency']:.2f} seconds",
        )

    with col2:
        st.metric(
            "Sources retrieved",
            len(result["documents"]),
        )

    # --------------------------------------------------
    # Display source references and excerpts
    # --------------------------------------------------

    st.divider()
    st.subheader("Retrieved evidence")

    st.caption(
        "The following document excerpts were provided to "
        "Gemini when generating the answer."
    )

    for index, document in enumerate(
        result["documents"],
        start=1,
    ):

        metadata = document.metadata

        source = metadata.get(
            "source",
            "Unknown source",
        )

        page_number = metadata.get(
            "page_number",
            "Unknown",
        )

        with st.expander(
            f"[{index}] {Path(str(source)).name} — Page {page_number}"
        ):

            st.markdown(f"**Source:** `{source}`")
            st.markdown(f"**Page:** {page_number}")

            st.markdown("**Retrieved excerpt:**")

            st.text(document.page_content)

    st.divider()

    st.caption(
        "Interaction saved to the project JSON log."
    )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Enterprise Knowledge Copilot | "
    "Evidence-based IT Helpdesk Assistant"
)
