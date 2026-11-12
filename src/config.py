"""Central configuration for the project."""

import os

from dotenv import load_dotenv


# Load environment variables from the .env file.
load_dotenv()


# --------------------------------------------------
# Gemini configuration
# --------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash",
)


# --------------------------------------------------
# Embedding model configuration
# --------------------------------------------------

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)


# --------------------------------------------------
# Document directories
# --------------------------------------------------

RAW_DATA_DIR = "data/raw"

PROCESSED_DATA_DIR = "data/processed"


# --------------------------------------------------
# Vector database configuration
# --------------------------------------------------

FAISS_INDEX_DIR = "indexes/faiss_index"


# --------------------------------------------------
# Interaction logging configuration
# --------------------------------------------------

INTERACTION_LOG_PATH = "logs/rag_history.json"


# --------------------------------------------------
# Evaluation configuration
# --------------------------------------------------

EVALUATION_DATASET_PATH = (
    "evaluation/datasets/evaluation_dataset_test.csv"
)

EVALUATION_RESULTS_DIR = "evaluation/results"