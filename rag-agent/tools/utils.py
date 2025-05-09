"""
Utility functions for the RAG tools.
"""

import re
from typing import Any, Dict

import vertexai
from google.adk.tools.tool_context import ToolContext
from vertexai import rag

from ..config import (
    DEFAULT_EMBEDDING_MODEL,
    LOCATION,
    PROJECT_ID,
)


def get_corpus_resource_name(corpus_name: str) -> str:
    """
    Convert a corpus name to its full resource name if needed.
    Handles various input formats and ensures the returned name follows Vertex AI's requirements.

    Args:
        corpus_name (str): The corpus name or display name

    Returns:
        str: The full resource name of the corpus
    """
    print(f"Corpus name: {corpus_name}")

    # If it's already a full resource name with the projects/locations/ragCorpora format
    if re.match(r"^projects/[^/]+/locations/[^/]+/ragCorpora/[^/]+$", corpus_name):
        return corpus_name

    # Check if this is a display name of an existing corpus
    try:
        # Initialize Vertex AI if needed
        vertexai.init(project=PROJECT_ID, location=LOCATION)

        # List all corpora and check if there's a match with the display name
        corpora = rag.list_corpora()
        for corpus in corpora:
            if hasattr(corpus, "display_name") and corpus.display_name == corpus_name:
                return corpus.name
    except Exception:
        # If we can't check, continue with the default behavior
        pass

    # If it contains partial path elements, extract just the corpus ID
    if "/" in corpus_name:
        # Extract the last part of the path as the corpus ID
        corpus_id = corpus_name.split("/")[-1]
    else:
        corpus_id = corpus_name

    # Remove any special characters that might cause issues
    corpus_id = re.sub(r"[^a-zA-Z0-9_-]", "_", corpus_id)

    # Construct the standardized resource name
    return f"projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{corpus_id}"


def check_corpus_exists(corpus_name: str, tool_context: ToolContext) -> bool:
    """
    Check if a corpus with the given name exists.

    Args:
        corpus_name (str): The name of the corpus to check
        tool_context (ToolContext): The tool context for state management

    Returns:
        bool: True if the corpus exists, False otherwise
    """
    # Check state first if tool_context is provided
    if tool_context.state.get(f"corpus_exists_{corpus_name}"):
        return True

    try:
        # Initialize Vertex AI
        vertexai.init(project=PROJECT_ID, location=LOCATION)

        # Get full resource name
        corpus_resource_name = get_corpus_resource_name(corpus_name)

        # List all corpora and check if this one exists
        corpora = rag.list_corpora()
        for corpus in corpora:
            if (
                corpus.name == corpus_resource_name
                or corpus.display_name == corpus_name
            ):
                # Update state
                tool_context.state[f"corpus_exists_{corpus_name}"] = True
                return True

        return False
    except Exception:
        # If we can't check, assume it doesn't exist
        return False


def create_corpus_if_not_exists(
    corpus_name: str, tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Create a corpus if it doesn't already exist.

    Args:
        corpus_name (str): The name of the corpus to create if needed
        tool_context (ToolContext): The tool context for state management

    Returns:
        Dict[str, Any]: Status information about the operation with the following keys:
            - success (bool): True if the corpus was created or already exists
            - corpus_name (str): The name of the corpus
            - was_created (bool): Whether the corpus was newly created
            - status (str): Status message ("success" or "error")
            - message (str): Detailed message about the operation
    """
    # Check if corpus already exists
    exists = check_corpus_exists(corpus_name, tool_context)
    if exists:
        return {
            "success": True,
            "status": "success",
            "message": f"Corpus '{corpus_name}' already exists",
            "corpus_name": corpus_name,
            "was_created": False,
        }

    try:
        # Initialize Vertex AI
        vertexai.init(project=PROJECT_ID, location=LOCATION)

        # Clean corpus name for use as display name
        display_name = re.sub(r"[^a-zA-Z0-9_-]", "_", corpus_name)

        # Configure embedding model
        embedding_model_config = rag.RagEmbeddingModelConfig(
            vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                publisher_model=DEFAULT_EMBEDDING_MODEL
            )
        )

        # Create the corpus
        rag_corpus = rag.create_corpus(
            display_name=display_name,
            backend_config=rag.RagVectorDbConfig(
                rag_embedding_model_config=embedding_model_config
            ),
        )

        # Update state
        tool_context.state[f"corpus_exists_{corpus_name}"] = True

        return {
            "success": True,
            "status": "success",
            "message": f"Successfully created corpus '{corpus_name}'",
            "corpus_name": rag_corpus.name,
            "display_name": rag_corpus.display_name,
            "was_created": True,
        }

    except Exception as e:
        return {
            "success": False,
            "status": "error",
            "message": f"Error creating corpus: {str(e)}",
            "corpus_name": corpus_name,
            "was_created": False,
        }
