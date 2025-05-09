"""
Tool for deleting a Vertex AI RAG corpus when it's no longer needed.
"""

from typing import Dict

import vertexai
from google.adk.tools.tool_context import ToolContext
from vertexai import rag

from ..config import LOCATION, PROJECT_ID
from .utils import check_corpus_exists, get_corpus_resource_name


def delete_corpus(
    corpus_name: str,
    confirm: bool,
    tool_context: ToolContext,
) -> Dict:
    """
    Delete a Vertex AI RAG corpus when it's no longer needed.
    Requires confirmation to prevent accidental deletion.

    Args:
        corpus_name (str): The full resource name of the corpus to delete.
                           Preferably use the resource_name from list_corpora results.
        confirm (bool): Must be set to True to confirm deletion
        tool_context (ToolContext): The tool context

    Returns:
        dict: Status information about the deletion operation
    """
    try:
        # Check if deletion is confirmed
        if not confirm:
            return {
                "status": "error",
                "message": "Deletion not confirmed. Please set confirm=True to delete the corpus.",
                "corpus_name": corpus_name,
            }

        # Initialize Vertex AI
        vertexai.init(project=PROJECT_ID, location=LOCATION)

        # Check if corpus exists
        if not check_corpus_exists(corpus_name, tool_context):
            return {
                "status": "error",
                "message": f"Corpus '{corpus_name}' does not exist, so it cannot be deleted.",
                "corpus_name": corpus_name,
            }

        # Get the corpus resource name
        corpus_resource_name = get_corpus_resource_name(corpus_name)

        # Delete the corpus
        rag.delete_corpus(corpus_resource_name)

        # Update state to reflect the deletion
        state_key = f"corpus_exists_{corpus_name}"
        if state_key in tool_context.state:
            # Set the value to False instead of deleting the key
            tool_context.state[state_key] = False

        return {
            "status": "success",
            "message": f"Successfully deleted corpus '{corpus_name}'",
            "corpus_name": corpus_name,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error deleting corpus: {str(e)}",
            "corpus_name": corpus_name,
        }
