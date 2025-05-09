"""
Tool for deleting a specific document from a Vertex AI RAG corpus.
"""

from typing import Dict

import vertexai
from google.adk.tools.tool_context import ToolContext
from vertexai import rag

from ..config import LOCATION, PROJECT_ID
from .utils import check_corpus_exists, get_corpus_resource_name


def delete_document(
    corpus_name: str,
    document_id: str,
    tool_context: ToolContext,
) -> Dict:
    """
    Delete a specific document from a Vertex AI RAG corpus.

    Args:
        corpus_name (str): The full resource name of the corpus containing the document.
                          Preferably use the resource_name from list_corpora results.
        document_id (str): The ID of the specific document/file to delete. This can be
                          obtained from get_corpus_info results.
        tool_context (ToolContext): The tool context

    Returns:
        dict: Status information about the deletion operation
    """
    try:
        # Initialize Vertex AI
        vertexai.init(project=PROJECT_ID, location=LOCATION)

        # Check if corpus exists
        if not check_corpus_exists(corpus_name, tool_context):
            return {
                "status": "error",
                "message": f"Corpus '{corpus_name}' does not exist, so the document cannot be deleted.",
                "corpus_name": corpus_name,
                "document_id": document_id,
            }

        # Get the corpus resource name
        corpus_resource_name = get_corpus_resource_name(corpus_name)

        # Construct the full document resource name
        document_resource_name = f"{corpus_resource_name}/ragFiles/{document_id}"

        # Delete the document
        rag.delete_file(name=document_resource_name)

        return {
            "status": "success",
            "message": f"Successfully deleted document '{document_id}' from corpus '{corpus_name}'",
            "corpus_name": corpus_name,
            "document_id": document_id,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error deleting document: {str(e)}",
            "corpus_name": corpus_name,
            "document_id": document_id,
        }
