"""
Tool for listing all available Vertex AI RAG corpora.
"""

from typing import Dict, List, Union

import vertexai
from google.adk.tools.tool_context import ToolContext
from vertexai import rag

from ..config import LOCATION, PROJECT_ID


def list_corpora(
    tool_context: ToolContext,
) -> Dict:
    """
    List all available Vertex AI RAG corpora.

    Args:
        tool_context (ToolContext): The tool context

    Returns:
        dict: A list of available corpora and status, with each corpus containing:
            - resource_name: The full resource name to use with other tools
            - display_name: The human-readable name of the corpus
            - create_time: When the corpus was created
            - update_time: When the corpus was last updated
    """
    try:
        print("Listing corpora...", PROJECT_ID)
        # Initialize Vertex AI
        vertexai.init(project=PROJECT_ID, location=LOCATION)

        # Get the list of corpora
        corpora = rag.list_corpora()

        # Process corpus information into a more usable format
        corpus_info: List[Dict[str, Union[str, int]]] = []
        for corpus in corpora:
            corpus_data: Dict[str, Union[str, int]] = {
                "resource_name": corpus.name,  # Full resource name for use with other tools
                "display_name": corpus.display_name,
                "create_time": (
                    str(corpus.create_time) if hasattr(corpus, "create_time") else ""
                ),
                "update_time": (
                    str(corpus.update_time) if hasattr(corpus, "update_time") else ""
                ),
            }

            corpus_info.append(corpus_data)

        print(f"Corpus info: {corpus_info}")

        return {
            "status": "success",
            "message": f"Found {len(corpus_info)} corpus/corpora",
            "corpora": corpus_info,
            "count": len(corpus_info),
            "note": "Use the 'resource_name' field (not 'display_name') when referencing corpora in other tools",
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error listing corpora: {str(e)}",
        }
