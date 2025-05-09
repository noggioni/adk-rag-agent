"""
Tool for retrieving detailed information about a specific RAG corpus.
"""

from typing import Dict

import vertexai
from google.adk.tools.tool_context import ToolContext
from vertexai import rag

from ..config import LOCATION, PROJECT_ID
from .utils import check_corpus_exists, get_corpus_resource_name


def get_corpus_info(
    corpus_name: str,
    tool_context: ToolContext,
) -> Dict:
    """
    Get detailed information about a specific RAG corpus, including its files.

    Args:
        corpus_name (str): The full resource name of the corpus to get information about.
                           Preferably use the resource_name from list_corpora results.
        tool_context (ToolContext): The tool context

    Returns:
        dict: Information about the corpus and its files
    """
    try:
        # Initialize Vertex AI
        vertexai.init(project=PROJECT_ID, location=LOCATION)

        # Check if corpus exists
        if not check_corpus_exists(corpus_name, tool_context):
            return {
                "status": "error",
                "message": f"Corpus '{corpus_name}' does not exist",
                "corpus_name": corpus_name,
            }

        # Get the corpus resource name
        corpus_resource_name = get_corpus_resource_name(corpus_name)
        print(f"Corpus resource name: {corpus_resource_name}")

        # Try to get corpus details first
        corpus_display_name = corpus_name  # Default if we can't get actual display name

        try:
            corpus = rag.get_corpus(corpus_resource_name)
            if hasattr(corpus, "display_name") and corpus.display_name:
                corpus_display_name = corpus.display_name
        except Exception as corpus_error:
            print(f"Error getting corpus details: {str(corpus_error)}")
            # Just continue without corpus details
            pass

        # Process file information
        file_details = []

        try:
            # Get files in the corpus
            files = list(rag.list_files(corpus_resource_name))
            print(f"Found {len(files)} files")

            for file in files:
                file_info = {
                    "file_id": (
                        file.name.split("/")[-1] if hasattr(file, "name") else ""
                    ),
                    "source_uri": (
                        file.source_uri if hasattr(file, "source_uri") else ""
                    ),
                    "display_name": (
                        file.display_name if hasattr(file, "display_name") else ""
                    ),
                    "create_time": (
                        str(file.create_time) if hasattr(file, "create_time") else ""
                    ),
                    "update_time": (
                        str(file.update_time) if hasattr(file, "update_time") else ""
                    ),
                    "mime_type": file.mime_type if hasattr(file, "mime_type") else "",
                    "state": str(file.state) if hasattr(file, "state") else "",
                }
                file_details.append(file_info)

        except Exception as files_error:
            print(f"Error retrieving files: {str(files_error)}")
            return {
                "status": "error",
                "message": f"Error retrieving files for corpus '{corpus_name}': {str(files_error)}",
                "corpus_name": corpus_name,
                "corpus_resource_name": corpus_resource_name,
            }

        # Corpus statistics
        corpus_stats = {
            "file_count": len(file_details),
        }

        return {
            "status": "success",
            "message": f"Successfully retrieved information for corpus '{corpus_name}'",
            "corpus_name": corpus_name,
            "corpus_resource_name": corpus_resource_name,
            "display_name": corpus_display_name,
            "stats": corpus_stats,
            "files": file_details,
        }

    except Exception as e:
        print(f"Error in get_corpus_info: {str(e)}")
        return {
            "status": "error",
            "message": f"Error retrieving corpus information: {str(e)}",
            "corpus_name": corpus_name,
        }
