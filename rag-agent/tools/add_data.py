"""
Tool for adding new data sources to a Vertex AI RAG corpus.
"""

import re
from typing import Dict, List

import vertexai
from google.adk.tools.tool_context import ToolContext
from vertexai import rag

from ..config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_REQUESTS_PER_MIN,
    LOCATION,
    PROJECT_ID,
)
from .utils import create_corpus_if_not_exists, get_corpus_resource_name


def add_data(
    corpus_name: str,
    paths: List[str],
    tool_context: ToolContext,
) -> Dict:
    """
    Add new data sources to a Vertex AI RAG corpus.
    If the specified corpus doesn't exist, it will be created automatically.

    Args:
        corpus_name (str): The name or full resource name of the corpus to add data to
        paths (List[str]): List of URLs or GCS paths to add to the corpus.
                          Supported formats:
                          - Google Drive: "https://drive.google.com/file/d/{FILE_ID}/view"
                          - Google Docs/Sheets/Slides: "https://docs.google.com/{type}/d/{FILE_ID}/..."
                          - Google Cloud Storage: "gs://{BUCKET}/{PATH}"
                          Example: ["https://drive.google.com/file/d/123", "gs://my_bucket/my_files_dir"]
        tool_context (ToolContext): The tool context

    Returns:
        dict: Information about the added data and status
    """
    try:
        # Initialize Vertex AI
        vertexai.init(project=PROJECT_ID, location=LOCATION)

        # Validate and convert URLs to the proper format
        validated_paths = []
        invalid_paths = []
        conversions = {}

        for path in paths:
            # Direct Google Drive file link - already valid
            if "drive.google.com/file/d/" in path:
                validated_paths.append(path)

            # Google Cloud Storage path - already valid
            elif path.startswith("gs://"):
                validated_paths.append(path)

            # Google Docs/Sheets/Slides links - extract ID and convert to Drive format
            elif "docs.google.com" in path:
                # Extract the document ID
                match = re.search(r"/d/([a-zA-Z0-9_-]+)", path)
                if match:
                    file_id = match.group(1)
                    drive_url = f"https://drive.google.com/file/d/{file_id}/view"
                    validated_paths.append(drive_url)
                    conversions[path] = drive_url
                else:
                    invalid_paths.append(path)

            # Not a recognized format
            else:
                invalid_paths.append(path)

        if not validated_paths:
            return {
                "status": "error",
                "message": "No valid paths provided. Please provide Google Drive URLs, Google Docs/Sheets/Slides URLs, or GCS paths.",
                "invalid_paths": invalid_paths,
            }

        # Check if corpus exists and create it if needed
        corpus_created = False
        corpus_result = create_corpus_if_not_exists(corpus_name, tool_context)
        if not corpus_result["success"]:
            return {
                "status": "error",
                "message": f"Unable to access or create corpus '{corpus_name}': {corpus_result['message']}",
                "corpus_name": corpus_name,
                "paths": paths,
            }

        corpus_created = corpus_result.get("was_created", False)

        # Get the corpus resource name
        corpus_resource_name = get_corpus_resource_name(corpus_name)

        # Set up chunking configuration
        transformation_config = rag.TransformationConfig(
            chunking_config=rag.ChunkingConfig(
                chunk_size=DEFAULT_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            ),
        )

        # Import files to the corpus
        import_result = rag.import_files(
            corpus_resource_name,
            validated_paths,
            transformation_config=transformation_config,
            max_embedding_requests_per_min=DEFAULT_EMBEDDING_REQUESTS_PER_MIN,
        )

        # Build the success message
        creation_msg = (
            f"Created new corpus '{corpus_name}' and " if corpus_created else ""
        )
        conversion_msg = ""
        if conversions:
            conversion_msg = " (Converted Google Docs URLs to Drive format)"

        return {
            "status": "success",
            "message": f"{creation_msg}Successfully added {import_result.imported_rag_files_count} file(s) to corpus '{corpus_name}'{conversion_msg}",
            "corpus_name": corpus_name,
            "corpus_created": corpus_created,
            "files_added": import_result.imported_rag_files_count,
            "paths": validated_paths,
            "invalid_paths": invalid_paths,
            "conversions": conversions,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error adding data to corpus: {str(e)}",
            "corpus_name": corpus_name,
            "paths": paths,
        }
