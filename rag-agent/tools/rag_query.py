"""
Tool for querying Vertex AI RAG corpora and retrieving relevant information.
"""

from typing import Dict

import vertexai
from google.adk.tools.tool_context import ToolContext
from vertexai import rag

from ..config import (
    DEFAULT_DISTANCE_THRESHOLD,
    DEFAULT_TOP_K,
    LOCATION,
    PROJECT_ID,
)
from .utils import create_corpus_if_not_exists, get_corpus_resource_name


def rag_query(
    corpus_name: str,
    query: str,
    tool_context: ToolContext,
) -> Dict:
    """
    Query a Vertex AI RAG corpus with a user question and return relevant information.
    If the specified corpus doesn't exist, it will be created automatically.

    Args:
        corpus_name (str): The full resource name of the corpus to query.
                           Preferably use the resource_name from list_corpora results.
        query (str): The text query to search for in the corpus
        tool_context (ToolContext): The tool context

    Returns:
        dict: The query results and status
    """
    try:
        # Initialize Vertex AI
        vertexai.init(project=PROJECT_ID, location=LOCATION)

        # Check if corpus exists and create it if needed
        corpus_result = create_corpus_if_not_exists(corpus_name, tool_context)
        if not corpus_result["success"]:
            return {
                "status": "error",
                "message": f"Unable to access or create corpus '{corpus_name}': {corpus_result['message']}",
                "query": query,
                "corpus_name": corpus_name,
            }

        # If corpus was created, there's no data to query yet
        if corpus_result.get("was_created", False):
            return {
                "status": "warning",
                "message": f"Created a new corpus '{corpus_name}', but it doesn't contain any data yet. Please add data to the corpus before querying.",
                "query": query,
                "corpus_name": corpus_name,
                "results": [],
                "results_count": 0,
            }

        # Get the corpus resource name
        corpus_resource_name = get_corpus_resource_name(corpus_name)

        # Configure retrieval parameters
        rag_retrieval_config = rag.RagRetrievalConfig(
            top_k=DEFAULT_TOP_K,
            filter=rag.Filter(vector_distance_threshold=DEFAULT_DISTANCE_THRESHOLD),
        )

        # Perform the query
        response = rag.retrieval_query(
            rag_resources=[
                rag.RagResource(
                    rag_corpus=corpus_resource_name,
                )
            ],
            text=query,
            rag_retrieval_config=rag_retrieval_config,
        )

        # Process the response into a more usable format
        results = []
        if hasattr(response, "contexts") and response.contexts:
            for ctx_group in response.contexts.contexts:
                result = {
                    "source_uri": (
                        ctx_group.source_uri if hasattr(ctx_group, "source_uri") else ""
                    ),
                    "source_name": (
                        ctx_group.source_display_name
                        if hasattr(ctx_group, "source_display_name")
                        else ""
                    ),
                    "text": ctx_group.text if hasattr(ctx_group, "text") else "",
                    "score": ctx_group.score if hasattr(ctx_group, "score") else 0.0,
                }
                results.append(result)

        # If we didn't find any results
        if not results:
            return {
                "status": "warning",
                "message": f"No results found in corpus '{corpus_name}' for query: '{query}'",
                "query": query,
                "results": [],
                "results_count": 0,
            }

        return {
            "status": "success",
            "message": f"Successfully queried corpus '{corpus_name}'",
            "query": query,
            "results": results,
            "results_count": len(results),
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error querying corpus: {str(e)}",
            "query": query,
            "corpus_name": corpus_name,
        }
