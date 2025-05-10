import os
import sys

import vertexai
from dotenv import load_dotenv
from vertexai import rag

# Load environment variables
load_dotenv()

# Get project and location
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION")

print("==== Vertex AI Initialization Test ====")
print(f"PROJECT_ID: {PROJECT_ID}")
print(f"LOCATION: {LOCATION}")

# Check for required environment variables
if not PROJECT_ID or PROJECT_ID == "your-project-id":
    print(
        "ERROR: GOOGLE_CLOUD_PROJECT environment variable not set or contains default value."
    )
    print("Please update your .env file with your actual Google Cloud project ID.")
    sys.exit(1)

if not LOCATION or LOCATION == "your-location":
    print(
        "ERROR: GOOGLE_CLOUD_LOCATION environment variable not set or contains default value."
    )
    print("Please update your .env file with a valid location (e.g., us-central1).")
    sys.exit(1)

# Initialize Vertex AI
try:
    print(f"\nInitializing Vertex AI with project={PROJECT_ID}, location={LOCATION}")
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    print("✅ Vertex AI initialization successful")
except Exception as e:
    print(f"❌ ERROR initializing Vertex AI: {str(e)}")
    print("Please check your Google Cloud credentials and project settings.")
    sys.exit(1)

# Try to list corpora
try:
    print("\nListing corpora...")
    corpora = rag.list_corpora()

    # Convert to list to get an accurate count
    corpora_list = list(corpora)

    if not corpora_list:
        print("No corpora found. You may need to create one.")
    else:
        print(f"Found {len(corpora_list)} corpora:")
        for corpus in corpora_list:
            print(f"  - {corpus.display_name} (ID: {corpus.name})")

    print("✅ RAG functionality working correctly")
except Exception as e:
    print(f"❌ ERROR listing corpora: {str(e)}")
    print("This could indicate a permission issue or API configuration problem.")

print("\n==== Test Complete ====")
