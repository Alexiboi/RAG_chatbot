import os
from dotenv import load_dotenv
from azure.storage.blob import ContainerClient
from openai import OpenAI
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

TRANSCRIPT_SAS_URL = str(os.getenv("TRANSCRIPT_SAS_URL"))
MEETING_NOTE_SAS_URL = str(os.getenv("MEETING_NOTE_SAS_URL"))
AZURE_OPENAI_API_KEY=os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
admin_key = os.getenv("AZURE_SEARCH_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
azure_client_version = os.getenv("AZURE_OPENAI_API_VERSION")

AZURE_OPENAI_EMBEDDING_KEY = os.getenv("AZURE_OPENAI_API_EMBEDDING_KEY")

transcript_container_client = ContainerClient.from_container_url(TRANSCRIPT_SAS_URL)
notes_container_client = ContainerClient.from_container_url(MEETING_NOTE_SAS_URL)

#deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2-chat")
deployment_name = "o4-mini"

vector_dimensions = 3072  # for text-embedding-3-large

# client = OpenAI(
#     base_url=AZURE_OPENAI_ENDPOINT,
#     api_key=AZURE_OPENAI_API_KEY
# )

client = OpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    base_url=AZURE_OPENAI_ENDPOINT,
)

EMBEDDING_CLIENT = OpenAI(
    api_key=AZURE_OPENAI_EMBEDDING_KEY,
    base_url=f"https://transcript-embeds-openai.openai.azure.com/openai/v1",
)