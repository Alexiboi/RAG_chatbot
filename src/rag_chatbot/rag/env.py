import os
from dotenv import load_dotenv
from azure.storage.blob import ContainerClient
from openai import AzureOpenAI, OpenAI


load_dotenv('.env')

BLOB_SAS_URL = os.getenv("BLOB_SAS_URL")
AZURE_OPENAI_API_KEY=os.getenv("AZURE_OPENAI_API_KEY")
search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
admin_key = os.getenv("AZURE_SEARCH_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
azure_client_version = os.getenv("AZURE_OPENAI_API_VERSION")

AZURE_OPENAI_EMBEDDING_KEY = os.getenv("AZURE_OPENAI_API_EMBEDDING_KEY")

container_client = ContainerClient.from_container_url(BLOB_SAS_URL)

index_name = "transcript-chunks"
deployment_name = "gpt-5.2-chat"
vector_dimensions = 3072  # for text-embedding-3-large

client = OpenAI(
    base_url="https://alex-mltg6myf-eastus2.openai.azure.com/openai/v1/",
    api_key=AZURE_OPENAI_API_KEY
)

EMBEDDING_CLIENT = OpenAI(
    api_key=AZURE_OPENAI_EMBEDDING_KEY,
    
    base_url=f"https://transcript-embeds-openai.openai.azure.com/openai/v1",
)