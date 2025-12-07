from azure.storage.blob import BlobServiceClient, ContainerClient
import os
import pandas as pd 
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, VectorField, SearchableField,
    HnswAlgorithmConfiguration, HnswParameters
)

load_dotenv('confidential.env')

BLOB_SAS_URL = os.getenv("BLOB_SAS_URL")
container_client = ContainerClient.from_container_url(BLOB_SAS_URL)

search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
admin_key = os.getenv("AZURE_SEARCH_KEY")
index_name = "transcript-chunks"

index_client = SearchIndexClient(
    endpoint=search_endpoint,
    credential=AzureKeyCredential(admin_key)
)

search_client = SearchClient(
    endpoint=search_endpoint,
    index_name="transcript-chunks",
    credential=AzureKeyCredential(admin_key)
)

vector_dimensions = 3072  # for text-embedding-3-large

def create_index_schema():
    """"
    Creates an index called transcript-chunks with fields if id, content and the embedding as a vector field.
    Content is the text content of the transcript which is searchable
    """
    index_schema = SearchIndex(
        name=index_name,
        fields=[
            SimpleField(name="id", type="Edm.String", key=True, filterable=True),
            SearchableField(name="content", type="Edm.String"),
            VectorField(
                name="embedding",
                searchable=True,
                dimensions=vector_dimensions,
                vector_search_configuration="my-vector-config"
            )
        ],
        vector_search={
            "algorithm_configurations": [
                HnswAlgorithmConfiguration(
                    name="my-vector-config",
                    parameters=HnswParameters(
                        m=48,
                        ef_construction=400,
                    )
                )
            ]
        }
    )

    index_client.create_or_update_index(index_schema)
    print("Index created.")



def main():
    pass

def process_transcripts_from_blob(chunk_size=256):
    """"""
    # blobs should be a list of every blob in the transcript container
    blobs = container_client.list_blobs()
    transcript_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
    for blob in blobs:
        blob_client = container_client.get_blob_client(blob)
        download_stream = blob_client.download_blob()
        transcript_text = download_stream.readall().decode("utf-8")
        chunks = text_splitter.split_text(transcript_text)
        transcript_chunks.extend((blob.name, chunk) for chunk in chunks) # appends onto list from an iterable
    return transcript_chunks
        

def generate_embeddings(texts):
    client = OpenAI(
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
    base_url="https://transcript-embeds-openai.openai.azure.com/"
    )

    response = client.embeddings.create(
        input = texts,
        model= "text-embedding-3-large"
    )
    
    return [item.embedding for item in response.data]
        
# Generate embeddings for chunks and store in vectorDB
def process_and_store_chunks(chunks):
    for transcript_name, chunk in chunks:
        chunk_blob_name = f"{transcript_name}-chunk-{chunks.index((transcript_name, chunk))}.txt"
        embedding = generate_embeddings([chunk])
        upsert_data = [
            {"id": chunk_blob_name,
            "content": chunk,
            "embedding": embedding["embedding"]
            }
            for i, embedding in enumerate(embedding)
        ]
        result = search_client.upload_documents(documents=upsert_data) # The document will be inserted if it is new and updated/replaced if it exists.
    return result
