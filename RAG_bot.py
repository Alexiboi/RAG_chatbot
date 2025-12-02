from azure.storage.blob import BlobServiceClient, ContainerClient
import os
import pandas as pd 
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv('confidential.env')

BLOB_SAS_URL = os.getenv("BLOB_SAS_URL")
container_client = ContainerClient.from_container_url(BLOB_SAS_URL)

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
        transcript_chunks.extend(chunks) # appends onto list from an iterable
    return transcript_chunks
        

        
        

