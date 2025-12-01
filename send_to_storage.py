from azure.storage.blob import BlobServiceClient, ContainerClient
import os
import pandas as pd 

BLOB_SAS_URL = "https://alexcompanystore.blob.core.windows.net/transcripts?sp=racwdl&st=2025-11-26T19:37:55Z&se=2026-01-05T03:52:55Z&sv=2024-11-04&sr=c&sig=X9YE0qCIWWZg2uILNbUVkz7CM0CNnfA3fa5cER%2BM%2Bro%3D"
container_name = "transcripts"
folder_path = "earnings_calls"  # virtual folder
local_file = "data/test1.txt"  # file to upload

container_client = ContainerClient.from_container_url(BLOB_SAS_URL)

blob_name = f"{folder_path}/{os.path.basename(local_file)}"


def format_transcript(transcript: str) -> str:
    """Method should return transcript in a format similar to that of 
    a .vtt file transcript from Teams to allow swapping of data in the future
    format:
    00:00:23.123 --> 00:00:28.843 (time speaker speaks for)
    <v Alex Hanna (speaker name))>This is a test for meeting transcriptions.
    This is about to be over.</v>"""
    pass

def add_meta_data():
    pass

def send_to_storage(transcript: str) -> str:
    pass

def read_in_transcript(url: str) -> pd.DataFrame:
    df = pd.read_json(url, lines=True)
    #df.to_csv("data/2024-earnings-call-transcript.csv", index=False)
    return df

#read_in_transcript("hf://datasets/yeong-hwan/2024-earnings-call-transcript/2024-earnings-call-transcripts.jsonl")