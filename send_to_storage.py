from azure.storage.blob import BlobServiceClient, ContainerClient
import os

BLOB_SAS_URL = "https://alexcompanystore.blob.core.windows.net/transcripts?sp=racwdl&st=2025-11-26T19:37:55Z&se=2026-01-05T03:52:55Z&sv=2024-11-04&sr=c&sig=X9YE0qCIWWZg2uILNbUVkz7CM0CNnfA3fa5cER%2BM%2Bro%3D"
container_name = "transcripts"
folder_path = "test_files"  # virtual folder
local_file = "data/test1.txt"  # file to upload

container_client = ContainerClient.from_container_url(BLOB_SAS_URL)

blob_name = f"{folder_path}/{os.path.basename(local_file)}"

# --------------------------
# UPLOAD FILE
# --------------------------
with open(local_file, "rb") as data:
    container_client.upload_blob(
        name=blob_name,
        data=data,
        overwrite=True  # set to False to avoid overwriting
    )

print(f"Uploaded '{local_file}' to container '{container_name}' at path '{blob_name}'")

def format_transcript(transcript: str) -> str:
    pass

def send_to_storage(transcript: str) -> str:
    pass