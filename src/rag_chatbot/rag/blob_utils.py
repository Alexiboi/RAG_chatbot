from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.rag_chatbot.rag.env import container_client

def chunk_transcripts_from_blob(chunk_size: int=756) -> list[tuple[str, int, str]]: # default = 556
    """
    Returns transcripts chunks from each blob in the Transcripts
    Resource group.

    :chunk_size: size in characters for chunk
    Larger chunk size means less chunks in total.
    """
    # blobs should be a list of every blob in the transcript container
    blobs = sorted(container_client.list_blobs(), key=lambda b:b.name)

    transcript_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
    for blob in blobs:
        blob_client = container_client.get_blob_client(blob)
        download_stream = blob_client.download_blob()
        transcript_text = download_stream.readall().decode("utf-8")
        chunks = text_splitter.split_text(transcript_text)

        for j, chunk in enumerate(chunks):
            transcript_chunks.append((blob.name, j, chunk))

        #transcript_chunks.extend((blob.name, chunk) for chunk in chunks) # appends onto list from an iterable
    return transcript_chunks