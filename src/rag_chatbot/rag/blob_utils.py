from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_transcripts_from_blob(
        container_client,
        doc_type: str,
        chunk_size: int=756) -> list[dict]: # default = 556
    """
    Returns transcripts chunks from each blob in the Transcripts
    Resource group.

    :container_client: specifies which container client to extract blobs from
    :doc_type: specifies what type of document we are chunking from
    :chunk_size: size in characters for chunk
    Larger chunk size means less chunks in total.
    """
    # blobs should be a list of every blob in the transcript container
    blobs = sorted(container_client.list_blobs(), key=lambda b:b.name)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)

    transcript_chunks = []

    for blob in blobs:
        # include check to see if the uploaded file is a txt file
        if not blob.name.lower().endswith(".txt"):
            continue
        blob_client = container_client.get_blob_client(blob)
        download_stream = blob_client.download_blob()
        transcript_text = download_stream.readall().decode("utf-8")
        chunks = text_splitter.split_text(transcript_text)

        for j, chunk in enumerate(chunks):
            transcript_chunks.append({
                "source": blob.name,
                "chunk_id": j,
                "content": chunk,
                "docType": doc_type
            })

        #transcript_chunks.extend((blob.name, chunk) for chunk in chunks) # appends onto list from an iterable
    return transcript_chunks