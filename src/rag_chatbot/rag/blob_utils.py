from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.rag_chatbot.rag.LLMChunker import LLMChunker
import re



def contextual_chunking(text: str, chunks: list[str]):
    """
    Appends context along with chunk text to provide better search results

    :text: entire text document e.g. one meeting transcript
    :chunks: list of chunks without context attached
    Larger chunk size means less chunks in total.
    """

    llm_chunker = LLMChunker()
    contextual_chunks = []
    
    for c in chunks:
        context = llm_chunker.return_response(document=text, chunk=c)
        contextual_chunks.append(context + " " + c)
    
    return contextual_chunks

def chunk_epics(meeting_notes: str) -> List[str]:
    """
    Split meeting notes into chunks where each chunk corresponds to one Epic.

    A chunk starts at a line like:
        Epic 1: Title
    and continues until the next Epic heading or the end of the document.

    Returns:
        List of dictionaries containing:
        - epic_title: the heading line
        - content: full chunk text for that epic
    """

    # Matches lines like:
    # Epic 1:
    # Epic 2: Dashboard
    # Epic 10 - Some title   (if you want to support '-' too)
    epic_pattern = re.compile(
        r'^(Epic\s+\d+\s*[:\-].*)$',
        re.MULTILINE | re.IGNORECASE
    )

    matches = list(epic_pattern.finditer(meeting_notes))

    if not matches:
        return []

    chunks = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(meeting_notes)

        chunk_text = meeting_notes[start:end].strip()
        epic_title = match.group(1).strip()

        # chunks.append({
        #     "epic_title": epic_title,
        #     "content": chunk_text
        # })
        chunks.append(epic_title + chunk_text)

    return chunks

def chunk_from_blob(
        container_client,
        doc_type: str,
        chunk_size: int=756,
        context_chunking: bool=False,
        overlap: bool=False,
        epic_chunking: bool=False) -> list[dict]: 
    """
    Returns transcripts chunks from each blob in the Transcripts
    Resource group.

    :container_client: specifies which container client to extract blobs from
    :doc_type: specifies what type of document we are chunking from
    :chunk_size: size in characters for chunk
    Larger chunk size means less chunks in total.
    """

    overlap = int(chunk_size * 0.1) if overlap else 0 # overlap should be 10-20% of chunks size

    # blobs should be a list of every blob in the transcript container
    blobs = sorted(container_client.list_blobs(), key=lambda b:b.name)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap)

    transcript_chunks = []

    for blob in blobs:
        # include check to see if the uploaded file is a txt file
        if not blob.name.lower().endswith(".txt"):
            continue
        blob_client = container_client.get_blob_client(blob)
        download_stream = blob_client.download_blob()
       
        transcript_text = download_stream.readall().decode("utf-8")
        # prefer to use agentic chunking with structured data like meeting notes
        

        chunks = text_splitter.split_text(transcript_text)

        if context_chunking:
            # replace chunks with contextual chunks
            chunks = contextual_chunking(text=transcript_text, chunks=chunks)

        if epic_chunking:
            epic_chunks = chunk_epics(transcript_text)
            chunks.extend(epic_chunks)

        for j, chunk in enumerate(chunks):
            transcript_chunks.append({
                "source": blob.name,
                "chunk_id": j,
                "content": chunk,
                "docType": doc_type
            })

    return transcript_chunks
