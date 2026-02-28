
from collections import defaultdict
from typing import Any

from src.rag_chatbot.rag.env import EMBEDDING_CLIENT
from src.rag_chatbot.rag.index_utils import TRANSCRIPT_SEARCH_CLIENT, MEETING_NOTES_SEARCH_CLIENT, ensure_index_exists
import hashlib
import re
from datetime import datetime

COMPANY_MAP = {
    "a": "Agilent",
    "aapl": "Apple",
    "amzn": "Amazon",
    "bx": "BlackStone"
}

def generate_embeddings(texts: list[str]) -> list[float]: 

    response = EMBEDDING_CLIENT.embeddings.create( 
        input = texts,
        model= "text-embedding-3-large"
    )
    return [item.embedding for item in response.data]
    return response.data[0].embedding
        
def process_and_store_chunks(chunks: list[dict[str, Any]]):
    """
    chunks: [{"source": ..., "chunk_id": ..., "content": ..., "docType": ...}, ...]
    """

    # Group chunks by docType so we upload to the right index client
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ch in chunks:
        doc_type = ch.get("docType")
        if not doc_type:
            raise ValueError(f"Chunk missing 'docType': {ch}")
        grouped[doc_type].append(ch)

    results: dict[str, Any] = {}

    for doc_type, group_chunks in grouped.items():
        search_client = get_search_client_for_doc_type(doc_type)

        # check if index exists before attempting upload
        ensure_index_exists(search_client._index_name)

        texts = [ch["content"] for ch in group_chunks]
        embeddings = generate_embeddings(texts)  # must return list[list[float]]

        documents = []
        for ch, embedding in zip(group_chunks, embeddings):
            source_name = ch["source"]
            content = ch["content"]

            # Use a stable id for Azure Search. Don't use chunk_id alone.
            doc_id = make_chunk_id(source_name, content, doc_type)

            metadata = extract_metadata(source_name, doc_type)

            upsert_data = {
                "id": doc_id,
                "content": content,
                "embedding": embedding,
                "docType": doc_type,            # store explicitly in the index too
                **metadata,
            }
            documents.append(upsert_data)

        results[doc_type] = search_client.upload_documents(documents=documents)

    return results


def get_search_client_for_doc_type(doc_type: str):
    if doc_type == "transcript":
        return TRANSCRIPT_SEARCH_CLIENT
    if doc_type == "meeting_note":
        return MEETING_NOTES_SEARCH_CLIENT
    raise ValueError(f"Unknown doc_type: {doc_type}")

def make_chunk_id(source_name: str, chunk: str, doc_type: str) -> str:
    h = hashlib.sha1((doc_type + "\n" + source_name + "\n" + chunk).encode("utf-8")).hexdigest()[:16]
    base = re.sub(r"[^a-zA-Z0-9_\-=]", "-", source_name)  # stricter: replace anything unsafe
    return f"{doc_type}-{base}-{h}"

def extract_metadata(source_name: str, doc_type: str) -> dict:
    if doc_type == "transcript":
        return extract_earning_call_metadata(source_name)
    elif doc_type == "meeting_note":
        return extract_meeting_note_metadata(source_name)
    else:
        raise ValueError(f"Unknown doc_type: {doc_type}")

def extract_earning_call_metadata(transcript_name: str) -> dict:
    """
    Docstring for extract_metadata
    
    :param transcript_name: Description
    :type transcript_name: str
    :return: Description
    :rtype: dict
    """

    filename = transcript_name.split("/")[-1]

    # should match a filename like a-2024-2.txt
    pattern = r"^(?P<code>[a-z]+)-(?P<year>\d{4})-(?P<quarter>[1-4])\.txt$"

    match = re.match(pattern, filename)

    if not match:
        raise ValueError(f"Invalid transcript format: {transcript_name} and filename: {filename}")

    code = match.group("code") # code: e.g. a for agilent
    year = int(match.group("year"))
    quarter = int(match.group("quarter"))

    company = COMPANY_MAP.get(code)
    if not company:
        raise ValueError(f"Unknown company code: {code}")

    return {
        "docType": "earnings_call",
        "company": company,
        "year": year,
        "quarter": quarter,
        # optional: useful for date range filtering
        "reportDate": datetime(year, quarter * 3, 1).isoformat() + "Z"
    }

def extract_meeting_note_metadata(blob_name: str) -> dict:
    """
    Example filename pattern: 'meeting-notes/2026-01-28-john-notes.txt'
    Adjust to match your real naming convention.
    """
    filename = blob_name.split("/")[-1].lower()

    author = "Reuben"
    # Try to parse a YYYY-MM-DD date from the filename
    m = re.search(r"(?P<date>\d{4}-\d{2}-\d{2})", filename)
    meeting_date = None
    if m:
        meeting_date = m.group("date")  # store as string or ISO date

    return {
        "docType": "meeting_note",
        "meetingDate": meeting_date,  # can be None if missing
        "author": author
    }