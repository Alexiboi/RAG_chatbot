
from collections import defaultdict
from typing import Any
from src.rag_chatbot.rag.env import EMBEDDING_CLIENT
from src.rag_chatbot.rag.index_utils import TRANSCRIPT_SEARCH_CLIENT, MEETING_NOTES_SEARCH_CLIENT, ensure_index_exists
import hashlib
import re
from datetime import datetime
from azure.search.documents.models import IndexingResult
from azure.search.documents import SearchClient


COMPANY_MAP = {
    "a": "Agilent",
    "aapl": "Apple",
    "amzn": "Amazon",
    "bx": "BlackStone"
}

def generate_embeddings(texts: list[str]) -> list[list[float]]: 
    """
    Generate vector embeddings for a list of input texts using the embedding model.

    This function sends the input texts to the embedding API and returns a list
    of embedding vectors, where each vector corresponds to one input string.

    Args:
        texts (list[str]): list of texts, each item in the list is a string of text
    Returns:
        list[list[float]]: list of embedding vectors. Each vector in the list represents a word/non-space de-limited string in vector form.
    """

    response = EMBEDDING_CLIENT.embeddings.create( 
        input = texts,
        model= "text-embedding-3-large"
    )
    return [item.embedding for item in response.data]
        
def process_and_store_chunks(chunks: list[dict]) -> dict[str, list[IndexingResult]]:
    """
    Processes a list of text chunks then uploads them to an azure search index by:
    1. Grouping them by document type (docType)
    2. Generating embeddings for each chunk
    3. Preparing documents for Azure Search
    4. Uploading them to the appropriate search index
    5. Returning indexing results for each docType

    Args:
        chunks: [{"source": ..., "chunk_id": ..., "content": ..., "docType": ...}, ...]
    Returns:
        dict[str, list[IndexingResult]]:
            A dictionary where:
            - key = docType
            - value = list of indexing results returned from Azure Search
    """

    # Group chunks by docType so we upload to the right index client
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    # grouped is a dictionary where key=docType (earning_call, meeting_note)
    # and value = list of chunks belonging to that docType #
    for ch in chunks:
        doc_type = ch.get("docType")
        if not doc_type:
            raise ValueError(f"Chunk missing 'docType': {ch}")
        grouped[doc_type].append(ch)
    """
    Format of grouped: 
    grouped = {
        "earnings_call": [
            {...chunk1...},
            {...chunk2...}
        ],
        "meeting_note": [
            {...chunk3...}
        ]
    }
    """
    results: dict[str, list[IndexingResult]] = {}

    for doc_type, group_chunks in grouped.items():
        search_client = get_search_client_for_doc_type(doc_type)

        # check if index exists before attempting upload
        ensure_index_exists(search_client._index_name)

        # group_chunks should be a list of all the chunks that are belonging to a docType
        texts = [ch["content"] for ch in group_chunks]
        embeddings = generate_embeddings(texts)  # must return list[list[float]]


        """
        Each ch looks like:
        {
            "source": "...",
            "chunk_id": "...",
            "content": "...",
            "docType": "earnings_call"
        }
        """
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

    """
    Example of results:
    results = {
        "earnings_call": [
            {"key": "abc", "status": True},
            {"key": "def", "status": True}
        ],
        "meeting_note": [
            {"key": "xyz", "status": False, "errorMessage": "..."}
        ]
    }
    """
    # Check if any errors occurred during uploading of chunks:
    for doc_type, res_list in results.items():
        for r in res_list:
            if not r.succeeded:
                print(f"Failed to index {r.key}: {r.error_message}")

    return results


def get_search_client_for_doc_type(doc_type: str) -> SearchClient:
    """
    Retrieves search index for that chunks will be uploaded too, matching the search index passed into doc_type

    Args:
        doc_type (str): should be either 'transcript' or 'meeting_note' indicating the source of the chunk
    Returns:
        SearchClient: returns azure vector DB index search client corresponding to respective document type
    """
    if doc_type == "transcript":
        return TRANSCRIPT_SEARCH_CLIENT
    if doc_type == "meeting_note":
        return MEETING_NOTES_SEARCH_CLIENT
    raise ValueError(f"Unknown doc_type: {doc_type}")

def make_chunk_id(source_name: str, chunk: str, doc_type: str) -> str:
    """
    Generate a stable, unique identifier for a document chunk. If the chunk text, doc_type or source_name changes,
    the chunk_id will change.

    The ID is constructed using:
    - The document type
    - A sanitized version of the source name
    - A truncated SHA1 hash of (doc_type + source_name + chunk content)

    This ensures:
    - Deterministic IDs (same input → same ID)
    - Uniqueness across different chunks and document types
    - Safe characters for Azure Search (no invalid symbols)

    Args:
        source_name (str):
            Name or identifier of the source document (e.g. file name)

        chunk (str):
            The text content of the chunk

        doc_type (str):
            Type of the document (e.g. "earnings_call", "meeting_note")

    Returns:
        str:
            A unique, stable ID string suitable for use as a key in Azure Search.
            Format:
            "<docType>-<sanitized_source_name>-<hash>"

            Example:
            "earnings_call-report_q2_2024-1a2b3c4d5e6f7g8h"
    """
    h = hashlib.sha1((doc_type + "\n" + source_name + "\n" + chunk).encode("utf-8")).hexdigest()[:16]
    base = re.sub(r"[^a-zA-Z0-9_\-=]", "-", source_name)  # stricter: replace anything unsafe
    return f"{doc_type}-{base}-{h}"

def extract_metadata(source_name: str, doc_type: str) -> dict:
    """
    Route metadata extraction based on document type.

    This function acts as a dispatcher that selects the appropriate
    metadata extraction function depending on the provided doc_type.

    Supported document types:
    - "transcript" → earnings call metadata extraction
    - "meeting_note" → meeting note metadata extraction

    Args:
        source_name (str):
            The source identifier (e.g. filename or blob path) from which
            metadata will be extracted.

        doc_type (str):
            The type of document. Determines which extraction logic to use.

    Returns:
        dict:
            A dictionary containing structured metadata extracted from the source.
            The exact fields depend on the document type:
            
            - transcript:
                {
                    "docType": "earnings_call",
                    "company": str,
                    "year": int,
                    "quarter": int,
                    "reportDate": str
                }

            - meeting_note:
                {
                    "docType": "meeting_note",
                    "meetingDate": str | None,
                    "author": str | None
                }

    Raises:
        ValueError:
            If an unsupported doc_type is provided.
    """
    if doc_type == "transcript":
        return extract_earning_call_metadata(source_name)
    elif doc_type == "meeting_note":
        return extract_meeting_note_metadata(source_name)
    else:
        raise ValueError(f"Unknown doc_type: {doc_type}")

def extract_earning_call_metadata(transcript_name: str) -> dict:
    """
    Extract structured metadata from an earnings call transcript filename.

    The function expects filenames in the format:
        "<company_code>-<year>-<quarter>.txt"
    Example:
        "a-2024-2.txt" → Agilent, 2024, Q2

    It parses:
    - company code → mapped to full company name via COMPANY_MAP
    - year → integer
    - quarter → integer (1–4)

    Args:
        transcript_name (str):
            Full path or filename of the transcript (e.g. "folder/a-2024-2.txt")

    Returns:
        dict:
            Dictionary containing extracted metadata:
            {
                "docType": "earnings_call",
                "company": str,
                "year": int,
                "quarter": int,
                "reportDate": str (ISO 8601 format, e.g. "2024-06-01T00:00:00Z")
            }

    Raises:
        ValueError:
            - If filename does not match expected pattern
            - If company code is not found in COMPANY_MAP
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
    Extract metadata from a meeting notes filename.

    The function attempts to parse:
    - meeting date (YYYY-MM-DD format) from the filename
    - author 

    Example filename:
        "meeting-notes/2026-01-28-john-notes.txt"

    Args:
        blob_name (str):
            Full path or filename of the meeting notes file

    Returns:
        dict:
            Dictionary containing extracted metadata:
            {
                "docType": "meeting_note",
                "meetingDate": str | None,  # ISO date string if found
                "author": str | None
            }

    Notes:
        - If no date is found in the filename, meetingDate will be None
        - If no author is found in the filename, author will be None
    """
    filename = blob_name.split("/")[-1].lower()

    # extract date (independent)
    date_match = re.search(r"(?P<date>\d{4}-\d{2}-\d{2})", filename)
    meeting_date = date_match.group("date") if date_match else None

    # extract author (independent)
    # Look for "<date>-author-" OR "author-notes" OR author-<date>
    patterns = [
        r"(?:\d{4}-\d{2}-\d{2}-)?(?P<author>[a-z]+)-notes",   # date → author
        r"(?P<author>[a-z]+)-\d{4}-\d{2}-\d{2}-notes",        # author → date
    ]

    author = None
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            author = match.group("author")
            break

    return {
        "docType": "meeting_note",
        "meetingDate": meeting_date,  # can be None if missing
        "author": author
    }