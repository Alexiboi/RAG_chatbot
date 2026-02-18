
from src.rag_chatbot.rag.env import client
from src.rag_chatbot.rag.index_utils import search_client
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

    response = client.embeddings.create( 
        input = texts,
        model= "text-embedding-3-large"
    )
    return response.data[0].embedding
        
def process_and_store_chunks(chunks: list[(str, int, str)]):
    """
    Generate embeddings for chunks and store in vectorDB
    
    :param chunks: Description
    """
    # transcript_name can only contain letters, digits, _, -, =.
    documents = []
    for transcript_name, j, chunk in chunks:

        chunk_id = f"{make_chunk_id(transcript_name, chunk)}"
        embedding = generate_embeddings([chunk]) # chunk will only be one string so no need to turn it into a list

        metadata = extract_metadata(transcript_name)
        upsert_data = {
            "id": chunk_id,
            "content": chunk,
            "embedding": embedding,
            **metadata # unpack metadata into the document
        }
    
        
        documents.append(upsert_data)
    result = search_client.upload_documents(documents=documents) # The document will be inserted if it is new and updated/replaced if it exists.
    return result

def make_chunk_id(transcript_name: str, chunk: str) -> str:
    h = hashlib.sha1((transcript_name + "\n" + chunk).encode("utf-8")).hexdigest()[:16]
    base = re.sub(r"[/,_.]", "-", transcript_name)
    return f"{base}-{h}"

def extract_metadata(transcript_name: str) -> dict:
    """
    Docstring for extract_metadata
    
    :param transcript_name: Description
    :type transcript_name: str
    :return: Description
    :rtype: dict
    """

    filename = transcript_name.split("/")[-1]

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