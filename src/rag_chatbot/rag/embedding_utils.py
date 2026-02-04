
from src.rag_chatbot.rag.env import client
from src.rag_chatbot.rag.index_utils import search_client
import hashlib
import re

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
        upsert_data = {
            "id": chunk_id,
            "content": chunk,
            "embedding": embedding
        }
    
        
        documents.append(upsert_data)
    result = search_client.upload_documents(documents=documents) # The document will be inserted if it is new and updated/replaced if it exists.
    return result

def make_chunk_id(transcript_name: str, chunk: str) -> str:
    h = hashlib.sha1((transcript_name + "\n" + chunk).encode("utf-8")).hexdigest()[:16]
    base = re.sub(r"[/,_.]", "-", transcript_name)
    return f"{base}-{h}"