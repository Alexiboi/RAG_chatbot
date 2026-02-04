from azure.storage.blob import ContainerClient
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from azure.core.exceptions import ResourceNotFoundError
import re
import hashlib


load_dotenv('.env')

BLOB_SAS_URL = os.getenv("BLOB_SAS_URL")
AZURE_OPENAI_API_KEY=os.getenv("AZURE_OPENAI_API_KEY")
search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
admin_key = os.getenv("AZURE_SEARCH_KEY")

container_client = ContainerClient.from_container_url(BLOB_SAS_URL)

index_name = "transcript-chunks"

# Index client manages the indexes that exist and allows new indexes to be added 
index_client = SearchIndexClient(
    endpoint=search_endpoint,
    credential=AzureKeyCredential(admin_key)
)

# Search client manages uploading and querying documents in a specific index
# The specific index the searchClient operates on in this case is transcript-chunks
search_client = SearchClient(
    endpoint=search_endpoint,
    index_name="transcript-chunks",
    credential=AzureKeyCredential(admin_key)
)


deployment_name = "o4-mini"
client = OpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    base_url=f"https://transcript-embeds-openai.openai.azure.com/openai/v1"
)

vector_dimensions = 3072  # for text-embedding-3-large


def delete_index_schema():
    try:
        index_client.delete_index(index_name)
        print(f"Deleted index: {index_name}")
    except ResourceNotFoundError:
        print("Index did not exist, nothing to delete")


def create_index_schema():
    """"
    Creates an index called transcript-chunks with fields of id, content and the embedding as a vector field.
    Content is the text content of the transcript which is searchable.
    If the index already exists it will update the index if any changes are made.
    """
    from azure.search.documents.indexes.models import (
        SearchIndex,
        SearchField,
        SearchFieldDataType,
        SimpleField,
        SearchableField,
        VectorSearch,
        VectorSearchProfile,
        HnswAlgorithmConfiguration,
    )
    
    index_schema = SearchIndex(
        name=index_name,
        fields=[
            SimpleField(name="id", type="Edm.String", key=True, filterable=True),
            SearchableField(name="content", type="Edm.String"),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=vector_dimensions,
                vector_search_profile_name="my-vector-config"
            )
        ],
        vector_search= VectorSearch(
            profiles=[VectorSearchProfile(name="my-vector-config", algorithm_configuration_name="my-algorithms-config")],
            algorithms=[HnswAlgorithmConfiguration(name="my-algorithms-config")],
        )
       
    )

    index_client.create_or_update_index(index_schema)
    print("Index created.")


def generate_response(context, user_query) -> str:
    context_texts = [doc["content"] for doc in context]

    context_block = "\n\n---\n\n".join(context_texts)
    final_prompt = f"""
    You are an assistant that answers questions using the transcript context below.
    If the answer is not in the context, say that the transcript does not contain the information.

    Context:
    {context_block}

    User question:
    {user_query}

    Answer:
    """
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers based on the provided context."},
            {"role": "user", "content": final_prompt}
        ],
        #temperature=0.2
    )

    return response.choices[0].message.content


def retrieve_context(query: str) -> tuple[str, str]:
    """
    As of now this returns context results using a cosine similarity from the query string embedding.
    There could be better ways to do this.
    
    :param query: Description
    :type query: str
    :return: Description
    :rtype: tuple[str, str]
    """
    query_embedding = generate_embeddings([query])
    
    vector_query = VectorizedQuery(
        vector=query_embedding,
        k_nearest_neighbors=6,
        fields="embedding"
    )
    results = search_client.search(
        vector_queries=[vector_query]
    )
    return list(results)


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


def generate_embeddings(texts: list[str]) -> list[float]: 
    client = OpenAI(
        api_key = AZURE_OPENAI_API_KEY,
        base_url="https://transcript-embeds-openai.openai.azure.com/openai/v1/"
    ) 

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

def main():
    print("Welcome to the RAG Chat application\n" \
    "Commands:\n" \
    "1. Chunk; generate embedding for chunk and store chunk with embedding in vectorDB for each transcript blob\n" \
    "2. query <your query> - Query the system\n" \
    "3. exit - Exit the application\n")
    while True:
        cmd = int(input("Enter 1, 2 or 3: "))
        if cmd == 1:
            chunks = chunk_transcripts_from_blob()
            process_and_store_chunks(chunks)
        elif cmd == 2:
            user_query = input("Enter your user query").strip()
            context_results = retrieve_context(user_query)
            for result in context_results:
                print(result["id"])
                
            answer = generate_response(context_results, user_query)
            print(answer)
        elif cmd == 3:
            return
        else:
            print("Invalid command try again") 

main()

def generate_contextualized_response(user_query):
    user_query = user_query.strip()
    context_results = retrieve_context(user_query)
    answer = generate_response(context_results, user_query)
    return answer




