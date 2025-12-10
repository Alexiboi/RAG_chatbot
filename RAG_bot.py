from azure.storage.blob import ContainerClient
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
import re


load_dotenv('confidential.env')

BLOB_SAS_URL = os.getenv("BLOB_SAS_URL")
container_client = ContainerClient.from_container_url(BLOB_SAS_URL)

search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
admin_key = os.getenv("AZURE_SEARCH_KEY")
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
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url=f"https://transcript-embeds-openai.openai.azure.com/openai/v1"
)

vector_dimensions = 3072  # for text-embedding-3-large

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
                #vector_search_configuration="my-vector-config",
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
    You are an assistant that answers questions ONLY using the transcript context below.
    If the answer is not in the context, say exactly: "The transcript does not contain that information."

    Context:
    {context_block}

    User question:
    {user_query}

    Answer:
    """
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers strictly based on the provided context."},
            {"role": "user", "content": final_prompt}
        ],
        #temperature=0.2
    )

    return response.choices[0].message.content

def retrieve_context(query: str) -> tuple[str, str]:
    
    query_embedding = generate_embeddings([query])
    
    vector_query = VectorizedQuery(
        vector=query_embedding,
        k_nearest_neighbors=3,
        fields="embedding"
    )
    results = search_client.search(
        vector_queries=[vector_query]
    )
    return list(results)


def process_transcripts_from_blob(chunk_size: int=756) -> list[tuple[str, str]]: # default = 556
    """
    Returns transcripts chunks from each blob in the Transcripts
    Resource group.

    :chunk_size: size in bytes for chunk
    """
    # blobs should be a list of every blob in the transcript container
    blobs = container_client.list_blobs()
    transcript_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
    for blob in blobs:
        blob_client = container_client.get_blob_client(blob)
        download_stream = blob_client.download_blob()
        transcript_text = download_stream.readall().decode("utf-8")
        chunks = text_splitter.split_text(transcript_text)
        transcript_chunks.extend((blob.name, chunk) for chunk in chunks) # appends onto list from an iterable
    return transcript_chunks
        

def generate_embeddings(texts: list[str]) -> list[float]: 
    client = OpenAI(
        api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
        base_url="https://transcript-embeds-openai.openai.azure.com/openai/v1/"
    ) # https://transcript-embeds-openai.openai.azure.com/

    response = client.embeddings.create( 
        input = texts,
        model= "text-embedding-3-large"
    )
    return response.data[0].embedding
        
def process_and_store_chunks(chunks: list[(str, str)]):
    """
    Generate embeddings for chunks and store in vectorDB
    
    :param chunks: Description
    """
    # transcript_name can only contain letters, digits, _, -, =.
    documents = []
    for i, (transcript_name, chunk) in enumerate(chunks):
        chunk_blob_name = f"{transcript_name}-chunk-{i}"
        chunk_blob_name = re.sub(r"[/,_.]", "-", chunk_blob_name)

        embedding = generate_embeddings([chunk]) # chunk will only be one string so no need to turn it into a list
        upsert_data = {
            "id": chunk_blob_name,
            "content": chunk,
            "embedding": embedding
        }
    
        
        documents.append(upsert_data)
    result = search_client.upload_documents(documents=documents) # The document will be inserted if it is new and updated/replaced if it exists.
    return result

def main():
    print("Welcome to the RAG Chat application\n" \
    "Commands:\n" \
    "1. Chunk; generate embedding for chunk and store chunk with embedding in vectorDB for each transcript blob\n" \
    "2. query <your query> - Query the system\n" \
    "3. exit - Exit the application\n")
    while True:
        cmd = int(input("Enter 1, 2 or 3: "))
        if cmd == 1:
            chunks = process_transcripts_from_blob()
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

if __name__ == "__main__": 
    #create_index_schema()
    main()


