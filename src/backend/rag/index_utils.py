from src.backend.rag.env import search_endpoint, admin_key, vector_dimensions
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError


# Index client manages the indexes that exist and allows new indexes to be added 
index_client = SearchIndexClient(
    endpoint=search_endpoint,
    credential=AzureKeyCredential(admin_key)
)

TRANSCRIPT_INDEX = "transcript-chunks"
MEETING_NOTES_INDEX = "meeting-notes"

# Search client manages uploading and querying documents in a specific index
# The specific index the searchClient operates on in this case is transcript-chunks
def make_search_client(index_name: str) -> SearchClient:
    """
    Create a SearchClient instance for interacting with a specific Azure AI Search index.

    This client is used for:
    - Uploading documents
    - Querying (keyword, vector, hybrid search)
    - Applying filters and retrieving results

    Args:
        index_name (str): Name of the Azure Search index to connect to

    Returns:
        SearchClient: Configured client for the given index
    """
    return SearchClient(
        endpoint=search_endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(admin_key),
    )

TRANSCRIPT_SEARCH_CLIENT = make_search_client(TRANSCRIPT_INDEX)
MEETING_NOTES_SEARCH_CLIENT = make_search_client(MEETING_NOTES_INDEX)

def ensure_index_exists(index_name: str) -> None:
    """
    Ensure that a given Azure Search index exists before performing operations.

    If the index does not exist:
    - Attempts to create/update all indexes
    - Raises an error to signal that setup was required

    Args:
        index_name (str): Name of the index to check

    Raises:
        RuntimeError: If the index does not exist and needs to be created
    """
    try:
        index_client.get_index(index_name)
    except ResourceNotFoundError:
        create_or_update_indexes()
        raise RuntimeError(
            f"Search index '{index_name}' does not exist. "
            f"Run create_index_schema() before ingesting."
        )


def create_or_update_indexes(transcripts: bool, index_name: str):
    """
    Create or update all required Azure Search indexes.

    This function:
    - Builds index schemas using helper functions
    - Sends them to Azure using create_or_update_index
    - Overwrites existing indexes if schema changes

    Useful for:
    - Initial setup
    - Schema migrations

    Side Effects:
        - Modifies Azure Search indexes
    """
    if transcripts:
        index_client.create_or_update_index(create_transcript_index_schema(index_name))
        print(f"{index_name} created/updated.")
    else:
        index_client.create_or_update_index(create_meeting_notes_index_schema(MEETING_NOTES_INDEX))
        print(f"{MEETING_NOTES_INDEX} created/updated.")
    




def delete_index_schema(index_name: str):
    """
    Delete an Azure Search index by name.

    Args:
        index_name (str): Name of the index to delete

    Notes:
        - Safe to call even if the index does not exist
        - Primarily used for development/testing resets
    """
    try:
        index_client.delete_index(index_name)
        print(f"Deleted index: {index_name}")
    except ResourceNotFoundError:
        print("Index did not exist, nothing to delete")

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

VECTOR_PROFILE = "my-vector-config"
ALGO_CONFIG = "my-algorithms-config"

def create_transcript_index_schema(index_name: str=TRANSCRIPT_INDEX) -> SearchIndex:
    """
    Define the schema for the transcript index used in Azure AI Search.

    This index stores:
    - Transcript text chunks
    - Vector embeddings for semantic search
    - Structured metadata for filtering (company, year, quarter, etc.)

    Fields:
        id (string): Unique identifier for each document chunk
        content (string): Transcript text (full-text searchable)
        embedding (vector): Dense vector for similarity search
        docType (string): Document type (e.g. "earnings_call")
        company (string): Company name
        year (int): Year of the transcript
        quarter (int): Quarter (1–4)
        reportDate (DateTimeOffset): Derived reporting date

    Vector Search:
        - Uses HNSW algorithm for approximate nearest neighbour search
        - Configured via VECTOR_PROFILE and ALGO_CONFIG

    Args:
        index_name (str): Name of the index

    Returns:
        SearchIndex: Fully defined index schema (not yet created in Azure)
    """
    return SearchIndex(
        name=index_name,
        fields=[
            SimpleField(name="id", type="Edm.String", key=True, filterable=True),
            SearchableField(name="content", type="Edm.String"),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=vector_dimensions,
                vector_search_profile_name=VECTOR_PROFILE
            ),
            # Metadata (filtering + faceting)
            SimpleField(
                name="docType",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
                sortable=True,
            ),
            SimpleField(
                name="company",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
                sortable=True,
            ),
            SimpleField(
                name="year",
                type=SearchFieldDataType.Int32,
                filterable=True,
                facetable=True,
                sortable=True,
            ),
            SimpleField(
                name="quarter",
                type=SearchFieldDataType.Int32,
                filterable=True,
                facetable=True,
                sortable=True,
            ),
            SimpleField(
                name="reportDate",
                type=SearchFieldDataType.DateTimeOffset,
                filterable=True,
                facetable=True,
                sortable=True,
            ),
        ],
        vector_search= VectorSearch(
            profiles=[VectorSearchProfile(name=VECTOR_PROFILE, algorithm_configuration_name=ALGO_CONFIG)],
            algorithms=[HnswAlgorithmConfiguration(name=ALGO_CONFIG)],
        )
       
    )

def create_meeting_notes_index_schema(index_name: str=MEETING_NOTES_INDEX) -> SearchIndex:
    """
    Define the schema for the meeting notes index used in Azure AI Search.

    This index stores:
    - Meeting note text
    - Vector embeddings for semantic search
    - Metadata specific to meetings (author, meeting date)

    Fields:
        id (string): Unique identifier
        content (string): Meeting note text
        docType (string): Document type (e.g. "meeting_note")
        meetingDate (DateTimeOffset): Date of the meeting
        author (string): Author of the notes
        embedding (vector): Semantic embedding for vector search

    Args:
        index_name (str): Name of the index

    Returns:
        SearchIndex: Fully defined index schema (not yet created in Azure)
    """
    return SearchIndex(
        name=index_name,
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
            SearchableField(name="content", type=SearchFieldDataType.String),

            SimpleField(name="docType", type=SearchFieldDataType.String, filterable=True, facetable=True, sortable=True),
            #SimpleField(name="source", type=SearchFieldDataType.String, filterable=True, sortable=True),
            #SimpleField(name="chunkIndex", type=SearchFieldDataType.Int32, filterable=True, sortable=True),

            # Meeting-note specific
            SimpleField(name="meetingDate", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
            SimpleField(name="author", type=SearchFieldDataType.String, filterable=True, facetable=True, sortable=True),

            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=vector_dimensions,
                vector_search_profile_name=VECTOR_PROFILE,
            ),
        ],
        vector_search=VectorSearch(
            profiles=[VectorSearchProfile(name=VECTOR_PROFILE, algorithm_configuration_name=ALGO_CONFIG)],
            algorithms=[HnswAlgorithmConfiguration(name=ALGO_CONFIG)],
        ),
    )


if __name__ == "__main__":
    #delete_index_schema(TRANSCRIPT_INDEX)
    #create_or_update_indexes(True, TRANSCRIPT_INDEX)
    index = index_client.get_index(TRANSCRIPT_INDEX)
    for config in index.semantic_search.configurations:
        print(config.name)
    pass