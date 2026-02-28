from src.rag_chatbot.rag.env import search_endpoint, admin_key, vector_dimensions
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
    return SearchClient(
        endpoint=search_endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(admin_key),
    )

TRANSCRIPT_SEARCH_CLIENT = make_search_client(TRANSCRIPT_INDEX)
MEETING_NOTES_SEARCH_CLIENT = make_search_client(MEETING_NOTES_INDEX)

def ensure_index_exists(index_name: str):
    try:
        index_client.get_index(index_name)
    except ResourceNotFoundError:
        create_or_update_indexes()
        raise RuntimeError(
            f"Search index '{index_name}' does not exist. "
            f"Run create_index_schema() before ingesting."
        )


def create_or_update_indexes():
    index_client.create_or_update_index(create_transcript_index_schema(TRANSCRIPT_INDEX))
    print(f"{TRANSCRIPT_INDEX} created/updated.")
    index_client.create_or_update_index(create_meeting_notes_index_schema(MEETING_NOTES_INDEX))
    print(f"{MEETING_NOTES_INDEX} created/updated.")
    




def delete_index_schema(index_name: str=TRANSCRIPT_INDEX):
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
    """"
    Creates an index called transcript-chunks with fields of id, content and the embedding as a vector field.
    Content is the text content of the transcript which is searchable.
    If the index already exists it will update the index if any changes are made.
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
    delete_index_schema()