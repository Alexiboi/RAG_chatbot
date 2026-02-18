from src.rag_chatbot.rag.env import search_endpoint, admin_key, index_name, vector_dimensions
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError


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
            profiles=[VectorSearchProfile(name="my-vector-config", algorithm_configuration_name="my-algorithms-config")],
            algorithms=[HnswAlgorithmConfiguration(name="my-algorithms-config")],
        )
       
    )

    index_client.create_or_update_index(index_schema)
    print("Index created.")

if __name__ == "__main__":
    create_index_schema()