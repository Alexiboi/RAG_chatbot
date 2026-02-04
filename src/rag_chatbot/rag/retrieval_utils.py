from src.rag_chatbot.rag.embedding_utils import generate_embeddings
from src.rag_chatbot.rag.index_utils import search_client
from azure.search.documents.models import VectorizedQuery


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