from src.rag_chatbot.rag.embedding_utils import generate_embeddings
from src.rag_chatbot.rag.index_utils import search_client
from azure.search.documents.models import VectorizedQuery
from sentence_transformers import CrossEncoder
import textwrap
import langextract as lx
from src.rag_chatbot.rag.env import deployment_name, client
from sentence_transformers import CrossEncoder
import os
from langextract import factory
import langextract_azureopenai

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")



K=30
FINAL_K = 6



def rerank(query, candidates, final_top_k: int = FINAL_K):
    texts = []
    idx_map = []  # maps rank-list index -> candidate index
    for i, c in enumerate(candidates):
        text = (c.get("content") or "").strip()
        if not text:
            continue
        texts.append(text)
        idx_map.append(i)

    if not texts:
        return []

    # 2) Rank
    # returns items in ranked order; when return_documents=True, it includes the text
    ranked = model.rank(query, texts, return_documents=True)

    ranked_texts = []
    if isinstance(ranked, dict) and "documents" in ranked:
        # sometimes { "documents": [...], "scores": [...] }
        ranked_texts = ranked["documents"]
    elif isinstance(ranked, list):
        # often list of {"text": "...", "score": ...} OR list[str]
        if ranked and isinstance(ranked[0], dict):
            # common: [{"text": "...", "score": ...}, ...]
            ranked_texts = [r.get("text") or r.get("document") or r.get("passage") for r in ranked]
        else:
            # could be list[str]
            ranked_texts = ranked
    else:
        # fallback
        ranked_texts = []

    # 4) Map ranked texts back to candidate dicts
    # If there are duplicates, this picks the first unused match.
    text_to_candidate_idxs = {}
    for j, t in enumerate(texts):
        text_to_candidate_idxs.setdefault(t, []).append(j)

    used = set()
    reranked_candidates = []
    for t in ranked_texts:
        if not t:
            continue
        if t not in text_to_candidate_idxs:
            continue
        # pick first occurrence not used
        for local_j in text_to_candidate_idxs[t]:
            if local_j in used:
                continue
            used.add(local_j)
            original_i = idx_map[local_j]
            reranked_candidates.append(candidates[original_i])
            break

        if len(reranked_candidates) >= final_top_k:
            break

    return reranked_candidates

def retrieve_context(query: str, k: int = K) -> list[dict]:
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
        k_nearest_neighbors=k,
        fields="embedding"
    )
    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        top=FINAL_K
    )

    candidates = list(results)
    
    return candidates


    

def return_filter(query: str) -> str:
    """
    Extracts metadata fields and creates filter for azure ai search in order to narrow down retrieved documents
    
    :param query: Description
    :type query: str
    :return: Description
    :rtype: str
    """
    prompt = textwrap.dedent("""\
    Extract document type (which can be one from earnings_call, policy_document or case_study), company, year, quarter and relationships in order of appearance.
    Use exact text for extractions. Do not paraphrase or overlap entities.
    Provide meaningful attributes for each entity to add context.""")
    examples = [
        lx.data.ExampleData(
            text=(
                """
            During the Agilent Technologies Q2 2024 Earnings Call,
            which executives were present for prepared remarks and which 
            leaders were joining specifically for the Q&A portion?"""
            ),
            extractions=[
                lx.data.Extraction(
                    extraction_class="document type",
                    extraction_text="Earnings Call",
                    attributes={"document_type": "earnings_call"},
                ),
                lx.data.Extraction(
                    extraction_class="company",
                    extraction_text="Agilent Technologies",
                    attributes={"company": "Agilent"},
                ),
                lx.data.Extraction(
                    extraction_class="quarter",
                    extraction_text="Q2",
                    attributes={"quarter": "2"},
                ),
                lx.data.Extraction(
                    extraction_class="year",
                    extraction_text="2024",
                    attributes={"year": "2024"},
                ),
            ],
        )
        
    ]
    config = lx.factory.ModelConfig(
        model_id=deployment_name,  # <-- your Azure *deployment name*
        provider="AzureOpenAILanguageModel",
        provider_kwargs={
            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
        },
    )
    # langextract uses the openAI api key not azure since langextract is incompatible with azure (for now)
    result = lx.extract(
        text_or_documents=query,
        prompt_description=prompt,
        examples=examples,
        model_id="gpt-4o-mini"
        
    )
    print(result)


if __name__ == "__main__":
    #retrieve_context("What commentary did Amazon provide about international profitability trends during the Amazon Q1 2024 Earnings Call?")
    return_filter("""In the Apple Q3 2022 earnings call,
            which executives were present for prepared remarks and which 
            leaders were joining specifically for the Q&A portion?""")
   


