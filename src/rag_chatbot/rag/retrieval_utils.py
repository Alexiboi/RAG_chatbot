from datetime import datetime
from src.rag_chatbot.rag.embedding_utils import generate_embeddings
from src.rag_chatbot.rag.index_utils import TRANSCRIPT_SEARCH_CLIENT, MEETING_NOTES_SEARCH_CLIENT
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

def retrieve_context(query: str, k: int = FINAL_K) -> list[dict]:
    """
    As of now this returns context results using a cosine similarity + BM25 from the query string embedding & content.
    There could be better ways to do this.
    
    :param query: Description
    :type query: str
    :return: Description
    :rtype: tuple[str, str]
    """
    query_embedding = generate_embeddings([query])[0]

    filter_text = retrieve_filter(query)
    
    vector_query = VectorizedQuery(
        vector=query_embedding,
        k_nearest_neighbors=k,
        fields="embedding"
    )

    print(f"filter text: {filter_text}")
    
    
    transcript_filter = safe_filter_for_index(filter_text, "transcripts")
    meeting_filter = safe_filter_for_index(filter_text, "meeting_notes")

    transcript_results = TRANSCRIPT_SEARCH_CLIENT.search(
        #search_text=query,
        vector_queries=[vector_query],
        filter = transcript_filter,
        top=k
    )

    meeting_results = MEETING_NOTES_SEARCH_CLIENT.search(
        #search_text=query,
        vector_queries=[vector_query],
        filter = meeting_filter,
        top=k
    )

    # Convert to list and attach source label
    combined = []

    for r in transcript_results:
        r_dict = dict(r)
        r_dict["_index"] = "transcripts"
        r_dict["_score"] = r.get("@search.score", 0)
        combined.append(r_dict)

    for r in meeting_results:
        r_dict = dict(r)
        r_dict["_index"] = "meeting_notes"
        r_dict["_score"] = r.get("@search.score", 0)
        combined.append(r_dict)

    # Sort across both indexes by score
    combined_sorted = sorted(
        combined,
        key=lambda x: x["_score"],
        reverse=True
    )

    # Return final top k
    print(combined_sorted[:k])
    return combined_sorted[:k]

def safe_filter_for_index(filter_text: str, index_kind: str) -> str | None:
    """
    index_kind: "transcripts" or "meeting_notes"
    """
    if not filter_text:
        return None

    if index_kind == "transcripts":
        # transcripts support: docType, company, quarter, year (example)
        # If the filter is clearly meeting_note-specific, skip it for transcripts.
        if "author" in filter_text or "meetingDate" in filter_text:
            #return "docType eq 'earnings_call'"  # or None to not filter at all
            return None
        return filter_text

    if index_kind == "meeting_notes":
        # meeting_notes support: docType, author, meetingDate
        # If the filter is earnings_call-specific, skip it for meeting notes.
        if "company" in filter_text or "quarter" in filter_text or "year" in filter_text:
            #return "docType eq 'meeting_note'"  # or None
            return None
        return filter_text

    return None


def retrieve_filter(query: str) -> str:
    result = return_metadata(query)
    metadata = langextract_to_metadata(result)
    filter = build_filter(metadata)
    return filter

    

def return_metadata(query: str) -> str:
    """
    Extracts metadata fields and creates filter for azure ai search in order to narrow down retrieved documents
    
    :param query: Description
    :type query: str
    :return: Description
    :rtype: str
    """
    prompt = textwrap.dedent("""\
    You are an information extraction system for building Azure AI Search filters.

    TASK
    1) First, determine the document type and extract it as:
    - document_type = "earnings_call" OR "meeting_note"
    Choose exactly one. If unclear, pick the most likely based on the text.

    2) Then extract ONLY the fields required for that document_type:

    EARNINGS CALL (document_type="earnings_call")
    - company: the company name mentioned (exact text span)
    - quarter: the quarter if present (e.g., "Q1", "Q2", "Q3", "Q4")
    - year: the 4-digit year if present (e.g., "2024")

    MEETING NOTE (document_type="meeting_note")
    - author: the person who wrote the notes (exact text span)
    - meetingDate: the meeting date if present (exact text span as written if in YYYY/MM/DD form otherwise convert the date to this form. For example 28th January 2026 should be converted to 2026/01/28)

    EXTRACTION RULES
    - Use exact text from the input for extraction_text (no paraphrasing).
    - Do not invent values. If a field is not explicitly present, omit it.
    - Do not output fields that are not listed for the chosen document_type.
    - Do not overlap entities (each extracted span should be distinct).
    - Prefer the earliest mention in the text when multiple candidates exist.
    - Add a single meaningful normalized attribute for each extraction:
    * document type -> {"document_type": "..."}
    * company -> {"company": "<normalized short company name if obvious, else exact>"}
    * quarter -> {"quarter": "1|2|3|4"} (convert Q1->1 etc.)
    * year -> {"year": "YYYY"}
    * author -> {"author": "<exact or normalized if obvious>"}
    * meetingDate -> {"meetingDate": "<as written>"}\
    """)
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
        ),
        lx.data.ExampleData(
            text=(
                """
                From Reuben's meeting notes on the 28th January 2026 (28/01/2026),
                the team discussed upcoming product milestones, internal tooling improvements,
                and priorities for the next development sprint.
                """
            ),
            extractions=[
                lx.data.Extraction(
                    extraction_class="document type",
                    extraction_text="meeting notes",
                    attributes={"document_type": "meeting_note"},
                ),
                lx.data.Extraction(
                    extraction_class="author",
                    extraction_text="Reuben",
                    attributes={"author": "Reuben"},
                ),
                lx.data.Extraction(
                    extraction_class="meetingDate",
                    extraction_text="28th January 2026 (28/01/2026)",
                    attributes={"meetingDate": "28/01/2026"},
                ),
            ],
        ),
    ]
    # langextract uses the openAI api key not azure since langextract is incompatible with azure (for now)
    result = lx.extract(
        text_or_documents=query,
        prompt_description=prompt,
        examples=examples,
        model_id="gpt-4o-mini",
    )
    return result


def langextract_to_metadata(annotated_doc):
    """
    Convert LangExtract AnnotatedDocument into structured metadata dict.
    """

    metadata = {
        # earning call metadata
        "docType": None,
        "company": None,
        "year": None,
        "quarter": None,
        "reportDate": None,
        # Meeting notes metadata
        "meetingDate": None,
        "author": None

    }

    for extraction in annotated_doc.extractions:
        attrs = extraction.attributes or {}

        # document type
        if "document_type" in attrs:
            metadata["docType"] = attrs["document_type"]

        # company
        if "company" in attrs:
            metadata["company"] = attrs["company"]

        # year
        if "year" in attrs:
            try:
                metadata["year"] = int(attrs["year"])
            except:
                pass

        # quarter
        if "quarter" in attrs:
            try:
                metadata["quarter"] = int(attrs["quarter"])
            except:
                pass
        
        if "author" in attrs:
            try:
                metadata["author"] = attrs["author"]
            except:
                pass

        if "meetingDate" in attrs:
            try:
                print(attrs["meetingDate"])
                metadata["meetingDate"] = datetime.strptime(
                    attrs["meetingDate"], "%Y/%m/%d"
                ).isoformat() + "Z"
            except Exception as e:
                raise(e)
                

    # Derive reportDate deterministically
    if metadata["year"] and metadata["quarter"]:
        month = metadata["quarter"] * 3
        metadata["reportDate"] = datetime(
            metadata["year"], month, 1
        ).isoformat() + "Z"



    return metadata

def build_filter(meta):
    parts = []

    if meta["docType"]:
        parts.append(f"docType eq '{meta['docType']}'")
    if meta["company"]:
        parts.append(f"company eq '{meta['company']}'")
    if meta["year"]:
        parts.append(f"year eq {meta['year']}")
    if meta["quarter"]:
        parts.append(f"quarter eq '{meta['quarter']}'")
    if meta["author"]:
        parts.append(f"author eq '{meta['author']}'")
    if meta["meetingDate"]:
        parts.append(f"meetingDate eq {meta['meetingDate']}")

    return " and ".join(parts)

if __name__ == "__main__":
    print(retrieve_context("""
From Reuben's meeting notes on the 28th January 2026 (28/01/2026),
                the team discussed upcoming product milestones, internal tooling improvements,
                and priorities for the next development sprint."""))
    #retrieve_context("What commentary did Amazon provide about international profitability trends during the Amazon Q1 2024 Earnings Call?")
    
   


