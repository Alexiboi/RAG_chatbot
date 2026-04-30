from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field
from src.backend.rag.embedding_utils import generate_embeddings
from src.backend.rag.env import client, deployment_name
from src.backend.rag.index_utils import TRANSCRIPT_SEARCH_CLIENT, MEETING_NOTES_SEARCH_CLIENT
from azure.search.documents.models import VectorizedQuery
import textwrap
import langextract as lx
from langextract.core.data import AnnotatedDocument

FINAL_K = 6

class RetrievalRoute(BaseModel):
    source: Literal["transcripts", "meeting_notes", "both"] = Field(
        description="Which index should be searched for this user query."
    )



def get_routing_prompt(query: str) -> str:
    """
    Returns prompt used by LLM that decides route

    Args:
        query (str): query that is placed into prompt to send to LLm

    Returns:
        str: prompt with query embedded
    """
    return textwrap.dedent(f"""\
    You are a retrieval router for an internal knowledge assistant.

    Your task is to decide which document source should be searched for the user's query.

    Available sources:
    1. transcripts
       - earnings calls
       - company/financial discussions
       - executives, prepared remarks, Q&A
       - company, quarter, year based queries

    2. meeting_notes
       - internal meeting notes
       - notes written by a person/author
       - project updates, sprint planning, tooling discussions, product milestones
       - author and meeting date based queries

    Routing rules:
    - Return "transcripts" if the query is primarily about earnings calls, companies, quarters, years, executives, or call discussions.
    - Return "meeting_notes" if the query is primarily about internal notes, project updates, sprint planning, milestones, tooling, an author, or a meeting date.
    - Return "both" only if the query is genuinely ambiguous or could reasonably refer to either source.

    User query:
    {query}
    """)

def route_query(query: str) -> RetrievalRoute:
    """
    Calls LLM with specific prompt that determines which index documents are retrieved from (earnings calls, meeting notes or both)

    Args:
        query (str): User's query
    
    Returns:
        RetrievalRoute: Object which has attribute source which is one of "transcripts", "meeting_notes", "both"
    """
    prompt = get_routing_prompt(query)

    response = client.responses.parse(
        model=deployment_name, # gpt-5.2-chat
        input=prompt,
        text_format=RetrievalRoute,
    )

    return response.output_parsed

def retrieve_context(query: str, filter: bool = False, k: int = FINAL_K) -> list:
    """
    Retrieves top K document chunks from the vector database based on a combination of cosine similarity and BM25 score
    Firstly generates embeddings for query, then route's query to one about either transcripts, meeting notes or both.
    Then retrieves filter text for Azure vector search.

    Args:
        query (str): User's query
        k (int): parameter to specify how many closest K documents to retrieve

    Returns:
        list: list of K closest document chunks
    """
    query_embedding = generate_embeddings([query])[0]

    vector_query = VectorizedQuery(
        vector=query_embedding,
        k_nearest_neighbors=k,
        fields="embedding"
    )

    # route deterimes which metadata fields should be filtered from filter text if any
    route = route_query(query)

    # get values for fields that will be used to build the filter text
    filter_metadata = retrieve_filter_metadata(query)
    
    combined = []
    # rather than retrieving context and filters for both types of documents we could route to specific ones based on the query
    if route.source in ("transcripts", "both"):
        transcript_filter = create_safe_filter_for_index(filter_metadata, "transcripts")
        transcript_results = TRANSCRIPT_SEARCH_CLIENT.search(
            search_text=query, # hybrid retrieval: populating search_text leads to BM25 score search as well as vector comparison
            vector_queries=[vector_query],
            filter = transcript_filter,
            top=k
        )
        for r in transcript_results:
            r_dict = dict(r)
            r_dict["_index"] = "transcripts"
            r_dict["_score"] = r.get("@search.score", 0)
            combined.append(r_dict)

    if route.source in ("meeting_notes", "both"):
        meeting_filter = create_safe_filter_for_index(filter_metadata, "meeting_notes")
        meeting_results = MEETING_NOTES_SEARCH_CLIENT.search(
            search_text=query, # hybrid retrieval: populating search_text leads to BM25 score search as well as vector comparison
            vector_queries=[vector_query],
            filter=meeting_filter,
            top=k
        )

        for r in meeting_results:
            r_dict = dict(r)
            r_dict["_index"] = "meeting_notes"
            r_dict["_score"] = r.get("@search.score", 0)
            combined.append(r_dict)

    combined_sorted = sorted(
        combined,
        key=lambda x: x["_score"],
        reverse=True
    )

    return combined_sorted[:k]
    

        

def create_safe_filter_for_index(metadata: dict, index_kind: str) -> str:
    """
    Generates filter text from the metadata and filters out any metadata fields
    that are not in the search index. E.g. if metadata cotains 'author' and index_kind == 'earning_call' then author is not sent to build the filter with.

    Args:
        filter_text (dict): filter that will be applied to azure search index e.g. 'author eq John'
        index_kind (str): "transcripts" or "meeting_notes"
    
    Returns:
        str: A valid Azure AI Search OData filter string combining all fields. Example output:

        "docType eq 'earnings_call' and
         (company eq 'Apple' or company eq 'Agilent') and
         year eq 2024 and
         (quarter eq 2 or quarter eq 4)"

        Returns an empty string if no filters are generated.

    """
    # Only include fields supported by index
    allowed = {
        "transcripts": {"docType", "company", "quarter", "year"},
        "meeting_notes": {"docType", "author", "meetingDate"},
    }

    allowed_fields = allowed.get(index_kind, set())

    filtered_meta = {
        k: v for k, v in metadata.items() if k in allowed_fields
    }

    return build_filter(filtered_meta)


def retrieve_filter_metadata(query: str) -> dict:
    """
    Runs full pipeline of retrieving metadata as a langextract object from query, then converting that to a dictionary of metadata.

    Args:
        query (str): User's query
    
    Returns:
        dict: metadata where keys are fields and values are values for those fields extracted from query
    """
    result = return_metadata(query)
    metadata = langextract_to_metadata(result)
    return metadata

    

def return_metadata(query: str) -> AnnotatedDocument:
    """
    Extracts specific classes from the query to be used as metadata for future filter text. Uses few-shot prompting which provide examples of which
    metadata fields should be extracted.

    Args:
        query (str): The user's query
    
    Returns:
        AnnotatedDocument: langextract AnnotatedDocument object which contains attributes and other fields
    
    :param query: Description
    :type query: str
    :return: Description
    :rtype: str
    """
    # Build prompt for metadata extractor LLM:
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
    # Provide example for metadata extractor LLM:
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


def langextract_to_metadata(annotated_doc: AnnotatedDocument) -> dict:
    """
    Convert LangExtract AnnotatedDocument into structured metadata dict. Also supports multiple attributes, e.g. if a query mentioned multiple companies or years

    Args:
        annotated_doc (AnnotatedDocument): langextract AnnotatedDocument object
    
    Returns:
        dict: metadata structured in a dictionary format with each key referencing a list of 1 or more attributes
    """

    metadata = {
        # earning call metadata
        "docType": [],
        "company": [],
        "year": [],
        "quarter": [],
        # Meeting notes metadata
        "meetingDate": [],
        "author": []

    }

    for extraction in annotated_doc.extractions:
        
        attrs = extraction.attributes or {}
        print(f"\n attribute: {attrs}")
        if "document_type" in attrs:
            metadata["docType"].append(attrs["document_type"])

        if "company" in attrs:
            metadata["company"].append(attrs["company"])

        if "year" in attrs:
            try:
                metadata["year"].append(int(attrs["year"]))
            except:
                pass

        if "quarter" in attrs:
            try:
                metadata["quarter"].append(int(attrs["quarter"]))
            except:
                pass
        
        if "author" in attrs:
            try:
                metadata["author"].append(attrs["author"])
            except:
                pass

        if "meetingDate" in attrs:
            try:
                metadata["meetingDate"].append(datetime.strptime(
                    attrs["meetingDate"], "%Y/%m/%d"
                ).isoformat() + "Z")
            except Exception as e:
                raise(e)

    # post-processing: remove any duplicate attributes (e.g. year: [2024,2024] -> year: [2024])
    for key in metadata:
        metadata[key] = list(set(metadata[key]))

    return metadata

def build_filter(meta: dict) -> str:
    """
    Construct a complete azure search filter string from extracted metadata.

    This function:
    - Iterates over known metadata fields (e.g. company, year, quarter)
    - Builds a filter clause for each field using `build_or`
    - Combines all field clauses using AND

    Each field can produce:
    - A single equality condition (e.g. "year eq 2024")
    - An OR group if multiple values exist
      (e.g. "(company eq 'Apple' or company eq 'Agilent')")

    Args:
        meta (dict):
            Dictionary containing extracted metadata, where each key maps
            to a list of values. Example:
            {
                "docType": ["earnings_call"],
                "company": ["Apple", "Agilent"],
                "year": [2024],
                "quarter": [2, 4],
                "meetingDate": [],
                "author": []
            }

    Returns:
        str:
            A valid Azure AI Search OData filter string combining all fields.
            Example output:
            "docType eq 'earnings_call' and
             (company eq 'Apple' or company eq 'Agilent') and
             year eq 2024 and
             (quarter eq 2 or quarter eq 4)"

            Returns an empty string if no filters are generated.
    """

    parts = []

    def build_or(field: str, values: list, is_string: bool = False, is_datetime: bool = False) -> str | None:
        """
        Build an azure search filter clause for a single field (year, company...)
        This function converts a list of values into:
        - a single 'eq' condition if one value is provided
        - an OR-combined group of 'eq' conditions if multiple values are provided

        Args:
            field (str): The field name in the Azure Search index (e.g. "company", "year")
            values (list): List of values to filter on
            is_string (bool): Whether the field is a string (wrap in quotes)
            is_datetime (bool): Whether the field is a DateTimeOffset (ISO string, no quotes)

        Returns:
            str | None:
                - A valid OData filter clause (e.g. "company eq 'Apple'" or
                "(company eq 'Apple' or company eq 'Agilent')")
                - None if the values list is empty
        """
        if not values:
            return None

        if len(values) == 1:
            v = values[0]
            if is_string:
                return f"{field} eq '{v}'" # if the field is a string in search index it must be wrapped in ''
            elif is_datetime:
                return f"{field} eq {v}"  # already ISO string
            else:
                return f"{field} eq {v}"

        # multiple values → OR
        conditions = []
        for v in values:
            if is_string:
                conditions.append(f"{field} eq '{v}'")
            elif is_datetime:
                conditions.append(f"{field} eq {v}")
            else:
                conditions.append(f"{field} eq {v}")

        return "(" + " or ".join(conditions) + ")"

    # (field, is_string, is_datetime)
    mappings = [
        ("docType", True, False),
        ("company", True, False),
        ("year", False, False),
        ("quarter", False, False),
        ("author", True, False),
        ("meetingDate", False, True),
    ]

    for field, is_string, is_datetime in mappings:
        clause = build_or(field, meta.get(field, []), is_string, is_datetime)
        if clause:
            parts.append(clause)

    return " and ".join(parts)


if __name__ == "__main__":
    pass
    test_query = """
            Can you summarize the earning call from Apple and Reuben's meeting notes
            """
    print(retrieve_context(test_query))
   


