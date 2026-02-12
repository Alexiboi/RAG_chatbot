from openai import OpenAI
from langsmith import evaluate, traceable, wrappers, Client
from pydantic import BaseModel, Field

# Include tests that test both:
# Relevance(Retrieved Docs <-> Query/prompt)
# Accuracy(Retrieve Docs <-> Gold standard docs)

# Metrics can be split up into both non-rank based and rank-based approaches
# non-rank based metrics don't consider the order of documents retrieved
# rank-based metrics do.

# Both Recall@K and MRR will be used for empirical measure of retrieved
# document accuracy
# LLM as a judge will be used as a Relevance measure.

judge_client = OpenAI()

def compact_docs(docs, max_docs=6, max_chars=600):
        compact = []
        for d in docs[:max_docs]:
            if isinstance(d, dict):
                doc_id = d.get("id", "")
                text = d.get("content") or d.get("text") or ""
                score = d.get("score")
                text = text[:max_chars]
                compact.append({"id": doc_id, "score": score, "text": text})
            else:
                # If docs are plain strings
                compact.append(str(d)[:max_chars])
        return compact

def recall_at_k(run, example, k=6): 
    """
    recall returns the fraction of correct documents retrieved irrespective
    of they're order in the retrieval.
    
    :param run: Description
    :param example: Description
    :param k: Description
    """
    pred_ids = [d["id"] for d in run.outputs["retrieved"][:k]]
    gold_ids = example.outputs.get("gold_chunk_ids", [])
    if not gold_ids:
        return {"key": f"recall@{k}", "score": None, "comment": "No gold chunks"}
    hit = len(set(pred_ids) & set(gold_ids))
    return {"key": f"recall@{k}", "score": hit / len(set(gold_ids))}

def mrr(run, example, k=6):
    """
    rank-based metric, returns a higher score if the 
    gold standard document Id was retrieved higher in the ordering.
    So 1.0 mrr indicates that the correct document was retrieved first,
    if there was only 1 gold standard document.
    
    :param run: dictiornary which contains the inputs
    (question) and outputs (retrieved docs & answer). 
    Represents the output of the chatbot for a single example

    :param example: Dictionary representing the inputs and
    outputs for a gold standard example. (Gold standard answer, chunks)

    :param k: number of documents retrieved
    """
    pred_ids = [d["id"] for d in run.outputs["retrieved"][:k]]
    gold_ids = set(example.outputs.get("gold_chunk_ids", []))
    if not gold_ids:
        return {"key": "mrr", "score": None, "comment": "No gold chunks"}
    for i, cid in enumerate(pred_ids, start=1):
        if cid in gold_ids:
            return {"key": "mrr", "score": 1.0 / i}
    return {"key": "mrr", "score": 0.0}

def _judge_call(msg: str, ResponseModel):
    response = judge_client.responses.parse(
        model="gpt-4o",
        input=msg,
        text_format=ResponseModel,
    )
    return response

def LLM_judge_relevance(run, example=None):
    """
    Use an LLM to judge if the retrieved documents are relevant
    to the user's question.
    
    :param inputs: Description
    :param gold: Description
    """
     # Prefer the original question if available; fallback to prompt
    if example is not None:
        answerable = (example.outputs or {}).get("answerable", True)
        if not answerable:
            return {
                "key": "retrieval_relevance_binary",
                "score": None,
                "comment": "Skipped: example is marked answerable=false",
            }

    question = None
    if hasattr(run, "inputs") and isinstance(run.inputs, dict):
        question = run.inputs.get("question")
    if not question:
        question = run.outputs.get("question") or run.outputs.get("prompt") or ""
    
    documents = run.outputs.get("retrieved") or []

    # Keep message readable: show IDs/snippets instead of dumping huge content
    docs_for_judge = compact_docs(documents)

    class Response(BaseModel):
        documents_are_relevant: bool = Field(..., description="True if the retrieved docs are relevant to the question.")
        rationale: str = Field(..., description="Brief reason for the decision (1-3 sentences).")
        # Optional extra fields if you want more detail:
        missing_info: str | None = Field(None, description="If not relevant, what evidence seems missing?")
        confidence: float = Field(..., ge=0, le=1, description="Confidence from 0 to 1.")

    instructions = """
    You are an expert evaluator assessing whether retrieved documents are relevant to a given user query.

    Your task is to determine whether the retrieved documents match the information need expressed in the query.

    <Rubric>

    Relevant retrieval:
    - Documents directly relate to the subject of the query
    - Documents contain information that could help answer the query
    - Documents address the specific entities, concepts, or constraints in the query
    - Documents are topically aligned with the query’s intent

    Irrelevant retrieval:
    - Documents discuss unrelated topics
    - Documents only partially match surface keywords but miss the query’s intent
    - Documents contain general background information without addressing the query
    - Documents would not meaningfully help answer the query

    </Rubric>

    <Instructions>

    For this evaluation:

    - Carefully read the query to understand the information being requested
    - Review the retrieved documents
    - Determine whether the documents match the query’s topic and intent
    - Assess whether the documents would help answer the query if used in generation
    - Judge overall retrieval relevance

    </Instructions>

    <Reminder>

    Focus on topical and semantic relevance to the query.
    Do NOT judge answer correctness.
    Do NOT evaluate writing quality.
    Only assess whether the retrieved documents match the query’s information need.

    </Reminder>

    Now evaluate the following:

    <Query>
    {query}
    </Query>

    <RetrievedDocuments>
    {documents}
    </RetrievedDocuments>

    Return your judgment in structured JSON format.
    """.strip()

    msg = instructions.format(query=question, documents=docs_for_judge)

    # Call the LLM to judge the output
    response = _judge_call(msg, Response)

    # Return the boolean score
    parsed = response.output_parsed

    score = 1.0 if parsed.documents_are_relevant else 0.0

    comment = (
        f"rationale: {parsed.rationale}\n"
        f"confidence: {parsed.confidence}\n"
        f"missing_info: {parsed.missing_info}\n"
        f""
    )

    return {
        "key": "retrieval_relevance_binary",
        "score": score,
        "comment": comment,
        # Optional: attach extra structured info (can help debugging)
        "extra": {
            "documents_are_relevant": parsed.documents_are_relevant,
            "confidence": parsed.confidence,
            "question_used": question,
            "docs_preview": docs_for_judge,
        }
    }



