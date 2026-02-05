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

def recall_at_k(run, example, k=6): 
    pred_ids = [d["id"] for d in run.outputs["retrieved"][:k]]
    gold_ids = example.outputs.get("gold_chunk_ids", [])
    if not gold_ids:
        return {"key": f"recall@{k}", "score": None, "comment": "No gold chunks"}
    hit = len(set(pred_ids) & set(gold_ids))
    return {"key": f"recall@{k}", "score": hit / len(set(gold_ids))}

def mrr(run, example, k=10):
    pred_ids = [d["id"] for d in run.outputs["retrieved"][:k]]
    gold_ids = set(example.outputs.get("gold_chunk_ids", []))
    if not gold_ids:
        return {"key": "mrr", "score": None, "comment": "No gold chunks"}
    for i, cid in enumerate(pred_ids, start=1):
        if cid in gold_ids:
            return {"key": "mrr", "score": 1.0 / i}
    return {"key": "mrr", "score": 0.0}

def LLM_judge_relevance(run, example=None):
    """
    Use an LLM to judge if the retrieved documents are relevant
    to the user's question.
    
    :param inputs: Description
    :param gold: Description
    """
     # Prefer the original question if available; fallback to prompt
    question = None
    if hasattr(run, "inputs") and isinstance(run.inputs, dict):
        question = run.inputs.get("question")
    if not question:
        question = run.outputs.get("question") or run.outputs.get("prompt") or ""
    
    documents = run.outputs.get("retrieved") or []

    # Keep message readable: show IDs/snippets instead of dumping huge content
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

    docs_for_judge = compact_docs(documents)

    class Response(BaseModel):
        documents_are_relevant: bool = Field(..., description="True if the retrieved docs are relevant to the question.")
        rationale: str = Field(..., description="Brief reason for the decision (1-3 sentences).")
        # Optional extra fields if you want more detail:
        missing_info: str | None = Field(None, description="If not relevant, what evidence seems missing?")
        confidence: float = Field(..., ge=0, le=1, description="Confidence from 0 to 1.")

    instructions="""
    Given the question and retrieved documents, decide whether the retrieved documents are relevant.\n
    Be strict: if the documents don't contain information that would help answer the question, mark False.\n
    Return JSON exactly matching the schema.
    """

    msg = {
        "question": question,
        "retrieved_documents": docs_for_judge
    }

    # Call the LLM to judge the output
    response = judge_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "system", "content": instructions}, {"role": "user", "content": str(msg)}],
        response_format=Response
    )

    # Return the boolean score
    parsed = response.choices[0].message.parsed

    score = 1.0 if parsed.documents_are_relevant else 0.0

    comment = (
        f"rationale: {parsed.rationale}\n"
        f"confidence: {parsed.confidence}\n"
        f"missing_info: {parsed.missing_info}"
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



