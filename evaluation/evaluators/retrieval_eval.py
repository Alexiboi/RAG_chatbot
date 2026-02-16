from .LLMJudge import RetrievalRelevanceJudge

# Include tests that test both:
# Relevance(Retrieved Docs <-> Query/prompt)
# Accuracy(Retrieve Docs <-> Gold standard docs)

# Metrics can be split up into both non-rank based and rank-based approaches
# non-rank based metrics don't consider the order of documents retrieved
# rank-based metrics do.

# Both Recall@K and MRR will be used for empirical measure of retrieved
# document accuracy
# LLM as a judge will be used as a Relevance measure.

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
    Compute Mean Reciprocal Rank (MRR) at k.

    Returns 1 / rank of the first retrieved document whose ID 
    appears in the gold_chunk_ids. If no relevant document is 
    found in the top-k results, returns 0.0.

    If no gold_chunk_ids are provided, returns None. 
    
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

def map_at_k(run, example, k=6):
    """
    Compute Average Precision (AP) at k for a single query/run.
    (In a dataset, MAP@k is the mean of these AP@k scores.)

    AP@k = (1 / min(|relevant|, k)) * sum_{i=1..k} (Precision@i * rel(i))

    - rel(i) = 1 if the doc at rank i is relevant (its id in gold_chunk_ids), else 0
    - Precision@i = (# relevant in top i) / i

    If no gold_chunk_ids are provided, returns None.

    :param run: object with run.outputs["retrieved"] list of dicts containing "id"
    :param example: object with example.outputs["gold_chunk_ids"] list
    :param k: cutoff for retrieved documents
    """
    retrieved = run.outputs.get("retrieved", []) or []
    pred_ids = [d.get("id") for d in retrieved[:k] if isinstance(d, dict)]
    gold_ids = set(example.outputs.get("gold_chunk_ids", []))

    if not gold_ids:
        return {"key": "map", "score": None, "comment": "No gold chunks"}

    # Count how many relevant docs exist "within k" for MAP@k normalisation
    denom = min(len(gold_ids), k)

    hits = 0
    ap_sum = 0.0
    for i, cid in enumerate(pred_ids, start=1):
        if cid in gold_ids:
            hits += 1
            precision_i = hits / i
            ap_sum += precision_i

    score = ap_sum / denom if denom > 0 else 0.0
    return {"key": "map", "score": score}

def LLM_judge_relevance(run, example=None):
    """
    Use an LLM to judge if the retrieved documents are relevant
    to the user's question.
    
    :param inputs: Description
    :param gold: Description
    """
    judge_client = RetrievalRelevanceJudge(run, example)
    return judge_client.judge()



