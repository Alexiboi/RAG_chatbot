from .LLMJudge import RetrievalRelevanceJudge
import json
import os

K = 6
OVERLAP_THRESHOLD = 0.4  # fraction of gold passage that must appear contiguously in a chunk
PASSAGE_MAP_PATH = "./evaluation/passage_map.json"


def _load_passage_map(path: str = PASSAGE_MAP_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Passage map not found at {path}. "
            f"Run evaluation/build_passage_map.py first."
        )
    with open(path, "r") as f:
        return json.load(f)


# Loaded once at module import — no repeated file reads during evaluation
PASSAGE_MAP = _load_passage_map()


def _passage_overlaps_chunk(gold_passage: str, chunk_content: str, threshold: float = OVERLAP_THRESHOLD) -> bool:
    """
    Returns True if a sufficient portion of the gold passage appears
    contiguously in the chunk content.

    Uses a sliding window over the gold passage to find the longest
    matching substring, then checks whether it meets the minimum
    character threshold. This handles cases where a gold passage spans
    a chunk boundary and is therefore only partially present in any
    single chunk.

    :param gold_passage: Raw gold passage string
    :param chunk_content: Raw chunk content string
    :param threshold: Fraction of gold passage length that must appear
                      contiguously in the chunk to count as a match
    :return: True if the overlap threshold is met
    """
    normalised_passage = " ".join(gold_passage.split()).lower()
    normalised_chunk = " ".join(chunk_content.split()).lower()
    passage_len = len(normalised_passage)
    min_match_len = int(passage_len * threshold)

    for window_size in range(passage_len, min_match_len - 1, -1):
        for start in range(0, passage_len - window_size + 1):
            substring = normalised_passage[start:start + window_size]
            if substring in normalised_chunk:
                return True

    return False


def _get_gold_chunk_ids(gold_passages: list[str]) -> set[str]:
    """
    Returns the set of chunk IDs whose normalised content overlaps
    sufficiently with any of the gold passages.

    Uses _passage_overlaps_chunk rather than exact substring matching
    to handle gold passages that span chunk boundaries.

    :param gold_passages: List of gold passage strings from ground truth
    :return: Set of matching chunk IDs
    """
    gold_ids = set()
    for passage in gold_passages:
        for chunk_id, content in PASSAGE_MAP.items():
            if _passage_overlaps_chunk(passage, content):
                gold_ids.add(chunk_id)
    return gold_ids


def _get_retrieved_ids(retrieved_k: list[dict]) -> list[str]:
    """
    Extracts chunk IDs from retrieved results in rank order.
    Azure AI Search returns an 'id' field on each result.

    :param retrieved_k: List of retrieved chunk dicts
    :return: Ordered list of chunk ID strings
    """
    return [chunk.get("id", "") for chunk in retrieved_k]


def recall_at_k(run, example, k=K):
    """
    Recall@K — fraction of gold passages covered by at least one
    of the top-K retrieved chunks, matched by chunk ID.

    A gold passage is considered covered if any retrieved chunk's
    content overlaps with it above the OVERLAP_THRESHOLD, handling
    cases where passages span chunk boundaries.

    :param run: RunObject with run.outputs["retrieved"] list of chunk dicts
    :param example: ExampleObject with example.outputs["gold_passages"] list
    :param k: Number of top retrieved documents to consider
    :return: dict with score in range [0.0, 1.0]
    """
    retrieved = run.outputs.get("retrieved", []) or []
    retrieved_k = list(retrieved)[:k]
    gold_passages = example.outputs.get("gold_passages", [])

    if not gold_passages:
        return {"key": f"recall@{k}", "score": None, "comment": "No gold passages"}

    retrieved_ids = set(_get_retrieved_ids(retrieved_k))
    covered = 0

    for passage in gold_passages:
        for chunk_id, content in PASSAGE_MAP.items():
            if chunk_id in retrieved_ids and _passage_overlaps_chunk(passage, content):
                covered += 1
                break  # this passage is covered, move to next

    return {"key": f"recall@{k}", "score": covered / len(gold_passages)}


def mrr(run, example, k=K):
    """
    MRR@K — reciprocal rank of the first retrieved chunk whose ID
    appears in the gold chunk ID set.

    Rank-based: rewards systems that surface a relevant chunk
    earlier in the retrieval list.

    :param run: RunObject with run.outputs["retrieved"] list of chunk dicts
    :param example: ExampleObject with example.outputs["gold_passages"] list
    :param k: Number of top retrieved documents to consider
    :return: dict with score in range [0.0, 1.0]
    """
    retrieved = run.outputs.get("retrieved", []) or []
    retrieved_k = list(retrieved)[:k]
    gold_passages = example.outputs.get("gold_passages", [])

    if not gold_passages:
        return {"key": "mrr", "score": None, "comment": "No gold passages"}

    gold_ids = _get_gold_chunk_ids(gold_passages)
    retrieved_ids = _get_retrieved_ids(retrieved_k)

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in gold_ids:
            return {"key": "mrr", "score": 1.0 / rank}

    return {"key": "mrr", "score": 0.0}


def map_at_k(run, example, k=K):
    """
    AP@K — average precision at K, matched by chunk ID.

    AP@K = (1 / min(|relevant|, K)) * sum_{i=1..K} (Precision@i * rel(i))

    Where rel(i) = 1 if the chunk at rank i is in the gold chunk ID set.
    The denominator is capped at min(|gold_passages|, k) to avoid
    penalising systems when there are more gold passages than retrieved
    documents.

    :param run: RunObject with run.outputs["retrieved"] list of chunk dicts
    :param example: ExampleObject with example.outputs["gold_passages"] list
    :param k: Number of top retrieved documents to consider
    :return: dict with score in range [0.0, 1.0]
    """
    retrieved = run.outputs.get("retrieved", []) or []
    retrieved_k = list(retrieved)[:k]
    gold_passages = example.outputs.get("gold_passages", [])

    if not gold_passages:
        return {"key": "map", "score": None, "comment": "No gold passages"}

    gold_ids = _get_gold_chunk_ids(gold_passages)
    retrieved_ids = _get_retrieved_ids(retrieved_k)
    denom = min(len(gold_passages), k)
    hits = 0
    ap_sum = 0.0

    for i, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in gold_ids:
            hits += 1
            ap_sum += hits / i

    return {"key": "map", "score": ap_sum / denom if denom > 0 else 0.0}


def LLM_judge_relevance(run, example=None):
    """
    LLM judge for retrieval relevance — unchanged, operates on
    content semantically so does not need ID matching.

    :param run: RunObject with run.inputs["question"] and
                run.outputs["retrieved"]
    :param example: ExampleObject (passed for interface consistency,
                    not used directly by the judge)
    :return: dict with relevance score
    """
    judge_client = RetrievalRelevanceJudge(run, example)
    return judge_client.judge()