import pytest
from evaluation.evaluators.retrieval_eval import mrr

class Dummy:
    def __init__(self, outputs: dict):
        self.outputs = outputs

def test_mrr_gold_first():
    run = Dummy(outputs={"retrieved": [{"id": "a"}, {"id": "b"}, {"id": "c"}]})
    example = Dummy(outputs={"gold_chunk_ids": ["a"]})
    result = mrr(run, example, k=3)
    assert result["score"] == 1.0

def test_mrr_gold_third():
    run = Dummy(outputs={"retrieved": [{"id": "z"}, {"id": "x"}, {"id": "y"}]})
    example = Dummy(outputs={"gold_chunk_ids": ["y"]})
    result = mrr(run, example, k=3)
    assert result["score"] == pytest.approx(1/3)

def test_mrr_no_gold_found():
    run = Dummy(outputs={"retrieved": [{"id": "x"}, {"id": "y"}, {"id": "z"}]})
    example = Dummy(outputs={"gold_chunk_ids": ["not_in_list"]})
    result = mrr(run, example, k=3)
    assert result["score"] == 0.0

def test_mrr_no_gold_chunks():
    run = Dummy(outputs={"retrieved": [{"id": "x"}, {"id": "y"}]})
    example = Dummy(outputs={"gold_chunk_ids": []})
    result = mrr(run, example, k=2)
    assert result["score"] is None

def test_mrr_multiple_gold_chunks():
    run = Dummy(outputs={"retrieved": [{"id": "x"}, {"id": "gold1"}, {"id": "gold2"}]})
    example = Dummy(outputs={"gold_chunk_ids": ["gold1", "gold2"]})
    result = mrr(run, example, k=3)
    assert result["score"] == pytest.approx(1/2)