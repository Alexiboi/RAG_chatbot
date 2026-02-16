import pytest
from evaluation.evaluators.retrieval_eval import map_at_k  # or: from ... import map  (use your real function name)

class Dummy:
    def __init__(self, outputs: dict):
        self.outputs = outputs


def test_map_gold_first_only():
    # retrieved: a first, only one relevant in gold
    run = Dummy(outputs={"retrieved": [{"id": "a"}, {"id": "b"}, {"id": "c"}]})
    example = Dummy(outputs={"gold_chunk_ids": ["a"]})
    result = map_at_k(run, example, k=3)
    assert result["score"] == 1.0


def test_map_gold_third_only():
    # first relevant at rank 3:
    # precision@3 = 1/3, denom=min(|gold|,k)=1 => AP = 1/3
    run = Dummy(outputs={"retrieved": [{"id": "z"}, {"id": "x"}, {"id": "y"}]})
    example = Dummy(outputs={"gold_chunk_ids": ["y"]})
    result = map_at_k(run, example, k=3)
    assert result["score"] == pytest.approx(1/3)


def test_map_no_gold_found():
    run = Dummy(outputs={"retrieved": [{"id": "x"}, {"id": "y"}, {"id": "z"}]})
    example = Dummy(outputs={"gold_chunk_ids": ["not_in_list"]})
    result = map_at_k(run, example, k=3)
    assert result["score"] == 0.0


def test_map_no_gold_chunks():
    run = Dummy(outputs={"retrieved": [{"id": "x"}, {"id": "y"}]})
    example = Dummy(outputs={"gold_chunk_ids": []})
    result = map_at_k(run, example, k=2)
    assert result["score"] is None


def test_map_multiple_gold_chunks_first_and_third():
    # retrieved: x, gold1, gold2
    # relevant at rank 2 => precision@2 = 1/2
    # relevant at rank 3 => precision@3 = 2/3
    # denom=min(|gold|,k)=2
    # AP = (1/2 + 2/3)/2 = 7/12
    run = Dummy(outputs={"retrieved": [{"id": "x"}, {"id": "gold1"}, {"id": "gold2"}]})
    example = Dummy(outputs={"gold_chunk_ids": ["gold1", "gold2"]})
    result = map_at_k(run, example, k=3)
    assert result["score"] == pytest.approx((1/2 + 2/3) / 2)


def test_map_example_abc_with_gaps_top6():
    # Gold relevant docs = {A, B, C}
    # Retrieved: [A, X, B, Y, C, Z]
    # Precision@1 = 1/1 = 1.0  (A)
    # Precision@3 = 2/3        (A,B in top3)
    # Precision@5 = 3/5        (A,B,C in top5)
    # AP = (1 + 2/3 + 3/5) / 3 = 0.755555...
    run = Dummy(
        outputs={"retrieved": [{"id": "A"}, {"id": "X"}, {"id": "B"}, {"id": "Y"}, {"id": "C"}, {"id": "Z"}]}
    )
    example = Dummy(outputs={"gold_chunk_ids": ["A", "B", "C"]})
    result = map_at_k(run, example, k=6)
    expected = (1.0 + (2/3) + (3/5)) / 3
    assert result["score"] == pytest.approx(expected, rel=1e-6)


def test_map_k_cuts_off_relevant_docs():
    # Gold has 3 relevant, but k=2 only allows ranks 1..2
    # retrieved top2: [A, X] -> only A hit at rank1 => precision@1=1
    # denom=min(|gold|,k)=2 (MAP@2 convention used in our implementation)
    # AP@2 = 1/2 = 0.5
    run = Dummy(outputs={"retrieved": [{"id": "A"}, {"id": "X"}, {"id": "B"}, {"id": "C"}]})
    example = Dummy(outputs={"gold_chunk_ids": ["A", "B", "C"]})
    result = map_at_k(run, example, k=2)
    assert result["score"] == pytest.approx(0.5)
