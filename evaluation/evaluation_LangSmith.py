from langsmith import evaluate, Client
from langsmith.schemas import Run, Example
from src.backend.rag.RAG_bot import RAGLLM
from src.backend.rag.retrieval_utils import retrieve_filter_metadata, create_safe_filter_for_index
from evaluation.evaluators.retrieval_eval import recall_at_k, LLM_judge_relevance, mrr, map_at_k
from evaluation.evaluators.generation_eval import LLM_judge_answer_relevance, LLM_judge_answer_correctness, LLM_judge_answer_faithfulness
import pandas as pd
import json
from src.backend.rag.embedding_utils import generate_embeddings
from src.backend.rag.index_utils import TRANSCRIPT_SEARCH_CLIENT
from azure.search.documents.models import VectorizedQuery

ls_client = Client()

JSONL_PATH = "./data/evaluation_data/gt_combined.jsonl"
COMBINED_DATASET_NAME = "Combined_Qs_75"


# ── Wrappers ───────────────────────────────────────────────────────────────────

class RunObject:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs


class ExampleObject:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs


def _wrap_run(run: Run) -> RunObject:
    return RunObject(inputs=run.inputs or {}, outputs=run.outputs or {})


def _wrap_example(example: Example) -> ExampleObject:
    return ExampleObject(inputs=example.inputs or {}, outputs=example.outputs or {})


# ── LangSmith-compatible evaluator wrappers ────────────────────────────────────

def ls_recall_at_k(run: Run, example: Example):
    return recall_at_k(_wrap_run(run), _wrap_example(example))

def ls_mrr(run: Run, example: Example):
    return mrr(_wrap_run(run), _wrap_example(example))

def ls_map_at_k(run: Run, example: Example):
    return map_at_k(_wrap_run(run), _wrap_example(example))

def ls_LLM_judge_relevance(run: Run, example: Example):
    return LLM_judge_relevance(_wrap_run(run), _wrap_example(example))

def ls_LLM_judge_answer_relevance(run: Run, example: Example):
    return LLM_judge_answer_relevance(_wrap_run(run), _wrap_example(example))

def ls_LLM_judge_answer_correctness(run: Run, example: Example):
    return LLM_judge_answer_correctness(_wrap_run(run), _wrap_example(example))

def ls_LLM_judge_answer_faithfulness(run: Run, example: Example):
    return LLM_judge_answer_faithfulness(_wrap_run(run), _wrap_example(example))


ls_evaluators = [
    ls_LLM_judge_relevance,
    ls_LLM_judge_answer_relevance,
    ls_LLM_judge_answer_correctness,
    ls_LLM_judge_answer_faithfulness,
    ls_mrr,
    ls_map_at_k,
    ls_recall_at_k,
]


# ── Dataset management ─────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    """Load examples from a JSONL file, skipping blank lines and comments."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("/"):
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping line {i}: {e} — content: {repr(line[:80])}")
    print(f"Loaded {len(examples)} examples from {path}")
    return examples


def get_or_create_combined_dataset(jsonl_path: str = JSONL_PATH) -> str:
    """
    Creates a LangSmith dataset from gt_combined.jsonl if it does not
    already exist. Returns the dataset name for use with evaluate().

    Each example is stored with:
      inputs:  { question }
      outputs: { reference_answer, gold_passages, answerable }

    If the dataset already exists in LangSmith, it is reused as-is
    to avoid duplicating examples on repeated runs.
    """
    # Check if dataset already exists
    existing = [d for d in ls_client.list_datasets() if d.name == COMBINED_DATASET_NAME]
    if existing:
        print(f"Dataset '{COMBINED_DATASET_NAME}' already exists in LangSmith — reusing it.")
        return COMBINED_DATASET_NAME

    print(f"Creating LangSmith dataset '{COMBINED_DATASET_NAME}' from {jsonl_path}...")
    raw_examples = load_jsonl(jsonl_path)

    dataset = ls_client.create_dataset(
        dataset_name=COMBINED_DATASET_NAME,
        description="Combined evaluation set across all five earnings calls (75 questions)"
    )

    inputs_list = []
    outputs_list = []

    for ex in raw_examples:
        inputs_list.append({
            "question": ex["inputs"]["question"]
        })
        outputs_list.append({
            "reference_answer": ex["outputs"].get("reference_answer", ""),
            "gold_passages": ex["outputs"].get("gold_passages", []),
            "answerable": ex["outputs"].get("answerable", True),
        })

    ls_client.create_examples(
        inputs=inputs_list,
        outputs=outputs_list,
        dataset_id=dataset.id,
    )

    print(f"Created dataset with {len(inputs_list)} examples.")
    return COMBINED_DATASET_NAME


# ── RAG application ────────────────────────────────────────────────────────────

def local_retrieve(query: str, filter_on: bool = False, hybrid: bool = False, k: int = 6, rerank: bool = False) -> list[dict]:
    
    query_embedding = generate_embeddings([query])[0]
    
    # When reranking, pass 50 candidates to the semantic ranker, then slice to k after
    search_top = 20 if rerank else k
    
    vector_query = VectorizedQuery(
        vector=query_embedding,
        k_nearest_neighbors=search_top,
        fields="embedding"
    )

    shared_params = dict(
        search_text=query if (hybrid or rerank) else "*",
        vector_queries=[vector_query],
        top=search_top,
        **({"query_type": "semantic", # means that RRF ranking (BM25 + cosine) first then reranker on top
            "semantic_configuration_name": "reranker-semantic", #reranker-semantic
            "semantic_error_mode": "partial"
        } if rerank else {})
    )

    if filter_on:
        filter_metadata = retrieve_filter_metadata(query)
        transcript_filter = create_safe_filter_for_index(filter_metadata, "transcripts")
        results = TRANSCRIPT_SEARCH_CLIENT.search(
            **shared_params,
            filter=transcript_filter,
        )
    else:
        results = TRANSCRIPT_SEARCH_CLIENT.search(
            **shared_params,
            filter="",
        )

    results_list = [dict(r) for r in results]



    if rerank:
        results_list.sort(
            key=lambda x: x.get("@search.reranker_score") or 0,
            reverse=True
        )

    return results_list[:k]


def rag_app(inputs: dict) -> dict:
    """
    Target function called by LangSmith for each example.
    Returns answer and retrieved chunks so evaluators can access both.
    """
    question = inputs["question"].strip()
    retrieved = local_retrieve(question, k=6, rerank=False, hybrid=True, filter_on=False)
    answer = RAGLLM.generate_answer(question, retrieved, [])
    return {
        "answer": answer,
        "retrieved": retrieved,
    }


# ── Aggregate score display ────────────────────────────────────────────────────

def display_aggregate_scores(results, experiment_name: str) -> pd.DataFrame:
    rows = []
    for result in results._results:
        run = result.run
        row = {
            "question": (run.inputs or {}).get("question", "")[:80]
        }
        for er in result.evaluation_results.results:
            row[er.key] = er.score
        rows.append(row)

    df = pd.DataFrame(rows)
    numeric_cols = df.select_dtypes(include="number").columns

    print(f"\n===== PER-QUESTION SCORES — {experiment_name} =====")
    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print(df.to_string(index=False))

    print(f"\n===== AGGREGATE SCORES — {experiment_name} =====")
    agg = df[numeric_cols].agg(["mean", "min", "max", "std"]).round(4)
    print(agg.to_string())

    return df


# ── Main experiment runner ─────────────────────────────────────────────────────

def run_experiment(experiment_name: str = "baseline") -> pd.DataFrame:
    """
    Runs a single experiment against the full gt_combined dataset.
    Results are visible in LangSmith and aggregate scores are printed locally.

    Args:
        experiment_name: Label for this experimental condition.
                         Change this between runs to track different
                         configurations in LangSmith, e.g.:
                         "cosine_only_k6", "hybrid_k10", "no_filter"
    Returns:
        DataFrame of per-question scores
    """
    dataset_name = get_or_create_combined_dataset(JSONL_PATH)

    print(f"\nRunning experiment: '{experiment_name}' on dataset: '{dataset_name}'")

    results = evaluate(
        rag_app,
        data=dataset_name,
        evaluators=ls_evaluators,
        experiment_prefix=experiment_name,
    )

    df = display_aggregate_scores(results, experiment_name)
    return df


if __name__ == "__main__":
    # Change experiment_name to label each experimental condition:
    # e.g. "cosine_only_k6", "hybrid_k6", "hybrid_k10", "no_filter", "with_filter"
    run_experiment(experiment_name="hybrid_nofilter_chunk700_04mini")

"""
Create new combined questions dataset that has 120 questions for 8 earnings calls including a 2023 call
"""

"""
Create a test for precision which tests the fraction of retrieved chunks that belong to the gold chunks:
retrieved / gold

Keep the number of possible queryable vectors the same
Make Agilent Q4 questions harder that need to retrieve from multiple chunks
Also include a calls dataset for Apple in the 2023 year to test the temporal filtering

Experiment on different chunk sizes (300, 700, 1000, 1500) which describes number of characters (done)
Experiment on different values of k: 5, 10, 20, 30
Experiment with different overlap levels then pick best one
Experiment on chunking using context vs not using context in each chunk
Experiment on just using cosine similarity vs using hyrbid search vs BM25
Experiment on using filters vs without
Experiment using the best combination of baseline RAG vs GPT without RAG vs RAG with filters vs RAG with hybrid retrieval vs RAG reranker

Implement a survey for UI and asking users to judge the model's answer

Look more at how literature conducted experiments (maybe introduce ROUGE and BERT)

Then we need some sort of way to evaluate MCP
"""