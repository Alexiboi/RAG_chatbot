from langsmith import evaluate, Client
from src.rag_chatbot.rag.RAG_bot import generate_contextualized_response
from evaluation.evaluators.retrieval_eval import recall_at_k, LLM_judge_relevance, mrr, map_at_k
from evaluation.evaluators.generation_eval import LLM_judge_answer_relevance, LLM_judge_answer_correctness, LLM_judge_answer_faithfulness

ls_client = Client()
dataset_ids = {
    "2024Q4_Agilent":"27823689-62e1-4151-9598-fe077db022ee",
    "2024Q2_Agilent":"b41232fc-afbd-41e7-9f3d-f103232c12e4",
    "2024Q1_Amazon":"f904b7e2-675b-4973-ad8e-deeadb532d03",
    "2024Q3_Apple":"c877c351-470f-4b6a-a211-672d006d77df",
    "2024Q3_Blackstone":"17ad2d7c-beb0-44fb-ba28-4645f6fb967c"
}

rag_app = generate_contextualized_response
evaluators = [LLM_judge_relevance, LLM_judge_answer_relevance, LLM_judge_answer_correctness, LLM_judge_answer_faithfulness, mrr, map_at_k, recall_at_k]


def run_experiment(key: str, experiment_name = ""):
    examples = list(ls_client.list_examples(dataset_id=dataset_ids[key]))
    experiment_prefix = (key + "_" + experiment_name).strip()
    results = evaluate(
        rag_app,              # Your application function
        data=examples,           # Dataset to evaluate on
        evaluators=evaluators,  # List of evaluator functions
        experiment_prefix=experiment_prefix,
    )

    print(results)

def run_all_experiments(experiment_name=""):
    for id in dataset_ids.keys():
        run_experiment(key=id, experiment_name=experiment_name)

if __name__ == '__main__':
    run_all_experiments(experiment_name="without_rerank")