from langsmith import evaluate, Client
from src.rag_chatbot.rag.RAG_bot import generate_contextualized_response
from evaluation.evaluators.retrieval_eval import recall_at_k, LLM_judge_relevance, mrr
from evaluation.evaluators.generation_eval import LLM_judge_answer_relevance, LLM_judge_answer_correctness, LLM_judge_answer_faithfulness

ls_client = Client()
dataset_id="27823689-62e1-4151-9598-fe077db022ee"
examples = list(ls_client.list_examples(dataset_id=dataset_id))

rag_app = generate_contextualized_response
evaluators = [LLM_judge_relevance, LLM_judge_answer_relevance, LLM_judge_answer_correctness, LLM_judge_answer_faithfulness, mrr, recall_at_k]

results = evaluate(
    rag_app,              # Your application function
    data=examples,           # Dataset to evaluate on
    evaluators=evaluators,  # List of evaluator functions
    experiment_prefix="2024Q4_Agilent_eval"
)

print(results)