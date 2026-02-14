# The 3 targets that we need to evaluate for generation are:
# Relevance (Response ↔ Query)
# Faithfulness (Response ↔ Relevant Documents)
# Correctness (Response ↔ Sample Response) 
from .LLMJudge import AnswerRelevanceJudge, AnswerFaithfulnessJudge, AnswerCorrectnessJudge


def LLM_judge_answer_relevance(run, example=None):
    """
    Use an LLM to judge if the contextualized response generated is relevant to the
    user's query.
    
    :param run: Description
    :param example: Description
    """
    judge_client = AnswerRelevanceJudge(run, example)
    return judge_client.judge()


def LLM_judge_answer_faithfulness(run, example=None):
    """
    Use an LLM to judge if the contextualized response generated is faithfull
    to the retrieved context
    
    :param run: Description
    :param example: Description
    """
    judge_client = AnswerFaithfulnessJudge(run, example)
    return judge_client.judge()


# create another evaluator that assigns 1 if the example was identified correctly as non answerable and 0 if not.
def LLM_judge_answer_correctness(run, example=None):
    """
    Use an LLM to judge if the contextualized response generated is relevant to the
    user's query.
    
    :param run: Description
    :param example: Description
    """
    judge_client = AnswerCorrectnessJudge(run, example)
    return judge_client.judge()