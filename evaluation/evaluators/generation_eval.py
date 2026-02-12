# The 3 targets that we need to evaluate for generation are:
# Relevance (Response ↔ Query)
# Faithfulness (Response ↔ Relevant Documents)
# Correctness (Response ↔ Sample Response) 

from pydantic import BaseModel, Field
from evaluation.evaluators.retrieval_eval import _judge_call, compact_docs

def LLM_judge_answer_relevance(run, example=None):
    """
    Use an LLM to judge if the contextualized response generated is relevant to the
    user's query.
    
    :param inputs: Description
    :param gold: Description
    """
     # Prefer the original question if available; fallback to prompt
    if example is not None:
        answerable = (example.outputs or {}).get("answerable", True)
        if not answerable:
            return {
                "key": "answer_relevance_binary",
                "score": None,
                "comment": "Skipped: example is marked answerable=false",
            }

    question = None
    if hasattr(run, "inputs") and isinstance(run.inputs, dict):
        question = run.inputs.get("question")
    if not question:
        question = run.outputs.get("question") or run.outputs.get("prompt") or ""
    
    answer = run.outputs.get("answer")

    class Response(BaseModel):
        answer_is_relevant: bool = Field(..., description="True if the response is relevant to the query")
        rationale: str = Field(..., description="Brief reason for the decision (1-3 sentences).")
        # Optional extra fields if you want more detail:
        confidence: float = Field(..., ge=0, le=1, description="Confidence from 0 to 1.")

    instructions="""
    You are an expert evaluator assessing whether outputs are relevant to the given input. Your task is to determine whether EACH statement appropriately addresses what was asked.

    <Rubric>
    A relevant output:
    - Directly answers the question or addresses the request
    - Provides information specifically asked for
    - Stays on topic with the input's intent
    - Contributes meaningfully to fulfilling the request

    An irrelevant output:
    - Discusses topics not requested or implied by the input
    - Provides unnecessary tangents or digressions
    - Includes information that doesn't answer the question
    - Addresses a different question than what was asked
    </Rubric>

    <Instructions>
    For each output:
    - Read the original input carefully to understand what was asked
    - Examine the output and identify its core claim or purpose
    - Determine if the output directly addresses the input's request
    - Assess whether the information helps fulfill what was asked
    - Determine the answer relevancy of output and output a score
    </Instructions>

    <Reminder>
    Focus on whether each statement helps answer the specific input question, not whether the statement is true or well-written. A statement can be factually correct but still irrelevant if it doesn't address what was asked.
    </Reminder>

    Now, grade the following example according to the above instructions:

    <example>
    <input>
    {inputs}
    </input>

    <output>
    {outputs}
    </output>
    </example>
    """

    msg = instructions.format(inputs=question, outputs=answer)

    # Call the LLM to judge the output
    response = _judge_call(msg, Response)

    # Return the boolean score
    parsed = response.output_parsed

    score = 1.0 if parsed.answer_is_relevant else 0.0

    comment = (
        f"rationale: {parsed.rationale}\n"
        f"confidence: {parsed.confidence}\n"
        f""
    )

    return {
        "key": "answer_relevance_binary",
        "score": score,
        "comment": comment,
        # Optional: attach extra structured info (can help debugging)
        "extra": {
            "answer_is_relevant": parsed.answer_is_relevant,
            "confidence": parsed.confidence,
        }
    }


def LLM_judge_answer_faithfulness(run, example=None):
    """
    Use an LLM to judge if the contextualized response generated is faithfull
    to the retrieved context
    
    :param inputs: Description
    :param gold: Description
    """
     # Prefer the original question if available; fallback to prompt
    if example is not None:
        answerable = (example.outputs or {}).get("answerable", True)
        if not answerable:
            return {
                "key": "answer_faithfulness_binary",
                "score": None,
                "comment": "Skipped: example is marked answerable=false",
            }

    question = None
    if hasattr(run, "inputs") and isinstance(run.inputs, dict):
        question = run.inputs.get("question")
    if not question:
        question = run.outputs.get("question") or run.outputs.get("prompt") or ""

    answer = run.outputs.get("answer")
    documents = run.outputs.get("retrieved") or []

    # Keep message readable: show IDs/snippets instead of dumping huge content
    docs_for_judge = compact_docs(documents)

    class Response(BaseModel):
        answer_is_faithful: bool = Field(..., description="True if the generated answer is faithful to the retrieved context")
        rationale: str = Field(..., description="Brief reason for the decision (1-3 sentences).")
        # Optional extra fields if you want more detail:
        confidence: float = Field(..., ge=0, le=1, description="Confidence from 0 to 1.")

    instructions="""
    You are an expert data labeler evaluating model outputs for hallucinations. Your task is to assign a score based on the following rubric:

    <Rubric>
    A response without hallucinations:
    - Contains only verifiable facts that are directly supported by the input context
    - Makes no unsupported claims or assumptions
    - Does not add speculative or imagined details
    - Maintains perfect accuracy in dates, numbers, and specific details
    - Appropriately indicates uncertainty when information is incomplete
    </Rubric>

    <Instructions>
    - Read the input context thoroughly
    - Identify all claims made in the output
    - Cross-reference each claim with the input context
    - Note any unsupported or contradictory information
    - Consider the severity and quantity of hallucinations
    </Instructions>

    <Reminder>
    Focus solely on factual accuracy and support from the input context. Do not consider style, grammar, or presentation in scoring. A shorter, factual response should score higher than a longer response with unsupported claims.
    </Reminder>

    Use the following context to help you evaluate for hallucinations in the output:

    <context>
    {context}
    </context>

    <input>
    {inputs}
    </input>

    <output>
    {outputs}
    </output>
    """

    msg = instructions.format(context=docs_for_judge, inputs=question, outputs=answer)

    # Call the LLM to judge the output
    response = _judge_call(msg, Response)

    # Return the boolean score
    parsed = response.output_parsed

    score = 1.0 if parsed.answer_is_faithful else 0.0

    comment = (
        f"rationale: {parsed.rationale}\n"
        f"confidence: {parsed.confidence}\n"
        f""
    )

    return {
        "key": "answer_faithfulness_binary",
        "score": score,
        "comment": comment,
        # Optional: attach extra structured info (can help debugging)
        "extra": {
            "answer_is_faithful": parsed.answer_is_faithful,
            "confidence": parsed.confidence,
        }
    }


# create another evaluator that assigns 1 if the example was identified correctly as non answerable and 0 if not.
def LLM_judge_answer_correctness(run, example=None):
    """
    Use an LLM to judge if the contextualized response generated is relevant to the
    user's query.
    
    :param inputs: Description
    :param gold: Description
    """
     # Prefer the original question if available; fallback to prompt
    if example is not None:
        answerable = (example.outputs or {}).get("answerable", True)
        if not answerable:
            return {
                "key": "answer_correctness_binary",
                "score": None,
                "comment": "Skipped: example is marked answerable=false",
            }

    # question input into the model (not gold standard)
    question = None
    if hasattr(run, "inputs") and isinstance(run.inputs, dict):
        question = run.inputs.get("question")
    if not question:
        question = run.outputs.get("question") or run.outputs.get("prompt") or ""
    
    # model answer:
    answer = run.outputs.get("answer")

    # sample response:
    sample_response = example.outputs.get("reference_answer")

    class Response(BaseModel):
        answer_is_correct: bool = Field(..., description="True if the response is correct compared to the sample response")
        rationale: str = Field(..., description="Brief reason for the decision (1-3 sentences).")
        # Optional extra fields if you want more detail:
        confidence: float = Field(..., ge=0, le=1, description="Confidence from 0 to 1.")

    instructions = """
    You are an expert evaluator assessing the correctness of a model-generated response compared to a sample response (ground truth).

    Your task is to determine whether the model response is factually correct relative to the sample response.

    <Rubric>

    A correct response:
    - Accurately answers the user’s query
    - Matches the key facts and conclusions in the sample response
    - Contains no contradictions with the sample response
    - Does not introduce incorrect factual information
    - Provides the essential information required by the query

    An incorrect response:
    - Contradicts the sample response on key facts
    - Contains major factual errors
    - Omits essential required information
    - Answers a different question
    - Introduces false or misleading claims

    </Rubric>

    <Instructions>
    - Read the user query carefully
    - Read the sample response to understand the ground-truth answer
    - Identify the key factual claims in the model response
    - Compare each claim against the sample response
    - Determine whether the model response is fully correct or not

    </Instructions>

    <Reminder>
    Focus strictly on factual correctness relative to the sample response.
    Do not evaluate writing style or fluency.
    Minor wording differences are acceptable if the factual meaning is preserved.
    Any major contradiction or key factual error should result in "incorrect".
    If essential information is missing, the response should be marked "incorrect".

    </Reminder>

    Now evaluate the following:

    <query>
    {query}
    </query>

    <sample_response>
    {sample_response}
    </sample_response>

    <model_response>
    {model_response}
    </model_response>

    Return your judgment in structured JSON format.
    """


    msg = instructions.format(query=question, sample_response=sample_response, model_response=answer)

    # Call the LLM to judge the output
    response = _judge_call(msg, Response)

    # Return the boolean score
    parsed = response.output_parsed

    score = 1.0 if parsed.answer_is_correct else 0.0

    comment = (
        f"rationale: {parsed.rationale}\n"
        f"confidence: {parsed.confidence}\n"
        f""
    )

    return {
        "key": "answer_correctness_binary",
        "score": score,
        "comment": comment,
        # Optional: attach extra structured info (can help debugging)
        "extra": {
            "answer_is_correct": parsed.answer_is_correct,
            "confidence": parsed.confidence,
        }
    }