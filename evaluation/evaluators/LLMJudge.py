from pydantic import BaseModel, Field


class LLMJudge:
    instructions = {
        "retrieval_relevance": """
        You are an expert evaluator assessing whether retrieved documents are relevant to a given user query.

        Your task is to determine whether the retrieved documents match the information need expressed in the query.

        <Rubric>

        Relevant retrieval:
        - Documents directly relate to the subject of the query
        - Documents contain information that could help answer the query
        - Documents address the specific entities, concepts, or constraints in the query
        - Documents are topically aligned with the query’s intent

        Irrelevant retrieval:
        - Documents discuss unrelated topics
        - Documents only partially match surface keywords but miss the query’s intent
        - Documents contain general background information without addressing the query
        - Documents would not meaningfully help answer the query

        </Rubric>

        <Instructions>

        For this evaluation:

        - Carefully read the query to understand the information being requested
        - Review the retrieved documents
        - Determine whether the documents match the query’s topic and intent
        - Assess whether the documents would help answer the query if used in generation
        - Judge overall retrieval relevance

        </Instructions>

        <Reminder>

        Focus on topical and semantic relevance to the query.
        Do NOT judge answer correctness.
        Do NOT evaluate writing quality.
        Only assess whether the retrieved documents match the query’s information need.

        </Reminder>

        Now evaluate the following:

        <Query>
        {query}
        </Query>

        <RetrievedDocuments>
        {documents}
        </RetrievedDocuments>

        Return your judgment in structured JSON format.""",
        "generation_relevance":"""
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
        """,
        "faithfulness": """
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
        """,
        "correctness": """
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
        """,
    }

    def __init__(self, run, example):
        from openai import OpenAI
        self.client = OpenAI()
        self.run = run
        self.example = example

    def compact_docs(self, docs, max_docs=6, max_chars=600):
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

    def returnResponse(self, msg: str, ResponseModel):
        response = self.client.responses.parse(
            model="gpt-4o",
            input=msg,
            text_format=ResponseModel,
        )
        return response.output_parsed
    
    def get_query(self):
        question = None
        if hasattr(self.run, "inputs") and isinstance(self.run.inputs, dict):
            question = self.run.inputs.get("question")
        if not question:
            question = self.run.outputs.get("question") or self.run.outputs.get("prompt") or ""

        return question
    
    def get_compact_documents(self):
        documents = self.run.outputs.get("retrieved") or []
        return self.compact_docs(documents)
    
    def get_answer(self):
        return self.run.outputs.get("answer")
    
    def get_sample_response(self):
        return self.example.outputs.get("reference_answer")
    
    def check_answerable(self, target: str):
        if self.example is not None:
            answerable = (self.example.outputs or {}).get("answerable", True)
            if not answerable:
                return {
                    "key": target,
                    "score": None,
                    "comment": "Skipped: example is marked answerable=false",
                }
            

    def judge(self):
        # Your LLM_judge_relevance logic here, using self.client
        pass


class RetrievalRelevanceJudge(LLMJudge):
     def judge(self):
        self.check_answerable("retrieval_relevance_binary")

        question = self.get_query()

        # Keep message readable: show IDs/snippets instead of dumping huge content
        docs_for_judge = self.get_compact_documents()

        msg = self.instructions["retrieval_relevance"].format(query=question, documents=docs_for_judge)

        # Call the LLM to judge the output
        response = self.returnResponse(msg, Response)

        # response is already parsed
        score = 1.0 if response.output else 0.0

        return {
            "key": "retrieval_relevance_binary",
            "score": score,
            "comment": (
                f"rationale: {response.rationale}\n"
                f""
            ),
            # Optional: attach extra structured info (can help debugging)
            "extra": {
                "question_used": question,
                "retrieved_docs": docs_for_judge,
            }
        }
     
class AnswerRelevanceJudge(LLMJudge):
     def judge(self):
        self.check_answerable("answer_relevance_binary")
        question = self.get_query()

        # Keep message readable: show IDs/snippets instead of dumping huge content
        answer = self.get_answer()

        msg = self.instructions["generation_relevance"].format(inputs=question, outputs=answer)

        # Call the LLM to judge the output
        response = self.returnResponse(msg, Response)

        # response is already parsed
        score = 1.0 if response.output else 0.0

        return {
            "key": "answer_relevance_binary",
            "score": score,
            "comment": (
                f"rationale: {response.rationale}\n"
                f""
            ),
        }
     
class AnswerFaithfulnessJudge(LLMJudge):
     def judge(self):
        self.check_answerable("answer_faithfulness_binary")
        question = self.get_query()

        # Keep message readable: show IDs/snippets instead of dumping huge content
        answer = self.get_answer()

        docs_for_judge = self.get_compact_documents()

        msg = self.instructions["faithfulness"].format(context=docs_for_judge, inputs=question, outputs=answer)

        # Call the LLM to judge the output
        response = self.returnResponse(msg, Response)

        # response is already parsed
        score = 1.0 if response.output else 0.0

        return {
            "key": "answer_faithfulness_binary",
            "score": score,
            "comment": (
                f"rationale: {response.rationale}\n"
                f""
            ),
        }
     
class AnswerCorrectnessJudge(LLMJudge):
     def judge(self):
        self.check_answerable("answer_correctness_binary")
        question = self.get_query()
        # Keep message readable: show IDs/snippets instead of dumping huge content
        answer = self.get_answer()

        sample_response = self.get_sample_response()

        msg = self.instructions["correctness"].format(query=question, sample_response=sample_response, model_response=answer)

        # Call the LLM to judge the output
        response = self.returnResponse(msg, Response)

        # response is already parsed
        score = 1.0 if response.output else 0.0

        return {
            "key": "answer_correctness_binary",
            "score": score,
            "comment": (
                f"rationale: {response.rationale}\n"
                f""
            ),
        }
        

class Response(BaseModel):
        # output can be relevance, faithfulness, correctness
        output: bool = Field(..., description="True or False depending on the judge criteria")
        rationale: str = Field(..., description="Brief reason for the decision (1-3 sentences).")
