from pydantic import BaseModel, Field

from src.rag_chatbot.rag.env import client, deployment_name


class LLMChunker:
    instructions = """
    <document>
    {WHOLE_DOCUMENT}
    </document>

    <chunk>
    {CHUNK_CONTENT}
    </chunk>

    Write 1–2 sentences of retrieval-oriented context for this chunk.

    The context should identify:
    1. the main topic of the chunk,
    2. where it sits in the document hierarchy,
    3. any parent epic or story if explicitly present in the document,
    4. the type of content (e.g. story description, acceptance criteria, implementation note, meeting discussion, action item).

    Prefer explicit document structure over vague summary.
    Do not invent missing hierarchy.
    Return only the context. 
    """
    
    def __init__(self):
        self.client = client


    def return_response(self, document: str, chunk: str):
        msg = self.instructions.format(
            WHOLE_DOCUMENT=document,
            CHUNK_CONTENT=chunk)

        response = self.client.responses.parse(
            model=deployment_name,
            input=msg,
            #text_format=response_model,
        )
        return response.output_text
    
class Response(BaseModel):
        # output can be relevance, faithfulness, correctness
        output: bool = Field(..., description="True or False depending on the judge criteria")
        rationale: str = Field(..., description="Brief reason for the decision (1-3 sentences).")