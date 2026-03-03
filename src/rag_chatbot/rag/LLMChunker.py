from pydantic import BaseModel, Field

from src.rag_chatbot.rag.env import client, deployment_name, notes_container_client


class LLMChunker:
    instructions = """
    <document> 
    {WHOLE_DOCUMENT} 
    </document> 
    Here is the chunk we want to situate within the whole document 
    <chunk> 
    {CHUNK_CONTENT} 
    </chunk> 
    Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
    Answer only with the succinct context and nothing else. 
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