
import json
from typing import Literal
from pydantic import BaseModel, Field
from src.rag_chatbot.rag.retrieval_utils import retrieve_context
from src.rag_chatbot.rag.env import deployment_name, client
from src.rag_chatbot.mcp.servers.clients.MCPClient import MCPClient
from pathlib import Path

HISTORY_LEN = 6

class QueryRoute(BaseModel):
    source: Literal["general", "rag", "mcp", "rag_then_mcp"] = Field(
        description="Which category should the user query be placed in"
    )

class MCPRoute(BaseModel):
    source: Literal["mcp", "rag_then_mcp"] = Field(
        description="Whether the request needs direct tool use only, or retrieval first then tool use"
    )

class GeneralLLM:
    """
    General LLM just straight call to GPT
    """

    @staticmethod
    def generate_answer(user_query: str, history: list[dict]) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful internal assistant for company employees. "
                    "Your primary role is to guide users on how to use this AI system effectively. "

                    "By default, explain what the system can do and how to use its different modes: "
                    "- General mode: for explanations, brainstorming, drafting, and general questions "
                    "- RAG mode: for retrieving and answering questions based on internal documents such as meeting notes and transcripts "
                    "- MCP mode: for performing actions in external systems such as creating Jira issues "

                    "If a user's request is unclear or general (e.g. 'help', 'what can you do'), respond with a clear explanation of these capabilities. "

                    "If the user asks a specific question, answer it clearly and professionally, "
                    "but also guide them toward the most appropriate mode if relevant. "

                    "You do not have access to documents or external tools in this mode, so do not claim to retrieve data or perform actions. "
                    "Instead, explain which mode should be used when needed."
                )
            }
        ]

        for msg in history[-HISTORY_LEN:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        messages.append({"role": "user", "content": user_query})

        response = client.chat.completions.create(
            model=deployment_name,
            messages=messages
        )
      
        return response.choices[0].message.content
    

class MCPLLM:
    server_module = "rag_chatbot.mcp.servers.jira_server"
    
    def __init__(self):
        self.client = MCPClient()
        self.connected = False
    
    async def connect_to_MCPserver(self):
        if self.connected:
            return
        try:
            await self.client.connect_to_server(self.server_module)
            self.connected = True
        except Exception as e:
            print(f"Exception occured {e}")
            raise
    
    async def generate_answer(self, user_query: str, history: list[dict]) -> str:
        messages = []

        for msg in history[-HISTORY_LEN:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        messages.append({"role": "user", "content": user_query})

        await self.connect_to_MCPserver()
        
        # message passed in includes history
        response = await self.client.process_query(query=messages)
        
        return response
    async def cleanup(self):
        if self.connected:
            await self.client.cleanup()
            self.connected = False

class RAGLLM:
    @staticmethod
    def generate_answer(user_query: str, context: list[dict], history: list[dict]) -> str:
        
        
        context_texts = [doc["content"] for doc in context]

        context_block = "\n\n---\n\n".join(context_texts)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Use the retrieved document context when relevant. "
                    "If the user is asking specifically about documents and the answer "
                    "is not in them, say there is not enough information."
                )
            }
        ]

        for msg in history[-HISTORY_LEN:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        messages.append({
            "role": "user",
            "content": f"""
                    Retrieved context:
                    {context_block}

                    User question:
                    {user_query}

                    Answer:
                    """
        })

        response = client.chat.completions.create(
            model=deployment_name,
            messages=messages
        )

        return response.choices[0].message.content
       

def decide_mcp_subroute(user_query: str) -> str:
    prompt = f"""
    Classify this user request into one of:
    - mcp
    - rag_then_mcp

    Definitions:
    - mcp: the request can be completed directly using tools/actions
    - rag_then_mcp: the request first needs company document retrieval, then a tool action

    User query: {user_query}

    Return only one of:
    - mcp
    - rag_then_mcp
    """

    response = client.responses.parse(
        model=deployment_name,
        input=prompt,
        text_format=MCPRoute,
    )

    return response.output_parsed.source

def get_routing_prompt(user_query: str) -> str:
    prompt = f"""
    Classify this user request into one of the following categories:
    - general
    - rag
    - mcp
    - rag_then_mcp

    Definitions:
    - general: can be answered without company documents or tool actions
    - rag: requires searching company documents
    - mcp: requires taking an external action using a tool
    - rag_then_mcp: requires searching documents first, then taking an action

    User query: {user_query}

    - Return "general" if the query can be answered without external documents or tool actions
    - Return "rag" if the query explicitly states to retrieve or find information using external documents
    - Return "mcp" If the query specifies taking an external action using a tool
    - Return "rag_then_mcp" If the query requires searching documents first, then taking an action. For example the query:
    "Create Jira tickets from today's meeting notes" should be classified as "rag_then_mcp" because meeting notes can only be
    accessed by RAG, and to perform the action MCP is needed.
    """
    return prompt

def decide_route(user_query: str, mode: str = "auto") -> str:
    # if the user has specified the query mode which is anything else but auto
    # the route should be returned as their decision, and if their choice is to use mcp,
    # the llm should decide whether their prompt requires mcp and rag or just mcp#
    if mode != "auto":
        match mode:
            case "llm":
                return "general"
            case "rag":
                return "rag"
            case "mcp":
                return decide_mcp_subroute(user_query)
            case _:
                pass

    prompt = get_routing_prompt(user_query)

    response = client.responses.parse(
        model=deployment_name, # gpt-5.2-chat
        input=prompt,
        text_format=QueryRoute,
    )

    return response.output_parsed.source

def build_grounded_task(user_query: str, context: list[dict]):
    context_texts = [doc["content"] for doc in context]
    context_block = "\n\n---\n\n".join(context_texts)

    prompt = f"""
    You are preparing an action request for a tool-using assistant.

    Use the document context below to interpret the user's request.
    Create a precise instruction for the tool agent.

    Requirements:
    - Include only actions supported by the user request
    - Preserve names, dates, epics, stories, tasks, and priorities where present
    - Do not invent details not supported by the context
    - If multiple tasks are needed, list them clearly in one concise instruction.

    Context:
    {context_block}

    User request:
    {user_query}

    Tool-ready instruction:
    """
    response = client.responses.parse(
        model=deployment_name, # gpt-5.2-chat
        input=prompt,
    )
    return response.output_text

async def handle_chat(user_query: str, history: list[dict], mode: str = "auto") -> dict:
    """
    depending on route returned will call the corresponding method 
    """
    route = decide_route(user_query, mode)

    if route == "general":
        return {
            "answer": GeneralLLM.generate_answer(user_query=user_query, history=history),
            "mode": "general",
        }
     
    

    elif route == "rag":
        context = retrieve_context(user_query)
        # fallback in case there is no context for some reason
        if not context:
            return {
                "answer": "I couldn't find relevant information in the documents for that request.",
                "mode": "rag",
                "retrieved": []
            }
        
        return {
            "answer": RAGLLM.generate_answer(user_query=user_query, context=context, history=history),
            "mode": "rag",
            "retrieved": context
        }

    elif route == "mcp":
        mcp_llm = MCPLLM()
        try:
                
            return {
                "answer": await mcp_llm.generate_answer(user_query=user_query, history=history),
                "mode": "mcp",
            }
        finally:
            await mcp_llm.cleanup()

    elif route == "rag_then_mcp":
        mcp_llm = MCPLLM()
        try:
            context = retrieve_context(user_query)
            grounded_task = build_grounded_task(user_query, context)

            # the mcp client needs the query to be in a string format 
            # This maintains structure while keeping the output a string
            if not isinstance(grounded_task, str):
                grounded_task = json.dumps(grounded_task)

            mcp_result = await mcp_llm.generate_answer(user_query=grounded_task, history=history)
            return {
                "answer": mcp_result,
                "mode": "rag_then_mcp",
                "retrieved": context,
                "grounded_task": grounded_task
            }
        finally:
            await mcp_llm.cleanup()
    else:
        return {
            "answer": GeneralLLM.generate_answer(user_query=user_query, history=history),
            "mode": "general",
        }

def generate_response(context: list[dict], user_query: str) -> str:
    context_texts = [doc["content"] for doc in context]

    context_block = "\n\n---\n\n".join(context_texts)

    system_prompt = """
    You are a helpful assistant.

    Use the retrieved document context when it is relevant to the user's question.
    If the user's question is about the retrieved documents, answer from that context and do not invent missing facts.
    If the retrieved context is irrelevant or insufficient and the user is asking a general question, answer using your general knowledge.
    If the user is specifically asking about the documents and the answer is not contained in them, say that the documents do not contain enough information.

    When useful, make it clear whether your answer is based on the documents or on general knowledge.
    """

    user_prompt = f"""
    Retrieved context:
    {context_block}

    User question:
    {user_query}

    Answer:
    """
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        #temperature=0.2
    )

    return response.choices[0].message.content



def generate_contextualized_response(inputs: dict) -> dict:
    user_query = inputs["question"]

    user_query = user_query.strip()
    context_results = retrieve_context(user_query)
    answer = generate_response(context_results, user_query)
    return {
        "answer": answer,
        "prompt": user_query,
        "retrieved": context_results
    }

async def chat_loop(user_query: str, history: list[dict] | None = None, mode: str = "auto"):
    history = history or []
    return await handle_chat(user_query, history, mode)