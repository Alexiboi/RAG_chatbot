
import json
from typing import Literal
from pydantic import BaseModel, Field
from src.backend.rag.retrieval_utils import retrieve_context
from src.backend.rag.env import deployment_name, client
from src.backend.mcp.servers.clients.MCPClient import MCPClient

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
    General LLM just behaves like an internal company assistant but cannot perform
    any external actions or document retrieval capabilities.
    """

    @staticmethod
    def generate_answer(user_query: str, history: list[dict]) -> str:
        """
        Generates an answer behaving as a general assistant (no RAG or MCP), answer is generated based on a query and history if it exists

        Args:
            user_query (str): The user's query
            history (list[dict]): chat history as a list of {"role":"user or system", "content":"message"}
        
        Returns:
            str: The client's response in string, text format
        """
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
        
        # appends last (HISTORY_LEN (6)) messages to messages array, messages sent to model
        for msg in history[-HISTORY_LEN:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        # append the user query as the last message
        messages.append({"role": "user", "content": user_query})

        # The client expects the last message to be the latest user_query so it should be last in the list
        response = client.chat.completions.create(
            model=deployment_name,
            messages=messages
        )
        
        # choices is a list of possible responses, choices[0] is the first one.
        return response.choices[0].message.content
    

class MCPLLM:
    """Client for interacting with an MCP server using an LLM.

    This class manages:
    - Connection to an MCP server
    - Formatting of user query + conversation history
    - Sending requests via MCPClient

    Attributes:
        server_module (str): Path to the MCP server module.
        client (MCPClient): Underlying MCP client instance.
        connected (bool): Tracks whether the client is connected.
    """

    server_module = "src.backend.mcp.servers.jira_server"

    def __init__(self):
        self.client = MCPClient()
        self.connected = False
    
    async def connect_to_MCPserver(self) -> None:
        """
        client attempts to connect to mcp server if it has not already connected
        """
        if self.connected:
            return
        try:
            await self.client.connect_to_server(self.server_module)
            self.connected = True
        except Exception as e:
            print(f"Exception occured {e}")
            raise
    
    async def generate_answer(self, user_query: str, history: list[dict]) -> str:
        """
        returns response of mcp client which has access to tools provided by mcp server (jira_server.py)

        Args:
            user_query (str): The user's query
            history (list[dict]): chat history as a list of {"role":"user or system", "content":"message"}

        Returns:
            str: response from mcp client
        """
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
    async def cleanup(self) -> None:
        """
        calls the cleanup function (only if the client is currently connected) from the MCP client and sets connected back to false
        """
        if self.connected:
            await self.client.cleanup()
            self.connected = False

class RAGLLM:
    """
    Client for generating answers using retrieved context (does not perform retrieval)
    """
    @staticmethod
    def generate_answer(user_query: str, context: list[dict], history: list[dict]) -> str:
        """
        Generates an answer using the provided documents exclusively.
        If they do not contain enough information to answer the question the LLM should not hallucinate

        Args:
            user_query (str): The user's query
            context (list[dict]): list of texts representing retrieved chunks of context
            history (list[dict]): chat history as a list of {"role":"user or system", "content":"message"}
        """
        
        context_texts = [doc["content"] for doc in context]

        context_block = "\n\n---\n\n".join(context_texts) # join to form one big string for ingestion into message to LLM

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
    """
    decides mode of LLM that will generate answer based on purely the user query
    or from the mode purposely chosen by the user.

    Args:
        user_query (str): The user's input query
        mode (str): one of [auto (let the LLM decide), llm (general), rag, mcp,]
    
    Returns:
        str: one of [general, rag, mcp, rag_then_mcp]
    """
    # if mode is not auto we either return that mode as selected by the user,
    # or if the mode == "mcp" then LLM decides if mcp or rag_then_mcp #
    if mode != "auto":
        match mode:
            case "llm":
                return "general"
            case "rag":
                return "rag"
            case "mcp":
                return decide_mcp_subroute(user_query)
            case _: 
                # If the mode is not auto but also does not match any of the options it should default to auto indicated by a pass which moves on
                pass

    # Reach here if mode == "auto" or not in [llm, rag, mcp]
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

    Args:
        user_query (str): The User's input query
        history (list[dict]): chat history as a list of {"role":"user or system", "content":"message"}
        mode: (str): one of [auto (let the LLM decide), llm (general), rag, mcp]
    Returns:
        dict: Containing fields: "answer", "mode", "retrieved" (optional), "grounded task" (optional)
    """
    route = decide_route(user_query, mode)

    if route == "general":
        return {
            "answer": GeneralLLM.generate_answer(user_query=user_query, history=history),
            "mode": "general",
        }
     
    elif route == "rag":
        context = retrieve_context(user_query, k=6)
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
            # This maintains a json format while keeping the output a string
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


async def chat_loop(user_query: str, history: list[dict] | None = None, mode: str = "auto"):
    history = history or []
    return await handle_chat(user_query, history, mode)