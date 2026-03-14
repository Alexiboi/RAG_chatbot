import asyncio
import json
from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
from dotenv import load_dotenv
from openai import OpenAI

DEPLOYMENT_NAME = "gpt-5.2-chat"


load_dotenv()

AZURE_OPENAI_API_KEY=os.getenv("AZURE_OPENAI_API_KEY")


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.client = OpenAI(
            base_url="https://alex-mltg6myf-eastus2.openai.azure.com/openai/v1/",
            api_key=AZURE_OPENAI_API_KEY
        )

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        # asking MCP server what tools it has:
        tool_response = await self.session.list_tools()
        # convert to OpenAI format
        available_tools = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            }
            for tool in tool_response.tools
        ]

        final_text = []

        # model receives user message + List of available tools 
        response = self.client.responses.create(
            model=DEPLOYMENT_NAME,
            input=[{"role": "user", "content": query}],
            tools=available_tools,
            max_output_tokens=1000,
        )

        # response of LLM will be the type (function_call e.g.)
        # name and arguments of the tool(s) it wants to call 

        # loop allows the model to call mutliple tools
        while True:
            had_tool_call = False
            next_inputs = []

            for item in response.output:
                # message is just text the model responded with
                if item.type == "message":
                    for part in item.content:
                        if part.type == "output_text":
                            # extract the text an add it to final answer
                            final_text.append(part.text)

                # function called a tool
                elif item.type == "function_call":
                    had_tool_call = True
                    tool_name = item.name
                    tool_args = json.loads(item.arguments)
                    # format of item:
                    # {
                    # "name": "create_jira_issue",
                    # "arguments": {
                    #     "summary": "...",
                    #     "description": "..."
                    # }

                    # call MCP tool: create_jira_issue(summary, description)
                    result = await self.session.call_tool(tool_name, tool_args)

                    if hasattr(result, "content"):
                        #output from MCP tool called
                        tool_output = str(result.content) 
                    else:
                        tool_output = str(result)

                    next_inputs.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": tool_output,
                    })

            # If a tool was called the loop starts again and sets has_tool_call to false again
            # so loop breaks when no more tools are left to call or no tools where ever called
            if not had_tool_call:
                break
            
            # the previous response ID allows the model to remember the user question,
            # tool call and tool result
            response = self.client.responses.create(
                model=DEPLOYMENT_NAME,
                previous_response_id=response.id,
                input=next_inputs,
                tools=available_tools,
                max_output_tokens=1000,
            )

        return "\n".join(final_text).strip()


    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    path_to_server = r"C:\Users\alexh\Desktop\LLM_uni_project\RAG_chatbot\src\rag_chatbot\mcp\servers\jira_server.py"
 

    client = MCPClient()
    try:
        await client.connect_to_server(path_to_server)
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())