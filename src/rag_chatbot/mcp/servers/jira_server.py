from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP


# Initialize FastMCP server
mcp = FastMCP("jira_server")

BASE_DOMAIN = "alexhanna413"
JIRA_API_BASE = f"https://{BASE_DOMAIN}.atlassian.net/rest/api/3/issue" # /issue is for creating issues


async def make_jira_request(url: str, payload: dict, ):
    headers = {}
    async with httpx.AsyncClient as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return Exception
    


@mcp.tool()
async def create_ticket():
    pass