import httpx
from mcp.server.fastmcp import FastMCP
import os
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

load_dotenv()


# Initialize FastMCP server
mcp = FastMCP("jira_server")
JIRA_DOMAIN = os.getenv("JIRA_DOMAIN")
JIRA_API_KEY = os.getenv("JIRA_API_TOKEN")
JIRA_API_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_BASE = f"https://{JIRA_DOMAIN}/rest/api/3" 

async def make_jira_issue_request(proj_key: str="KAN", summary: str="", description: str="") -> dict | str:
    """
    Send a HTTP request to Jira to create a new issue.

    this function:
    - Builds the correct Jira api payload
    - Authenticates using email + API token
    - Sends the request asynchronously using httpx
    - Returns the JSON response (issue key, id, etc.) or an error string

    Args:
        proj_key (str): Jira project key (e.g. "KAN")
        summary (str): Title of the issue
        description (str): Description text for the issue

    Returns:
        dict | str:
            - dict: Successful JSON response from Jira API
            - str: Error message if request fails
    """
    url = f"{JIRA_API_BASE}/issue" # /issue is for creating issues
    headers = {
    "Accept": "application/json", # content type client receives
    "Content-Type": "application/json" # content type of request client sends
    }
    auth = HTTPBasicAuth(JIRA_API_EMAIL, JIRA_API_KEY)
    payload = {
        "fields": {
            "project": {
                "key": proj_key
            },
            "summary": summary,
            "description": jira_description(description),
            "issuetype": {
                "name": "Task"
            }
        }
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url=url,
                headers=headers,
                auth=auth,
                json=payload,
                timeout=30.0)
            response.raise_for_status()
            return response.json() # JSON response should contain id, key and url to issue
        except Exception as e:
            return str(e)
    
def jira_description(text: str) -> dict:
    """
    Takes text for description of a task and puts it in the correct format
    for the payload in the request

    Args:
        text (str): Plain text description
    Returns:
        dict: formatted description object
    """
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": text}
                ]
            }
        ]
    }

async def test():
    response = await list_jira_projects()
    print(response)

async def get_all_projects() -> dict | str:   
    """
    Fetch all Jira projects accessible to the authenticated user.

    Sends a GET request to Jira's project search endpoint.

    Returns:
        dict | str:
            - dict: JSON response containing project metadata
            - str: Error message if request fails
    """
    url = f"{JIRA_API_BASE}/project/search"
    headers = {
    "Accept": "application/json", # content type client receives
    #"Content-Type": "application/json" # content type of request client sends
    }
    auth = HTTPBasicAuth(JIRA_API_EMAIL, JIRA_API_KEY)
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                url=url,
                headers=headers,
                auth=auth,
                timeout=30.0)
            response.raise_for_status()
            return response.json() # JSON response should contain id, key and url to issue
        except Exception as e:
            return str(e)

@mcp.tool()
async def list_jira_projects() -> dict | str:
    """
    MCP tool: List all Jira projects available to the user.

    This function:
    - Calls Jira API to retrieve projects
    - Extracts key metadata (key, name, id)
    - Returns a simplified list for LLM/tool consumption

    Returns:
        list[dict] | str:
            - list of projects with keys: key, name, id
            - error string if request fails
    """
    response = await get_all_projects()
    if not response:
        return "Unable to make Jira Request"
    
    projects = [
        {
            "key": p["key"],
            "name": p["name"],
            "id": p["id"]
        }
        for p in response["values"]
    ]

    return projects

@mcp.tool()
async def create_jira_issue(summary: str, description: str, proj_key: str) -> str:
    """
    MCP tool: Create a Jira issue in a specified project.

    This function:
    - Calls Jira API to create a new issue
    - Formats a user-friendly response including issue URL

    Args:
        summary (str): Title of the issue
        description (str): Detailed description
        proj_key (str): Jira project key (e.g. "KAN")

    Returns:
        str:
            A formatted message containing:
            - issue key
            - direct URL to the issue
            - success message
            OR an error message if creation fails
    """

    jira_site = f"https://{JIRA_DOMAIN}.atlassian.net/"

    response = await make_jira_issue_request(
        proj_key=proj_key,
        summary=summary,
        description=description)
    
    if not response:
        return "Unable to make Jira Request"
    
    # Return string because LLM prefers a formatted string format for it's input
    return f"""
            "issue_key": {response['key']},
            "issue_url":{jira_site}/browse/{response['key']},
            "message": Jira issue **{response['key']}** created successfully"
        """

def main():
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()