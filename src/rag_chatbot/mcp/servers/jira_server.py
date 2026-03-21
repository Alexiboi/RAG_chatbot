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

async def make_jira_issue_request(proj_key: str="KAN", summary: str="", description: str=""):
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
    
def jira_description(text: str):
    """
    Takes text for description of a task and puts it in the correct format
    for the payload in the request
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

async def get_all_projects():   
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
async def list_jira_projects():
    """
    Lists all the spaces/projects key's and their name's currently visible to the user.
    A project key can for example be 'KAN'

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
    Create an Issue on Jira in a specific project/space specified by proj_key
    
    Args:
        summary: Title of the issue/ticket
        description: description of the issue
        proj_key: project key specifying the project we want to create an Issue/task in
    """
    jira_site = f"https://{JIRA_DOMAIN}.atlassian.net/"

    response = await make_jira_issue_request(
        proj_key=proj_key,
        summary=summary,
        description=description)
    if not response:
        return "Unable to make Jira Request"
    return f"""
            "issue_key": {response['key']},
            "issue_url":{jira_site}/browse/{response['key']},
            "message": Jira issue **{response['key']}** created successfully"
        """

def main():
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()