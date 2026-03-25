import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.rag_chatbot.mcp.servers import jira_server
from src.rag_chatbot.mcp.servers.jira_server import (
	make_jira_issue_request,
	jira_description,
	get_all_projects,
	list_jira_projects,
	create_jira_issue,
	JIRA_API_BASE,
)


# ==================== FIXTURES ====================


@pytest.fixture
def event_loop():
	"""Use a fresh event loop for async tests in this module."""
	loop = asyncio.new_event_loop()
	yield loop
	loop.close()


@pytest.fixture
def sample_summary():
	return "Test issue summary"


@pytest.fixture
def sample_description():
	return "Detailed description of the test issue."


@pytest.fixture
def sample_proj_key():
	return "KAN"


# ==================== TEST jira_description ====================


class TestJiraDescription:
	def test_jira_description_structure(self):
		"""jira_description should wrap text in the expected Atlassian document format."""
		text = "Some description"
		result = jira_description(text)

		assert result["type"] == "doc"
		assert result["version"] == 1
		assert isinstance(result["content"], list)
		paragraph = result["content"][0]
		assert paragraph["type"] == "paragraph"
		inner = paragraph["content"][0]
		assert inner["type"] == "text"
		assert inner["text"] == text


# ==================== TEST make_jira_issue_request ====================


class TestMakeJiraIssueRequest:
	@pytest.mark.asyncio
	@patch("src.rag_chatbot.mcp.servers.jira_server.httpx.AsyncClient")
	async def test_success_returns_json_response(
		self,
		mock_async_client_cls,
		sample_proj_key,
		sample_summary,
		sample_description,
	):
		"""On success, make_jira_issue_request should return response.json() and call Jira with expected payload."""
		# Setup AsyncClient context manager
		client = AsyncMock()
		response = MagicMock()
		response.json.return_value = {"key": "KAN-123"}
		response.raise_for_status.return_value = None
		client.post = AsyncMock(return_value=response)

		cm = AsyncMock()
		cm.__aenter__.return_value = client
		cm.__aexit__.return_value = None
		mock_async_client_cls.return_value = cm

		with patch(
			"src.rag_chatbot.mcp.servers.jira_server.jira_description",
			return_value={"formatted": True},
		) as mock_jira_desc:
			result = await make_jira_issue_request(
				proj_key=sample_proj_key,
				summary=sample_summary,
				description=sample_description,
			)

		# Returned JSON
		assert result == {"key": "KAN-123"}

		# Correct URL and payload
		client.post.assert_awaited_once()
		call_kwargs = client.post.call_args.kwargs
		assert call_kwargs["url"] == f"{JIRA_API_BASE}/issue"
		assert call_kwargs["headers"]["Accept"] == "application/json"
		assert call_kwargs["headers"]["Content-Type"] == "application/json"

		payload = call_kwargs["json"]
		assert payload["fields"]["project"]["key"] == sample_proj_key
		assert payload["fields"]["summary"] == sample_summary
		assert payload["fields"]["issuetype"]["name"] == "Task"
		mock_jira_desc.assert_called_once_with(sample_description)
		assert payload["fields"]["description"] == {"formatted": True}

	@pytest.mark.asyncio
	@patch("src.rag_chatbot.mcp.servers.jira_server.httpx.AsyncClient")
	async def test_error_returns_string_message(self, mock_async_client_cls):
		"""If the HTTP call raises, make_jira_issue_request should return the exception string."""
		client = AsyncMock()
		client.post = AsyncMock(side_effect=Exception("boom"))

		cm = AsyncMock()
		cm.__aenter__.return_value = client
		cm.__aexit__.return_value = None
		mock_async_client_cls.return_value = cm

		result = await make_jira_issue_request(
			proj_key="KAN",
			summary="S",
			description="D",
		)

		assert "boom" in result


# ==================== TEST get_all_projects ====================


class TestGetAllProjects:
	@pytest.mark.asyncio
	@patch("src.rag_chatbot.mcp.servers.jira_server.httpx.AsyncClient")
	async def test_success_returns_json(self, mock_async_client_cls):
		"""On success, get_all_projects should return the JSON body."""
		client = AsyncMock()
		response = MagicMock()
		response.json.return_value = {"values": []}
		response.raise_for_status.return_value = None
		client.get = AsyncMock(return_value=response)

		cm = AsyncMock()
		cm.__aenter__.return_value = client
		cm.__aexit__.return_value = None
		mock_async_client_cls.return_value = cm

		result = await get_all_projects()

		assert result == {"values": []}
		client.get.assert_awaited_once()

	@pytest.mark.asyncio
	@patch("src.rag_chatbot.mcp.servers.jira_server.httpx.AsyncClient")
	async def test_error_returns_string(self, mock_async_client_cls):
		"""If the HTTP call fails, get_all_projects should return the exception string."""
		client = AsyncMock()
		client.get = AsyncMock(side_effect=Exception("nope"))

		cm = AsyncMock()
		cm.__aenter__.return_value = client
		cm.__aexit__.return_value = None
		mock_async_client_cls.return_value = cm

		result = await get_all_projects()

		assert "nope" in result


# ==================== TEST list_jira_projects (MCP tool wrapper) ====================


class TestListJiraProjects:
	@pytest.mark.asyncio
	@patch("src.rag_chatbot.mcp.servers.jira_server.get_all_projects", new_callable=AsyncMock)
	async def test_returns_simplified_project_list(self, mock_get_all_projects):
		"""list_jira_projects should transform raw project JSON into a simplified list."""
		mock_get_all_projects.return_value = {
			"values": [
				{"key": "KAN", "name": "Kanban", "id": "1", "other": "x"},
				{"key": "ENG", "name": "Engineering", "id": "2", "foo": "bar"},
			]
		}

		result = await list_jira_projects()

		assert isinstance(result, list)
		assert result == [
			{"key": "KAN", "name": "Kanban", "id": "1"},
			{"key": "ENG", "name": "Engineering", "id": "2"},
		]
		mock_get_all_projects.assert_awaited_once()

	@pytest.mark.asyncio
	@patch("src.rag_chatbot.mcp.servers.jira_server.get_all_projects", new_callable=AsyncMock)
	async def test_handles_falsy_response(self, mock_get_all_projects):
		"""If Jira returns no data, list_jira_projects should return an error string."""
		mock_get_all_projects.return_value = None

		result = await list_jira_projects()

		assert result == "Unable to make Jira Request"


# ==================== TEST create_jira_issue (MCP tool) ====================


class TestCreateJiraIssue:
	@pytest.mark.asyncio
	@patch("src.rag_chatbot.mcp.servers.jira_server.make_jira_issue_request", new_callable=AsyncMock)
	async def test_success_formats_issue_message(
		self,
		mock_make_request,
		sample_summary,
		sample_description,
		sample_proj_key,
		monkeypatch,
	):
		"""On success, create_jira_issue should return a formatted string containing key and URL."""
		mock_make_request.return_value = {"key": "KAN-999"}
		# Control domain so URL format is deterministic
		monkeypatch.setattr(jira_server, "JIRA_DOMAIN", "example-domain")

		result = await create_jira_issue(
			summary=sample_summary,
			description=sample_description,
			proj_key=sample_proj_key,
		)

		mock_make_request.assert_awaited_once_with(
			proj_key=sample_proj_key,
			summary=sample_summary,
			description=sample_description,
		)

		# Check key and URL appear in the returned string
		assert "KAN-999" in result
		assert "https://example-domain.atlassian.net/" in result
		assert "/browse/KAN-999" in result

	@pytest.mark.asyncio
	@patch("src.rag_chatbot.mcp.servers.jira_server.make_jira_issue_request", new_callable=AsyncMock)
	async def test_handles_falsy_response(self, mock_make_request, sample_summary, sample_description, sample_proj_key):
		"""If Jira request fails, create_jira_issue should return an error string."""
		mock_make_request.return_value = None

		result = await create_jira_issue(
			summary=sample_summary,
			description=sample_description,
			proj_key=sample_proj_key,
		)

		assert result == "Unable to make Jira Request"

