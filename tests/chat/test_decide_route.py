import pytest
from unittest.mock import patch, MagicMock

from src.rag_chatbot.rag.RAG_bot import decide_route

def test_decide_route_llm_mode():
    result = decide_route("What can this system do?", mode="llm")
    assert result == "general"

def test_decide_route_rag_mode():
    result = decide_route("Can you retrieve the documents for me", mode="rag")
    assert result == "rag"

@patch("src.rag_chatbot.rag.RAG_bot.client")
def test_decide_route_match_no_cases(mock_client):
    """
    handle edge case if user enters mode that does not match any cases.
    Should fallback to auto and call LLM classifier.

    asserts that the decide route client is called once
    """
    mock_response = MagicMock()
    mock_response.output_parsed.source = "rag"
    mock_client.responses.parse.return_value = mock_response

    result = decide_route("Find info about the Apple earning call from the documents", mode="invalid")
    
    assert result == "rag"
    mock_client.responses.parse.assert_called_once()

@patch("src.rag_chatbot.rag.RAG_bot.decide_mcp_subroute")
def test_decide_route_mcp_mode_returns_ragmcp(mock_decide_mcp_subroute):
    mock_decide_mcp_subroute.return_value = "rag_then_mcp"
    result = decide_route("Create Jira tickets from the meeting notes", mode="mcp")
    assert result == "rag_then_mcp"
    mock_decide_mcp_subroute.assert_called_once_with("Create Jira tickets from the meeting notes")

@patch("src.rag_chatbot.rag.RAG_bot.decide_mcp_subroute")
def test_decide_route_mcp_mode_returns_mcp(mock_decide_mcp_subroute):
    """
    Tests if the subroute returned is just mcp when the mcp subrout decider returns just mcp
    """
    mock_decide_mcp_subroute.return_value = "mcp"
    result = decide_route("Create Jira tickets", mode="mcp")
    assert result == "mcp"
    mock_decide_mcp_subroute.assert_called_once_with("Create Jira tickets")

@patch("src.rag_chatbot.rag.RAG_bot.client")
def test_decide_route_uses_correct_model(mock_client):
    """
    Tests that the model that decides the route has model name of 'gpt-5.2-chat'
    """
    mock_response = MagicMock()
    mock_response.output_parsed.source = "general"
    mock_client.responses.parse.return_value = mock_response

    decide_route("What can this system do?", mode="auto")

    mock_client.responses.parse.assert_called_once()

    _, kwargs = mock_client.responses.parse.call_args

    assert kwargs["model"] == "gpt-5.2-chat"