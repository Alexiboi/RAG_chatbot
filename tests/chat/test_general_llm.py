import pytest
from unittest.mock import patch, MagicMock

from src.rag_chatbot.rag.RAG_bot import GeneralLLM, HISTORY_LEN, deployment_name

@pytest.fixture
def mock_client():
    with patch("src.rag_chatbot.rag.RAG_bot.client") as mock:
        yield mock


@pytest.fixture
def mock_response():
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "test response"
    return response

def test_returns_llm_response(mock_client, mock_response):
    """Returns the content from the LLM response"""
    mock_client.chat.completions.create.return_value = mock_response

    result = GeneralLLM.generate_answer("Hello", [])

    assert result == "test response"


def test_calls_model_with_messages(mock_client, mock_response):
    """Calls the LLM with the correct parameters"""
    mock_client.chat.completions.create.return_value = mock_response

    GeneralLLM.generate_answer("Hello", [])

    mock_client.chat.completions.create.assert_called_once()

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs

    assert call_kwargs["model"] == deployment_name # gpt-5.2-chat
    assert isinstance(call_kwargs["messages"], list)

def test_empty_user_query(mock_client, mock_response):
    # Arrange
    mock_client.chat.completions.create.return_value = mock_response

    user_query = ""
    history = []

    GeneralLLM.generate_answer(user_query, history)

    messages = mock_client.chat.completions.create.call_args.kwargs["messages"]

    # Check that the empty query was still appended
    assert messages[-1] == {"role": "user", "content": ""}

    # Ensure the API was still called
    assert mock_client.chat.completions.create.called

def test_empty_history(mock_client, mock_response):
    mock_client.chat.completions.create.return_value = mock_response

    GeneralLLM.generate_answer("Hello", [])

    messages = mock_client.chat.completions.create.call_args.kwargs["messages"]

    assert len(messages) == 2  # system + user
    assert messages[-1]["content"] == "Hello"
    assert messages[-1]["role"] == "user"

def test_user_query_is_last_message(mock_client, mock_response):
    """User query is always the last message"""
    mock_client.chat.completions.create.return_value = mock_response

    GeneralLLM.generate_answer("Final question", [])

    messages = mock_client.chat.completions.create.call_args.kwargs["messages"]

    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "Final question"

def test_history_is_included_and_truncated(mock_client, mock_response):
    """Only the last HISTORY_LEN messages are included"""
    mock_client.chat.completions.create.return_value = mock_response

    history = [
        {"role": "user", "content": f"msg {i}"}
        for i in range(HISTORY_LEN + 3)
    ]

    GeneralLLM.generate_answer("Query", history)

    messages = mock_client.chat.completions.create.call_args.kwargs["messages"]

    # system + truncated history + user query
    assert len(messages) == HISTORY_LEN + 2

    history_messages = messages[1:-1]

    # ensure we took the last messages
    assert history_messages[0]["content"] == f"msg 3"


def test_system_message_exists(mock_client, mock_response):
    """System prompt is always included"""
    mock_client.chat.completions.create.return_value = mock_response

    GeneralLLM.generate_answer("Hello", [])

    messages = mock_client.chat.completions.create.call_args.kwargs["messages"]

    assert messages[0]["role"] == "system"
    assert isinstance(messages[0]["content"], str)
    assert len(messages[0]["content"]) > 0


def test_propagates_client_errors(mock_client):
    """Errors from the client should propagate"""
    mock_client.chat.completions.create.side_effect = Exception("API failure")

    with pytest.raises(Exception, match="API failure"):
        GeneralLLM.generate_answer("Hello", [])