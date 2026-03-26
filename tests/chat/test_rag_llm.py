import pytest
from unittest.mock import MagicMock, patch

from src.rag_chatbot.rag.RAG_bot import RAGLLM, HISTORY_LEN


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_client():
    """Mock the Azure OpenAI client"""
    with patch("src.rag_chatbot.rag.RAG_bot.client") as mock:
        yield mock


@pytest.fixture
def mock_response():
    """Create a properly structured mock response"""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "This is an answer based on the context"
    return response


@pytest.fixture
def sample_context():
    """Sample retrieved documents"""
    return [
        {"content": "Q1 earnings were strong at $100M"},
        {"content": "Q2 performance showed growth of 15%"},
    ]


@pytest.fixture
def sample_history():
    """Sample chat history"""
    return [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First response"},
        {"role": "user", "content": "Second question"},
        {"role": "assistant", "content": "Second response"},
    ]


@pytest.fixture
def long_history():
    """History longer than HISTORY_LEN to test truncation"""
    return [
        {"role": "user", "content": f"Message {i}"}
        for i in range(HISTORY_LEN + 5)
    ]


@pytest.fixture
def large_context():
    """Large context with many documents"""
    return [
        {"content": f"Document {i}: Some content about topic {i % 3}"}
        for i in range(10)
    ]


# ============================================================================
# TESTS FOR BASIC FUNCTIONALITY
# ============================================================================

class TestRAGLLMBasicFunctionality:
    """Test basic functionality of RAGLLM.generate_answer"""
    
    def test_returns_string(self, mock_client, mock_response):
        """Test that generate_answer returns a string"""
        mock_client.chat.completions.create.return_value = mock_response
        
        result = RAGLLM.generate_answer("What were Q1 earnings?", [], [])
        
        assert isinstance(result, str)
        assert result == "This is an answer based on the context"
    
    def test_calls_client_once(self, mock_client, mock_response, sample_context):
        """Test that client.chat.completions.create is called exactly once"""
        mock_client.chat.completions.create.return_value = mock_response
        
        RAGLLM.generate_answer("Query", sample_context, [])
        
        mock_client.chat.completions.create.assert_called_once()
    
    
    def test_different_response_contents(self, mock_client, sample_context):
        """Test with various response contents"""
        test_responses = [
            "Short response",
            "A much longer response with multiple sentences. " * 10,
            "",  # Empty response
            "Response with special chars: !@#$%^&*()",
            "Multi-line\nresponse\nhere",
        ]
        
        for expected_response in test_responses:
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = expected_response
            mock_client.chat.completions.create.return_value = response
            
            result = RAGLLM.generate_answer("Query", sample_context, [])
            
            assert result == expected_response


# ============================================================================
# TESTS FOR CONTEXT HANDLING
# ============================================================================

class TestRAGLLMContextHandling:
    """Test context handling in generate_answer"""
    
    def test_context_included_in_messages(self, mock_client, mock_response, sample_context):
        """Test that context is included in messages"""
        mock_client.chat.completions.create.return_value = mock_response
        
        RAGLLM.generate_answer("Query", sample_context, [])
        
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        
        # Context should be in the last message (user message)
        user_message = messages[-1]["content"]
        assert sample_context[0]["content"] in user_message
        assert sample_context[1]["content"] in user_message
    
    
    def test_empty_context(self, mock_client, mock_response):
        """Test with empty context"""
        mock_client.chat.completions.create.return_value = mock_response
        
        result = RAGLLM.generate_answer("Query", [], [])
        
        assert isinstance(result, str)
    
    def test_single_context_document(self, mock_client, mock_response):
        """Test with single context document"""
        context = [{"content": "Only one document"}]
        mock_client.chat.completions.create.return_value = mock_response
        
        RAGLLM.generate_answer("Query", context, [])
        
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_message = messages[-1]["content"]
        
        assert "Only one document" in user_message
    
    def test_large_context(self, mock_client, mock_response, large_context):
        """Test with large context"""
        mock_client.chat.completions.create.return_value = mock_response
        
        RAGLLM.generate_answer("Query", large_context, [])
        
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_message = messages[-1]["content"]
        
        # All documents should be included
        for doc in large_context:
            assert doc["content"] in user_message


# ============================================================================
# TESTS FOR MESSAGE CONSTRUCTION
# ============================================================================

class TestRAGLLMMessageConstruction:
    """Test that messages are constructed correctly"""
    
    def test_system_message_included(self, mock_client, mock_response, sample_context):
        """Test that system message is included"""
        mock_client.chat.completions.create.return_value = mock_response
        
        RAGLLM.generate_answer("Query", sample_context, [])
        
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        
        assert messages[0]["role"] == "system"
    
# ============================================================================
# TESTS FOR HISTORY HANDLING
# ============================================================================

class TestRAGLLMHistoryHandling:
    """Test how history is handled"""
    
    def test_empty_history(self, mock_client, mock_response, sample_context):
        """Test with empty history"""
        mock_client.chat.completions.create.return_value = mock_response
        
        RAGLLM.generate_answer("Query", sample_context, [])
        
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        
        # Should have system message + user query only
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
    
    def test_history_included_in_messages(self, mock_client, mock_response, sample_context, sample_history):
        """Test that history is included in messages"""
        mock_client.chat.completions.create.return_value = mock_response
        
        RAGLLM.generate_answer("Query", sample_context, sample_history)
        
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        
        # Should have: system + history items + user query
        assert len(messages) == 1 + len(sample_history) + 1
    
    def test_history_order_preserved(self, mock_client, mock_response, sample_context):
        """Test that history order is preserved"""
        history = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Second"},
            {"role": "assistant", "content": "Response 2"},
        ]
        mock_client.chat.completions.create.return_value = mock_response
        
        RAGLLM.generate_answer("Query", sample_context, history)
        
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        
        # Skip system message (index 0)
        history_messages = messages[1:-1]  # Exclude last user query
        
        for i, hist_msg in enumerate(history_messages):
            assert hist_msg == history[i]
    
    def test_history_limited_to_history_len(self, mock_client, mock_response, sample_context, long_history):
        """Test that history is limited to HISTORY_LEN messages"""
        mock_client.chat.completions.create.return_value = mock_response
        
        RAGLLM.generate_answer("Query", sample_context, long_history)
        
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        
        # Should have: system + HISTORY_LEN messages + user query
        assert len(messages) == 1 + HISTORY_LEN + 1
    
    def test_history_truncation_uses_last_messages(self, mock_client, mock_response, sample_context):
        """Test that history truncation keeps the last messages"""
        history = [
            {"role": "user", "content": f"Old message {i}"}
            for i in range(HISTORY_LEN + 5)
        ]
        mock_client.chat.completions.create.return_value = mock_response
        
        RAGLLM.generate_answer("Query", sample_context, history)
        
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        
        # Extract history messages (skip system and final user query)
        history_messages = messages[1:-1]
        
        # Should be the last HISTORY_LEN messages
        expected_content = f"Old message {HISTORY_LEN - 1}"
        assert history_messages[0]["content"] == expected_content
    

# ============================================================================
# TESTS FOR CLIENT CALLS
# ============================================================================

class TestRAGLLMClientCall:
    """Test that client is called with correct parameters"""
    
    def test_correct_model_passed(self, mock_client, mock_response, sample_context):
        """Test that correct model is passed to client"""
        mock_client.chat.completions.create.return_value = mock_response
        
        RAGLLM.generate_answer("Query", sample_context, [])
        
        call_args = mock_client.chat.completions.create.call_args
        
        # Check that model and messages are passed
        assert "model" in call_args.kwargs
        assert "messages" in call_args.kwargs
    
    def test_messages_passed_as_list(self, mock_client, mock_response, sample_context):
        """Test that messages are passed as a list"""
        mock_client.chat.completions.create.return_value = mock_response
        
        RAGLLM.generate_answer("Query", sample_context, [])
        
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        
        assert isinstance(messages, list)
        assert len(messages) > 0


# ============================================================================
# TESTS FOR EDGE CASES
# ============================================================================

class TestRAGLLMEdgeCases:
    """Test edge cases and boundary conditions"""
    
    @pytest.mark.parametrize("query", [
        "",  # Empty query
        " ",  # Whitespace only
        "\n",  # Newline
        "\t",  # Tab
    ])
    def test_empty_or_whitespace_queries(self, mock_client, mock_response, sample_context, query):
        """Test with empty or whitespace queries"""
        mock_client.chat.completions.create.return_value = mock_response
        
        result = RAGLLM.generate_answer(query, sample_context, [])
        
        assert isinstance(result, str)
    
    @pytest.mark.parametrize("query", [
        "x" * 1000,  # Very long query
        "x" * 10000,  # Extremely long query
    ])
    def test_very_long_queries(self, mock_client, mock_response, sample_context, query):
        """Test with very long queries"""
        mock_client.chat.completions.create.return_value = mock_response
        
        result = RAGLLM.generate_answer(query, sample_context, [])
        
        assert isinstance(result, str)
    
    @pytest.mark.parametrize("query", [
        "What is 2+2?",
        "Create a test: @#$%^&*()",
        "Multi\nline\nquery",
        "Query with üñíçödé",
        '{"json": "like"}',
    ])
    def test_special_character_queries(self, mock_client, mock_response, sample_context, query):
        """Test with special characters in queries"""
        mock_client.chat.completions.create.return_value = mock_response
        
        result = RAGLLM.generate_answer(query, sample_context, [])
        
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        
        # Query should be included in messages
        user_message = messages[-1]["content"]
        assert query in user_message
    
    def test_history_at_boundary_history_len(self, mock_client, mock_response, sample_context):
        """Test with history exactly at HISTORY_LEN"""
        history = [{"role": "user", "content": f"Message {i}"} for i in range(HISTORY_LEN)]
        mock_client.chat.completions.create.return_value = mock_response
        
        RAGLLM.generate_answer("Query", sample_context, history)
        
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        
        # All history should be included
        assert len(messages) == 1 + HISTORY_LEN + 1
    
    def test_history_at_boundary_history_len_plus_one(self, mock_client, mock_response, sample_context):
        """Test with history at HISTORY_LEN + 1"""
        history = [{"role": "user", "content": f"Message {i}"} for i in range(HISTORY_LEN + 1)]
        mock_client.chat.completions.create.return_value = mock_response
        
        RAGLLM.generate_answer("Query", sample_context, history)
        
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        
        # Only last HISTORY_LEN should be included
        assert len(messages) == 1 + HISTORY_LEN + 1


# ============================================================================
# TESTS FOR ERROR HANDLING
# ============================================================================

class TestRAGLLMErrorHandling:
    """Test error handling"""
    
    def test_api_error_propagates(self, mock_client, sample_context):
        """Test that API errors are propagated"""
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        with pytest.raises(Exception, match="API Error"):
            RAGLLM.generate_answer("Query", sample_context, [])
    
    def test_attribute_error_on_response(self, mock_client, sample_context):
        """Test error handling if response structure is unexpected"""
        response = MagicMock()
        response.choices = []  # Empty choices
        mock_client.chat.completions.create.return_value = response
        
        with pytest.raises(IndexError):
            RAGLLM.generate_answer("Query", sample_context, [])
    
    def test_connection_error(self, mock_client, sample_context):
        """Test connection error handling"""
        mock_client.chat.completions.create.side_effect = ConnectionError("Network error")
        
        with pytest.raises(ConnectionError):
            RAGLLM.generate_answer("Query", sample_context, [])


# ============================================================================
# TESTS FOR INPUT TYPES
# ============================================================================

class TestRAGLLMInputTypes:
    """Test with various input types"""
    
    def test_various_query_types(self, mock_client, mock_response, sample_context):
        """Test with different string types"""
        queries = [
            "Simple query",
            'Double "quoted" string',
            "Single 'quoted' string",
            "String with\ttabs",
            "String with\nnewlines",
        ]
        
        mock_client.chat.completions.create.return_value = mock_response
        
        for query in queries:
            result = RAGLLM.generate_answer(query, sample_context, [])
            assert isinstance(result, str)
    
    def test_history_with_different_roles(self, mock_client, mock_response, sample_context):
        """Test history with various role types"""
        history = [
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "Assistant message"},
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "Another user message"},
        ]
        
        mock_client.chat.completions.create.return_value = mock_response
        
        result = RAGLLM.generate_answer("Query", sample_context, history)
        
        assert isinstance(result, str)
        
        # Verify all history items are included
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        
        history_messages = messages[1:-1]  # Exclude system and final query
        assert len(history_messages) == len(history)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestRAGLLMIntegration:
    """Integration-style tests"""
    
    def test_realistic_rag_flow(self, mock_client, mock_response):
        """Test realistic RAG flow"""
        mock_client.chat.completions.create.return_value = mock_response
        
        context = [
            {"content": "Amazon Q1 2024: Revenue $139.2B, Net income $10.6B"},
            {"content": "Key drivers: AWS growth of 17%, advertising revenue up 25%"},
        ]
        
        history = [
            {"role": "user", "content": "Tell me about Amazon's recent earnings"},
            {"role": "assistant", "content": "I'd be happy to share information about Amazon's earnings..."},
        ]
        
        result = RAGLLM.generate_answer("What was the Q1 revenue?", context, history)
        
        assert isinstance(result, str)
        
        # Verify context and history were used
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        
        user_message = messages[-1]["content"]
        assert "139.2B" in user_message  # From context
        assert "Q1 revenue" in user_message  # From query
    
    def test_full_context_and_history(self, mock_client, mock_response):
        """Test with complete realistic context and history"""
        mock_client.chat.completions.create.return_value = mock_response
        
        context = [
            {"content": f"Document {i}: Information about quarterly results"}
            for i in range(5)
        ]
        
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ] * (HISTORY_LEN // 2)  # Fill up to close to HISTORY_LEN
        
        result = RAGLLM.generate_answer("New question", context, history)
        
        assert isinstance(result, str)
        
        # Verify structure
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert len(messages) <= 1 + HISTORY_LEN + 1