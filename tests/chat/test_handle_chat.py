import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from src.rag_chatbot.rag.RAG_bot import handle_chat


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_history():
    """Sample chat history"""
    return [
        {"role": "user", "content": "Hello"},
        {"role": "system", "content": "Hi there!"},
    ]


@pytest.fixture
def sample_context():
    """Sample retrieved documents"""
    return [
        {"content": "Q1 earnings were strong at $100M"},
        {"content": "Q2 performance showed growth of 15%"},
    ]


@pytest.fixture
def mock_decide_route():
    """Mock decide_route function"""
    with patch("src.rag_chatbot.rag.RAG_bot.decide_route") as mock:
        yield mock


@pytest.fixture
def mock_retrieve_context():
    """Mock retrieve_context function"""
    with patch("src.rag_chatbot.rag.RAG_bot.retrieve_context") as mock:
        yield mock


@pytest.fixture
def mock_general_llm():
    """Mock GeneralLLM class"""
    with patch("src.rag_chatbot.rag.RAG_bot.GeneralLLM") as mock:
        yield mock


@pytest.fixture
def mock_rag_llm():
    """Mock RAGLLM class"""
    with patch("src.rag_chatbot.rag.RAG_bot.RAGLLM") as mock:
        yield mock


@pytest.fixture
def mock_mcp_llm():
    """Mock MCPLLM class"""
    with patch("src.rag_chatbot.rag.RAG_bot.MCPLLM") as mock:
        yield mock


@pytest.fixture
def mock_build_grounded_task():
    """Mock build_grounded_task function"""
    with patch("src.rag_chatbot.rag.RAG_bot.build_grounded_task") as mock:
        yield mock


# ============================================================================
# TESTS FOR GENERAL ROUTE
# ============================================================================

class TestHandleChatGeneralRoute:
    """Test handle_chat when route is 'general'"""
    
    @pytest.mark.asyncio
    async def test_general_route_returns_correct_structure(
        self, mock_decide_route, mock_general_llm
    ):
        """Test that general route returns correct response structure"""
        mock_decide_route.return_value = "general"
        mock_general_llm.generate_answer.return_value = "Here's a general answer"
        
        result = await handle_chat("Hello", [])
        
        assert result["mode"] == "general"
        assert result["answer"] == "Here's a general answer"
        assert len(result) == 2  # Only mode and answer
    
    @pytest.mark.asyncio
    async def test_general_route_passes_query_and_history(
        self, mock_decide_route, mock_general_llm
    ):
        """Test that query and history are passed to GeneralLLM"""
        mock_decide_route.return_value = "general"
        mock_general_llm.generate_answer.return_value = "Answer"
        history = [{"role": "user", "content": "Previous"}]
        
        await handle_chat("New query", history)
        
        mock_general_llm.generate_answer.assert_called_once_with(
            user_query="New query",
            history=history
        )
    
    @pytest.mark.asyncio
    async def test_general_route_with_empty_history(
        self, mock_decide_route, mock_general_llm
    ):
        """Test general route with empty history"""
        mock_decide_route.return_value = "general"
        mock_general_llm.generate_answer.return_value = "Answer"
        
        result = await handle_chat("Query", [])
        
        assert result["answer"] == "Answer"
        mock_general_llm.generate_answer.assert_called_once()


# ============================================================================
# TESTS FOR RAG ROUTE
# ============================================================================

class TestHandleChatRAGRoute:
    """Test handle_chat when route is 'rag'"""
    
    @pytest.mark.asyncio
    async def test_rag_route_with_context(
        self, mock_decide_route, mock_retrieve_context, mock_rag_llm
    ):
        """Test RAG route when context is found"""
        mock_decide_route.return_value = "rag"
        context = [{"content": "Document content"}]
        mock_retrieve_context.return_value = context
        mock_rag_llm.generate_answer.return_value = "Answer from RAG"
        
        result = await handle_chat("Find information", [])
        
        assert result["mode"] == "rag"
        assert result["answer"] == "Answer from RAG"
        assert result["retrieved"] == context
    
    @pytest.mark.asyncio
    async def test_rag_route_without_context(
        self, mock_decide_route, mock_retrieve_context
    ):
        """Test RAG route when no context is retrieved"""
        mock_decide_route.return_value = "rag"
        mock_retrieve_context.return_value = []
        
        result = await handle_chat("Find nonexistent information", [])
        
        assert result["mode"] == "rag"
        assert "couldn't find" in result["answer"].lower()
        assert result["retrieved"] == []
    
    @pytest.mark.asyncio
    async def test_rag_route_passes_context_to_llm(
        self, mock_decide_route, mock_retrieve_context, mock_rag_llm, sample_history
    ):
        """Test that context is passed correctly to RAGLLM"""
        mock_decide_route.return_value = "rag"
        context = [{"content": "Doc 1"}, {"content": "Doc 2"}]
        mock_retrieve_context.return_value = context
        mock_rag_llm.generate_answer.return_value = "Answer"
        
        await handle_chat("Query", sample_history)
        
        mock_rag_llm.generate_answer.assert_called_once_with(
            user_query="Query",
            context=context,
            history=sample_history
        )
    
    @pytest.mark.asyncio
    async def test_rag_route_calls_retrieve_context(
        self, mock_decide_route, mock_retrieve_context, mock_rag_llm
    ):
        """Test that retrieve_context is called with the user query"""
        mock_decide_route.return_value = "rag"
        mock_retrieve_context.return_value = [{"content": "Doc"}]
        mock_rag_llm.generate_answer.return_value = "Answer"
        
        await handle_chat("Specific query", [])
        
        mock_retrieve_context.assert_called_once_with("Specific query")


# ============================================================================
# TESTS FOR MCP ROUTE
# ============================================================================

class TestHandleChatMCPRoute:
    """Test handle_chat when route is 'mcp'"""
    
    @pytest.mark.asyncio
    async def test_mcp_route_returns_correct_structure(
        self, mock_decide_route, mock_mcp_llm
    ):
        """Test that MCP route returns correct response structure"""
        mock_decide_route.return_value = "mcp"
        mcp_instance = AsyncMock()
        mcp_instance.generate_answer.return_value = "Task completed"
        mock_mcp_llm.return_value = mcp_instance
        
        result = await handle_chat("Create ticket", [])
        
        assert result["mode"] == "mcp"
        assert result["answer"] == "Task completed"
        assert len(result) == 2
    
    @pytest.mark.asyncio
    async def test_mcp_route_creates_mcp_instance(self, mock_decide_route, mock_mcp_llm):
        """Test that MCPLLM instance is created"""
        mock_decide_route.return_value = "mcp"
        mcp_instance = AsyncMock()
        mcp_instance.generate_answer.return_value = "Result"
        mock_mcp_llm.return_value = mcp_instance
        
        await handle_chat("Create ticket", [])
        
        mock_mcp_llm.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mcp_route_calls_cleanup(self, mock_decide_route, mock_mcp_llm):
        """Test that cleanup is always called"""
        mock_decide_route.return_value = "mcp"
        mcp_instance = AsyncMock()
        mcp_instance.generate_answer.return_value = "Result"
        mock_mcp_llm.return_value = mcp_instance
        
        await handle_chat("Create ticket", [])
        
        mcp_instance.cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mcp_route_passes_query_and_history(
        self, mock_decide_route, mock_mcp_llm, sample_history
    ):
        """Test that query and history are passed to MCP generate_answer"""
        mock_decide_route.return_value = "mcp"
        mcp_instance = AsyncMock()
        mcp_instance.generate_answer.return_value = "Result"
        mock_mcp_llm.return_value = mcp_instance
        
        await handle_chat("Create Jira ticket", sample_history)
        
        mcp_instance.generate_answer.assert_called_once_with(
            user_query="Create Jira ticket",
            history=sample_history
        )
    
    @pytest.mark.asyncio
    async def test_mcp_route_cleanup_on_error(self, mock_decide_route, mock_mcp_llm):
        """Test that cleanup is called even if generate_answer raises error"""
        mock_decide_route.return_value = "mcp"
        mcp_instance = AsyncMock()
        mcp_instance.generate_answer.side_effect = Exception("MCP failed")
        mock_mcp_llm.return_value = mcp_instance
        
        with pytest.raises(Exception, match="MCP failed"):
            await handle_chat("Create ticket", [])
        
        # Cleanup should still be called
        mcp_instance.cleanup.assert_called_once()


# ============================================================================
# TESTS FOR RAG_THEN_MCP ROUTE
# ============================================================================

class TestHandleChatRAGThenMCPRoute:
    """Test handle_chat when route is 'rag_then_mcp'"""
    
    @pytest.mark.asyncio
    async def test_rag_then_mcp_route_returns_correct_structure(
        self, mock_decide_route, mock_retrieve_context, 
        mock_build_grounded_task, mock_mcp_llm
    ):
        """Test that rag_then_mcp route returns correct response structure"""
        mock_decide_route.return_value = "rag_then_mcp"
        context = [{"content": "Meeting notes"}]
        mock_retrieve_context.return_value = context
        mock_build_grounded_task.return_value = "Grounded task string"
        
        mcp_instance = AsyncMock()
        mcp_instance.generate_answer.return_value = "Tickets created"
        mock_mcp_llm.return_value = mcp_instance
        
        result = await handle_chat("Create tickets from meeting", [])
        
        assert result["mode"] == "rag_then_mcp"
        assert result["answer"] == "Tickets created"
        assert result["retrieved"] == context
        assert result["grounded_task"] == "Grounded task string"
    
    @pytest.mark.asyncio
    async def test_rag_then_mcp_route_retrieves_context(
        self, mock_decide_route, mock_retrieve_context,
        mock_build_grounded_task, mock_mcp_llm
    ):
        """Test that retrieve_context is called"""
        mock_decide_route.return_value = "rag_then_mcp"
        context = [{"content": "Doc"}]
        mock_retrieve_context.return_value = context
        mock_build_grounded_task.return_value = "Task"
        
        mcp_instance = AsyncMock()
        mcp_instance.generate_answer.return_value = "Result"
        mock_mcp_llm.return_value = mcp_instance
        
        await handle_chat("Create from meeting", [])
        
        mock_retrieve_context.assert_called_once_with("Create from meeting")
    
    @pytest.mark.asyncio
    async def test_rag_then_mcp_route_builds_grounded_task(
        self, mock_decide_route, mock_retrieve_context,
        mock_build_grounded_task, mock_mcp_llm
    ):
        """Test that build_grounded_task is called with query and context"""
        mock_decide_route.return_value = "rag_then_mcp"
        context = [{"content": "Doc"}]
        mock_retrieve_context.return_value = context
        mock_build_grounded_task.return_value = "Task"
        
        mcp_instance = AsyncMock()
        mcp_instance.generate_answer.return_value = "Result"
        mock_mcp_llm.return_value = mcp_instance
        
        await handle_chat("Query", [])
        
        mock_build_grounded_task.assert_called_once_with("Query", context)
    
    @pytest.mark.asyncio
    async def test_rag_then_mcp_route_passes_grounded_task_to_mcp(
        self, mock_decide_route, mock_retrieve_context,
        mock_build_grounded_task, mock_mcp_llm, sample_history
    ):
        """Test that grounded task is passed to MCP generate_answer"""
        mock_decide_route.return_value = "rag_then_mcp"
        context = [{"content": "Doc"}]
        mock_retrieve_context.return_value = context
        mock_build_grounded_task.return_value = "Grounded task instruction"
        
        mcp_instance = AsyncMock()
        mcp_instance.generate_answer.return_value = "Result"
        mock_mcp_llm.return_value = mcp_instance
        
        await handle_chat("Query", sample_history)
        
        mcp_instance.generate_answer.assert_called_once_with(
            user_query="Grounded task instruction",
            history=sample_history
        )
    
    @pytest.mark.asyncio
    async def test_rag_then_mcp_route_converts_dict_to_json(
        self, mock_decide_route, mock_retrieve_context,
        mock_build_grounded_task, mock_mcp_llm
    ):
        """Test that non-string grounded task is converted to JSON"""
        mock_decide_route.return_value = "rag_then_mcp"
        context = [{"content": "Doc"}]
        mock_retrieve_context.return_value = context
        # Return a dict instead of string
        mock_build_grounded_task.return_value = {"action": "create", "type": "ticket"}
        
        mcp_instance = AsyncMock()
        mcp_instance.generate_answer.return_value = "Result"
        mock_mcp_llm.return_value = mcp_instance
        
        await handle_chat("Query", [])
        
        # Should convert dict to JSON string
        call_args = mcp_instance.generate_answer.call_args
        user_query = call_args.kwargs["user_query"]
        assert isinstance(user_query, str)
        assert "action" in user_query
    
    @pytest.mark.asyncio
    async def test_rag_then_mcp_route_cleanup_on_error(
        self, mock_decide_route, mock_retrieve_context,
        mock_build_grounded_task, mock_mcp_llm
    ):
        """Test that cleanup is called even if error occurs"""
        mock_decide_route.return_value = "rag_then_mcp"
        context = [{"content": "Doc"}]
        mock_retrieve_context.return_value = context
        mock_build_grounded_task.return_value = "Task"
        
        mcp_instance = AsyncMock()
        mcp_instance.generate_answer.side_effect = Exception("MCP error")
        mock_mcp_llm.return_value = mcp_instance
        
        with pytest.raises(Exception, match="MCP error"):
            await handle_chat("Query", [])
        
        # Cleanup should still be called
        mcp_instance.cleanup.assert_called_once()


# ============================================================================
# TESTS FOR DEFAULT/UNKNOWN ROUTE
# ============================================================================

class TestHandleChatDefaultRoute:
    """Test handle_chat when route is unknown/default"""
    
    @pytest.mark.asyncio
    async def test_unknown_route_defaults_to_general(
        self, mock_decide_route, mock_general_llm
    ):
        """Test that unknown routes default to general"""
        mock_decide_route.return_value = "unknown_route"
        mock_general_llm.generate_answer.return_value = "Default answer"
        
        result = await handle_chat("Query", [])
        
        assert result["mode"] == "general"
        assert result["answer"] == "Default answer"
        mock_general_llm.generate_answer.assert_called_once()


# ============================================================================
# TESTS FOR MODE PARAMETER
# ============================================================================

class TestHandleChatModeParameter:
    """Test handle_chat mode parameter handling"""
    
    @pytest.mark.asyncio
    async def test_mode_parameter_passed_to_decide_route(
        self, mock_decide_route, mock_general_llm
    ):
        """Test that mode parameter is passed to decide_route"""
        mock_decide_route.return_value = "general"
        mock_general_llm.generate_answer.return_value = "Answer"
        
        await handle_chat("Query", [], mode="rag")
        
        mock_decide_route.assert_called_once_with("Query", "rag")
    
    @pytest.mark.asyncio
    async def test_default_mode_is_auto(
        self, mock_decide_route, mock_general_llm
    ):
        """Test that default mode is 'auto'"""
        mock_decide_route.return_value = "general"
        mock_general_llm.generate_answer.return_value = "Answer"
        
        await handle_chat("Query", [])
        
        # Second argument to decide_route should be "auto"
        assert mock_decide_route.call_args[0][1] == "auto"


# ============================================================================
# TESTS FOR HISTORY PARAMETER
# ============================================================================

class TestHandleChatHistoryParameter:
    """Test handle_chat history parameter handling"""
    
    @pytest.mark.asyncio
    async def test_empty_history(self, mock_decide_route, mock_general_llm):
        """Test with empty history"""
        mock_decide_route.return_value = "general"
        mock_general_llm.generate_answer.return_value = "Answer"
        
        await handle_chat("Query", [])
        
        call_args = mock_general_llm.generate_answer.call_args
        assert call_args.kwargs["history"] == []
    
    @pytest.mark.asyncio
    async def test_with_history(self, mock_decide_route, mock_general_llm, sample_history):
        """Test with existing history"""
        mock_decide_route.return_value = "general"
        mock_general_llm.generate_answer.return_value = "Answer"
        
        await handle_chat("Query", sample_history)
        
        call_args = mock_general_llm.generate_answer.call_args
        assert call_args.kwargs["history"] == sample_history
    
    @pytest.mark.asyncio
    async def test_large_history(self, mock_decide_route, mock_general_llm):
        """Test with large history"""
        mock_decide_route.return_value = "general"
        mock_general_llm.generate_answer.return_value = "Answer"
        large_history = [{"role": "user", "content": f"Msg {i}"} for i in range(100)]
        
        await handle_chat("Query", large_history)
        
        call_args = mock_general_llm.generate_answer.call_args
        assert call_args.kwargs["history"] == large_history


# ============================================================================
# TESTS FOR RESPONSE STRUCTURE
# ============================================================================

class TestHandleChatResponseStructure:
    """Test that handle_chat always returns proper response structure"""
    
    @pytest.mark.asyncio
    async def test_response_always_has_mode(
        self, mock_decide_route, mock_general_llm
    ):
        """Test that response always includes 'mode' key"""
        mock_decide_route.return_value = "general"
        mock_general_llm.generate_answer.return_value = "Answer"
        
        result = await handle_chat("Query", [])
        
        assert "mode" in result
        assert result["mode"] in ["general", "rag", "mcp", "rag_then_mcp"]
    
    @pytest.mark.asyncio
    async def test_response_always_has_answer(
        self, mock_decide_route, mock_general_llm
    ):
        """Test that response always includes 'answer' key"""
        mock_decide_route.return_value = "general"
        mock_general_llm.generate_answer.return_value = "Some answer"
        
        result = await handle_chat("Query", [])
        
        assert "answer" in result
        assert isinstance(result["answer"], str)


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestHandleChatEdgeCases:
    """Test edge cases and error handling"""
    
    @pytest.mark.asyncio
    async def test_empty_query(self, mock_decide_route, mock_general_llm):
        """Test with empty query string"""
        mock_decide_route.return_value = "general"
        mock_general_llm.generate_answer.return_value = "Response"
        
        result = await handle_chat("", [])
        
        assert result["answer"] == "Response"
        mock_decide_route.assert_called_once_with("", "auto")
    
    @pytest.mark.asyncio
    async def test_very_long_query(self, mock_decide_route, mock_general_llm):
        """Test with very long query"""
        long_query = "x" * 10000
        mock_decide_route.return_value = "general"
        mock_general_llm.generate_answer.return_value = "Response"
        
        result = await handle_chat(long_query, [])
        
        assert result["answer"] == "Response"
        mock_decide_route.assert_called_once_with(long_query, "auto")
    
    @pytest.mark.asyncio
    async def test_query_with_special_characters(
        self, mock_decide_route, mock_general_llm
    ):
        """Test with special characters in query"""
        special_query = "What is 2+2? $$$!@#%"
        mock_decide_route.return_value = "general"
        mock_general_llm.generate_answer.return_value = "Response"
        
        result = await handle_chat(special_query, [])
        
        assert result["answer"] == "Response"
        mock_decide_route.assert_called_once_with(special_query, "auto")
    
    @pytest.mark.asyncio
    async def test_multiline_query(self, mock_decide_route, mock_general_llm):
        """Test with multiline query"""
        multiline_query = "Line 1\nLine 2\nLine 3"
        mock_decide_route.return_value = "general"
        mock_general_llm.generate_answer.return_value = "Response"
        
        result = await handle_chat(multiline_query, [])
        
        assert result["answer"] == "Response"
