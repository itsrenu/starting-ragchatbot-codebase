import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator

class MockAnthropicResponse:
    """Mock Anthropic API response"""
    def __init__(self, content_text=None, stop_reason="end_turn", tool_use_content=None):
        self.stop_reason = stop_reason
        if tool_use_content:
            self.content = tool_use_content
        else:
            mock_content = Mock()
            mock_content.text = content_text or "Default response"
            self.content = [mock_content]

class MockToolUseContent:
    """Mock tool use content block"""
    def __init__(self, tool_name, tool_input, tool_id="test_id"):
        self.type = "tool_use"
        self.name = tool_name
        self.input = tool_input
        self.id = tool_id

class TestAIGenerator:
    """Test suite for AIGenerator"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.api_key = "test_api_key"
        self.model = "claude-3-sonnet-20240229"
        self.ai_generator = AIGenerator(self.api_key, self.model)
    
    def test_init(self):
        """Test AIGenerator initialization"""
        assert self.ai_generator.model == self.model
        assert self.ai_generator.base_params["model"] == self.model
        assert self.ai_generator.base_params["temperature"] == 0
        assert self.ai_generator.base_params["max_tokens"] == 800
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_without_tools(self, mock_anthropic_class):
        """Test response generation without tools"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        mock_response = MockAnthropicResponse("Simple response without tools")
        mock_client.messages.create.return_value = mock_response
        
        # Create fresh instance with mocked client
        ai_gen = AIGenerator(self.api_key, self.model)
        
        result = ai_gen.generate_response("What is Python?")
        
        # Verify API call
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args[1]
        
        assert call_args["model"] == self.model
        assert call_args["messages"][0]["role"] == "user"
        assert call_args["messages"][0]["content"] == "What is Python?"
        assert "tools" not in call_args
        
        assert result == "Simple response without tools"
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_conversation_history(self, mock_anthropic_class):
        """Test response generation with conversation history"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        mock_response = MockAnthropicResponse("Response with history")
        mock_client.messages.create.return_value = mock_response
        
        ai_gen = AIGenerator(self.api_key, self.model)
        
        history = "User: Hello\\nAssistant: Hi there!"
        result = ai_gen.generate_response("How are you?", conversation_history=history)
        
        # Verify system prompt includes history
        call_args = mock_client.messages.create.call_args[1]
        assert "Previous conversation:" in call_args["system"]
        assert history in call_args["system"]
        
        assert result == "Response with history"
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_tools_no_tool_use(self, mock_anthropic_class):
        """Test response generation with tools available but not used"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        mock_response = MockAnthropicResponse("Response without using tools", stop_reason="end_turn")
        mock_client.messages.create.return_value = mock_response
        
        ai_gen = AIGenerator(self.api_key, self.model)
        
        mock_tools = [{"name": "search_course_content", "description": "Search tool"}]
        mock_tool_manager = Mock()
        
        result = ai_gen.generate_response(
            "What is AI?", 
            tools=mock_tools, 
            tool_manager=mock_tool_manager
        )
        
        # Verify tools were passed to API
        call_args = mock_client.messages.create.call_args[1]
        assert call_args["tools"] == mock_tools
        assert call_args["tool_choice"]["type"] == "auto"
        
        # Verify tool manager wasn't called
        mock_tool_manager.execute_tool.assert_not_called()
        
        assert result == "Response without using tools"
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_tool_use(self, mock_anthropic_class):
        """Test response generation with tool use"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock initial response with tool use
        tool_content = MockToolUseContent("search_course_content", {"query": "Python basics"})
        initial_response = MockAnthropicResponse(
            stop_reason="tool_use",
            tool_use_content=[tool_content]
        )
        
        # Mock final response after tool execution
        final_response = MockAnthropicResponse("Based on the search, Python is a programming language")
        
        mock_client.messages.create.side_effect = [initial_response, final_response]
        
        ai_gen = AIGenerator(self.api_key, self.model)
        
        mock_tools = [{"name": "search_course_content", "description": "Search tool"}]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results about Python"
        
        result = ai_gen.generate_response(
            "What is Python?", 
            tools=mock_tools, 
            tool_manager=mock_tool_manager
        )
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", 
            query="Python basics"
        )
        
        # Verify two API calls were made
        assert mock_client.messages.create.call_count == 2
        
        # Verify final response
        assert result == "Based on the search, Python is a programming language"
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_api_key_error(self, mock_anthropic_class):
        """Test API key authentication error"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock authentication error
        mock_client.messages.create.side_effect = Exception(
            "Could not resolve authentication method. Expected either api_key or auth_token to be set"
        )
        
        ai_gen = AIGenerator("invalid_key", self.model)
        
        with pytest.raises(Exception) as exc_info:
            ai_gen.generate_response("Test query")
        
        assert "authentication method" in str(exc_info.value)
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_network_error(self, mock_anthropic_class):
        """Test network/connection error"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock network error
        mock_client.messages.create.side_effect = Exception("Connection timeout")
        
        ai_gen = AIGenerator(self.api_key, self.model)
        
        with pytest.raises(Exception) as exc_info:
            ai_gen.generate_response("Test query")
        
        assert "Connection timeout" in str(exc_info.value)
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_execution_with_multiple_tools(self, mock_anthropic_class):
        """Test handling multiple tool calls in one response"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock response with multiple tool uses
        tool_content_1 = MockToolUseContent("search_course_content", {"query": "Python"}, "id1")
        tool_content_2 = MockToolUseContent("get_course_outline", {"course_name": "Python Course"}, "id2")
        
        initial_response = MockAnthropicResponse(
            stop_reason="tool_use",
            tool_use_content=[tool_content_1, tool_content_2]
        )
        
        final_response = MockAnthropicResponse("Combined results from both tools")
        mock_client.messages.create.side_effect = [initial_response, final_response]
        
        ai_gen = AIGenerator(self.api_key, self.model)
        
        mock_tools = [
            {"name": "search_course_content", "description": "Search tool"},
            {"name": "get_course_outline", "description": "Outline tool"}
        ]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Search results", "Outline results"]
        
        result = ai_gen.generate_response(
            "Tell me about Python course",
            tools=mock_tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="Python")
        mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="Python Course")
        
        assert result == "Combined results from both tools"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])