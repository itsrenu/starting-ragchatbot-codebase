import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest
from rag_system import RAGSystem

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MockConfig:
    """Mock configuration for testing"""

    def __init__(self):
        self.CHUNK_SIZE = 800
        self.CHUNK_OVERLAP = 100
        self.CHROMA_PATH = "./test_chroma_db"
        self.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        self.MAX_RESULTS = 5
        self.ANTHROPIC_API_KEY = "test_api_key"
        self.ANTHROPIC_MODEL = "claude-3-sonnet-20240229"
        self.MAX_HISTORY = 10


class TestRAGSystem:
    """Test suite for RAGSystem"""

    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    @patch("rag_system.CourseSearchTool")
    @patch("rag_system.CourseOutlineTool")
    @patch("rag_system.ToolManager")
    def setup_method(
        self,
        mock_tool_manager,
        mock_outline_tool,
        mock_search_tool,
        mock_session_manager,
        mock_ai_generator,
        mock_vector_store,
        mock_document_processor,
    ):
        """Setup test fixtures"""
        self.mock_config = MockConfig()

        # Setup mock instances
        self.mock_doc_processor = Mock()
        self.mock_vector_store_instance = Mock()
        self.mock_ai_generator_instance = Mock()
        self.mock_session_manager_instance = Mock()
        self.mock_search_tool_instance = Mock()
        self.mock_outline_tool_instance = Mock()
        self.mock_tool_manager_instance = Mock()

        # Configure mock classes to return instances
        mock_document_processor.return_value = self.mock_doc_processor
        mock_vector_store.return_value = self.mock_vector_store_instance
        mock_ai_generator.return_value = self.mock_ai_generator_instance
        mock_session_manager.return_value = self.mock_session_manager_instance
        mock_search_tool.return_value = self.mock_search_tool_instance
        mock_outline_tool.return_value = self.mock_outline_tool_instance
        mock_tool_manager.return_value = self.mock_tool_manager_instance

        # Create RAG system instance
        self.rag_system = RAGSystem(self.mock_config)

    def test_initialization(self):
        """Test RAGSystem initialization"""
        # Verify all components were initialized with correct parameters
        assert self.rag_system.config == self.mock_config
        assert self.rag_system.document_processor == self.mock_doc_processor
        assert self.rag_system.vector_store == self.mock_vector_store_instance
        assert self.rag_system.ai_generator == self.mock_ai_generator_instance
        assert self.rag_system.session_manager == self.mock_session_manager_instance

        # Verify tools were registered
        self.mock_tool_manager_instance.register_tool.assert_any_call(
            self.mock_search_tool_instance
        )
        self.mock_tool_manager_instance.register_tool.assert_any_call(
            self.mock_outline_tool_instance
        )

    def test_query_without_session(self):
        """Test query processing without session ID"""
        # Setup mocks
        self.mock_ai_generator_instance.generate_response.return_value = "AI response"
        self.mock_tool_manager_instance.get_tool_definitions.return_value = [
            {"name": "test_tool"}
        ]
        self.mock_tool_manager_instance.get_last_sources.return_value = []

        response, sources = self.rag_system.query("What is Python?")

        # Verify AI generator was called correctly
        self.mock_ai_generator_instance.generate_response.assert_called_once()
        call_args = self.mock_ai_generator_instance.generate_response.call_args

        assert (
            "Answer this question about course materials: What is Python?"
            in call_args[1]["query"]
        )
        assert call_args[1]["conversation_history"] is None
        assert call_args[1]["tools"] == [{"name": "test_tool"}]
        assert call_args[1]["tool_manager"] == self.mock_tool_manager_instance

        # Verify response
        assert response == "AI response"
        assert sources == []

        # Verify sources were reset
        self.mock_tool_manager_instance.reset_sources.assert_called_once()

    def test_query_with_session(self):
        """Test query processing with session ID"""
        session_id = "test_session_123"
        conversation_history = "User: Hello\\nAssistant: Hi!"

        # Setup mocks
        self.mock_session_manager_instance.get_conversation_history.return_value = (
            conversation_history
        )
        self.mock_ai_generator_instance.generate_response.return_value = (
            "AI response with history"
        )
        self.mock_tool_manager_instance.get_tool_definitions.return_value = []
        self.mock_tool_manager_instance.get_last_sources.return_value = [
            {"text": "Source 1", "link": "http://test1.com"}
        ]

        response, sources = self.rag_system.query(
            "Follow-up question", session_id=session_id
        )

        # Verify session history was retrieved
        self.mock_session_manager_instance.get_conversation_history.assert_called_once_with(
            session_id
        )

        # Verify AI generator received history
        call_args = self.mock_ai_generator_instance.generate_response.call_args[1]
        assert call_args["conversation_history"] == conversation_history

        # Verify session was updated
        self.mock_session_manager_instance.add_exchange.assert_called_once_with(
            session_id, "Follow-up question", "AI response with history"
        )

        # Verify response and sources
        assert response == "AI response with history"
        assert len(sources) == 1
        assert sources[0]["text"] == "Source 1"

    def test_query_with_ai_generator_error(self):
        """Test query handling when AI generator fails"""
        # Setup AI generator to raise exception
        self.mock_ai_generator_instance.generate_response.side_effect = Exception(
            "API key not found"
        )

        with pytest.raises(Exception) as exc_info:
            self.rag_system.query("Test query")

        assert "API key not found" in str(exc_info.value)

    def test_query_with_authentication_error(self):
        """Test query handling with authentication error"""
        # Setup AI generator to raise authentication exception
        auth_error = Exception(
            "Could not resolve authentication method. Expected either api_key or auth_token to be set"
        )
        self.mock_ai_generator_instance.generate_response.side_effect = auth_error

        with pytest.raises(Exception) as exc_info:
            self.rag_system.query("Test query")

        assert "authentication method" in str(exc_info.value)

    def test_get_course_analytics(self):
        """Test course analytics retrieval"""
        # Setup mock data
        self.mock_vector_store_instance.get_course_count.return_value = 5
        self.mock_vector_store_instance.get_existing_course_titles.return_value = [
            "Course 1",
            "Course 2",
            "Course 3",
            "Course 4",
            "Course 5",
        ]

        analytics = self.rag_system.get_course_analytics()

        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 5
        assert "Course 1" in analytics["course_titles"]

    def test_query_tool_integration(self):
        """Test query with tool execution flow"""
        # Setup tool manager to simulate tool execution
        self.mock_tool_manager_instance.get_tool_definitions.return_value = [
            {"name": "search_course_content", "description": "Search tool"}
        ]
        self.mock_tool_manager_instance.get_last_sources.return_value = [
            {"text": "Python Course - Lesson 1", "link": "http://course.com/lesson1"}
        ]

        # Setup AI generator to return response
        self.mock_ai_generator_instance.generate_response.return_value = (
            "Python is a programming language used for..."
        )

        response, sources = self.rag_system.query("What is Python?")

        # Verify tool definitions were passed to AI generator
        call_args = self.mock_ai_generator_instance.generate_response.call_args[1]
        assert call_args["tools"] == [
            {"name": "search_course_content", "description": "Search tool"}
        ]
        assert call_args["tool_manager"] == self.mock_tool_manager_instance

        # Verify sources were retrieved and reset
        self.mock_tool_manager_instance.get_last_sources.assert_called_once()
        self.mock_tool_manager_instance.reset_sources.assert_called_once()

        # Verify response includes tool results
        assert response == "Python is a programming language used for..."
        assert len(sources) == 1
        assert sources[0]["text"] == "Python Course - Lesson 1"


class TestRAGSystemIntegration:
    """Integration tests for RAGSystem with real components (but mocked external dependencies)"""

    @patch("rag_system.anthropic.Anthropic")
    def test_real_query_flow_with_mock_api(self, mock_anthropic_class):
        """Test actual query flow with mocked Anthropic API"""
        # This test would require setting up real vector store and other components
        # but with mocked external dependencies
        pass

    def test_error_propagation(self):
        """Test how errors propagate through the system"""
        # Test various error scenarios and ensure they're handled appropriately
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
