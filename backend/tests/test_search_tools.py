import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCourseSearchTool:
    """Test suite for CourseSearchTool"""

    def setup_method(self):
        """Setup test fixtures"""
        self.mock_vector_store = Mock()
        self.search_tool = CourseSearchTool(self.mock_vector_store)

    def test_get_tool_definition(self):
        """Test tool definition structure"""
        definition = self.search_tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["query"]

    def test_execute_with_successful_results(self):
        """Test execute method with successful search results"""
        # Mock successful search results
        mock_results = SearchResults(
            documents=["Test content from lesson 1", "More test content"],
            metadata=[
                {
                    "course_title": "Test Course",
                    "lesson_number": 1,
                    "lesson_link": "http://test.com/lesson1",
                },
                {
                    "course_title": "Test Course",
                    "lesson_number": 2,
                    "lesson_link": "http://test.com/lesson2",
                },
            ],
            distances=[0.1, 0.2],
            error=None,
        )

        self.mock_vector_store.search.return_value = mock_results

        result = self.search_tool.execute("test query")

        # Verify search was called correctly
        self.mock_vector_store.search.assert_called_once_with(
            query="test query", course_name=None, lesson_number=None
        )

        # Verify formatted results
        assert "Test Course - Lesson 1" in result
        assert "Test content from lesson 1" in result
        assert "Test Course - Lesson 2" in result
        assert "More test content" in result

        # Verify sources were stored
        assert len(self.search_tool.last_sources) == 2
        assert self.search_tool.last_sources[0]["text"] == "Test Course - Lesson 1"
        assert self.search_tool.last_sources[0]["link"] == "http://test.com/lesson1"

    def test_execute_with_course_filter(self):
        """Test execute method with course name filter"""
        mock_results = SearchResults(
            documents=["Filtered content"],
            metadata=[{"course_title": "Specific Course", "lesson_number": 1}],
            distances=[0.1],
            error=None,
        )

        self.mock_vector_store.search.return_value = mock_results

        result = self.search_tool.execute("test query", course_name="Specific Course")

        self.mock_vector_store.search.assert_called_once_with(
            query="test query", course_name="Specific Course", lesson_number=None
        )

        assert "Specific Course" in result

    def test_execute_with_lesson_filter(self):
        """Test execute method with lesson number filter"""
        mock_results = SearchResults(
            documents=["Lesson 3 content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 3}],
            distances=[0.1],
            error=None,
        )

        self.mock_vector_store.search.return_value = mock_results

        result = self.search_tool.execute("test query", lesson_number=3)

        self.mock_vector_store.search.assert_called_once_with(
            query="test query", course_name=None, lesson_number=3
        )

        assert "Lesson 3" in result

    def test_execute_with_error(self):
        """Test execute method when vector store returns error"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Search failed due to connection error",
        )

        self.mock_vector_store.search.return_value = mock_results

        result = self.search_tool.execute("test query")

        assert result == "Search failed due to connection error"
        assert len(self.search_tool.last_sources) == 0

    def test_execute_with_no_results(self):
        """Test execute method when no results found"""
        mock_results = SearchResults(
            documents=[], metadata=[], distances=[], error=None
        )

        self.mock_vector_store.search.return_value = mock_results

        result = self.search_tool.execute("nonexistent query")

        assert "No relevant content found" in result
        assert len(self.search_tool.last_sources) == 0

    def test_execute_with_no_results_with_filters(self):
        """Test execute method when no results found with filters"""
        mock_results = SearchResults(
            documents=[], metadata=[], distances=[], error=None
        )

        self.mock_vector_store.search.return_value = mock_results

        result = self.search_tool.execute(
            "nonexistent query", course_name="Nonexistent Course", lesson_number=99
        )

        assert (
            "No relevant content found in course 'Nonexistent Course' in lesson 99"
            in result
        )


class TestToolManager:
    """Test suite for ToolManager"""

    def setup_method(self):
        """Setup test fixtures"""
        self.tool_manager = ToolManager()
        self.mock_tool = Mock()
        self.mock_tool.get_tool_definition.return_value = {
            "name": "test_tool",
            "description": "Test tool",
        }
        self.mock_tool.execute.return_value = "Tool executed successfully"

    def test_register_tool(self):
        """Test tool registration"""
        self.tool_manager.register_tool(self.mock_tool)

        assert "test_tool" in self.tool_manager.tools
        assert self.tool_manager.tools["test_tool"] == self.mock_tool

    def test_get_tool_definitions(self):
        """Test getting tool definitions"""
        self.tool_manager.register_tool(self.mock_tool)

        definitions = self.tool_manager.get_tool_definitions()

        assert len(definitions) == 1
        assert definitions[0]["name"] == "test_tool"

    def test_execute_tool(self):
        """Test tool execution"""
        self.tool_manager.register_tool(self.mock_tool)

        result = self.tool_manager.execute_tool("test_tool", param1="value1")

        self.mock_tool.execute.assert_called_once_with(param1="value1")
        assert result == "Tool executed successfully"

    def test_execute_nonexistent_tool(self):
        """Test execution of nonexistent tool"""
        result = self.tool_manager.execute_tool("nonexistent_tool")

        assert "Tool 'nonexistent_tool' not found" in result

    def test_get_last_sources(self):
        """Test getting sources from tools"""
        mock_search_tool = Mock()
        mock_search_tool.get_tool_definition.return_value = {"name": "search_tool"}
        mock_search_tool.last_sources = [
            {"text": "Test source", "link": "http://test.com"}
        ]

        self.tool_manager.register_tool(mock_search_tool)

        sources = self.tool_manager.get_last_sources()

        assert len(sources) == 1
        assert sources[0]["text"] == "Test source"

    def test_reset_sources(self):
        """Test resetting sources from tools"""
        mock_search_tool = Mock()
        mock_search_tool.get_tool_definition.return_value = {"name": "search_tool"}
        mock_search_tool.last_sources = [{"text": "Test source"}]

        self.tool_manager.register_tool(mock_search_tool)

        self.tool_manager.reset_sources()

        assert mock_search_tool.last_sources == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
