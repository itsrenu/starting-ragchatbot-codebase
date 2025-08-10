#!/usr/bin/env python3
"""
CourseSearchTool Tests with Real Data

These tests specifically verify the CourseSearchTool execute method
works correctly with the actual course data in the system.
"""

import os
import sys

import pytest
from config import config
from search_tools import CourseSearchTool, ToolManager
from vector_store import VectorStore

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCourseSearchToolRealData:
    """Test CourseSearchTool with actual course data"""

    def setup_method(self):
        """Setup with real vector store"""
        self.vector_store = VectorStore(
            chroma_path=config.CHROMA_PATH,
            embedding_model=config.EMBEDDING_MODEL,
            max_results=config.MAX_RESULTS,
        )
        self.search_tool = CourseSearchTool(self.vector_store)

    def test_course_search_tool_definition(self):
        """Test that tool definition is correct"""
        definition = self.search_tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["query"]

        print("✅ CourseSearchTool definition is correct")

    def test_execute_with_existing_courses(self):
        """Test execute method with queries that should find existing course content"""

        # Check what courses exist
        course_count = self.vector_store.get_course_count()
        course_titles = self.vector_store.get_existing_course_titles()

        print(f"📊 Testing with {course_count} existing courses:")
        for title in course_titles:
            print(f"   - {title}")

        assert course_count > 0, "No courses found in database"

        # Test 1: General search
        result1 = self.search_tool.execute("retrieval")
        print(f"\\n🔍 General search for 'retrieval':")
        print(f"   Result length: {len(result1)} characters")
        print(f"   Preview: {result1[:200]}...")

        assert (
            "No relevant content found" not in result1
        ), "Should find content for 'retrieval'"
        assert len(self.search_tool.last_sources) > 0, "Should have sources"

        # Test 2: Course-specific search
        first_course = course_titles[0]
        result2 = self.search_tool.execute("overview", course_name=first_course)
        print(f"\\n🎯 Course-specific search in '{first_course}':")
        print(f"   Result length: {len(result2)} characters")
        print(f"   Preview: {result2[:200]}...")

        assert first_course in result2, f"Should contain course name '{first_course}'"
        assert len(self.search_tool.last_sources) > 0, "Should have sources"

        # Test 3: Lesson-specific search
        result3 = self.search_tool.execute("embedding", lesson_number=1)
        print(f"\\n📚 Lesson-specific search in lesson 1:")
        print(f"   Result length: {len(result3)} characters")
        print(f"   Preview: {result3[:200]}...")

        # Should either find content or return no results message
        assert isinstance(result3, str), "Should return string result"

        print("✅ All CourseSearchTool execute tests passed")

    def test_execute_with_nonexistent_query(self):
        """Test execute method with query that shouldn't match anything"""

        result = self.search_tool.execute("completely_nonexistent_topic_12345")
        print(f"\\n❌ Search for nonexistent topic:")
        print(f"   Result: {result}")

        assert (
            "No relevant content found" in result
        ), "Should return 'no content found' message"
        assert len(self.search_tool.last_sources) == 0, "Should have no sources"

        print("✅ No-results test passed")

    def test_execute_with_course_filter(self):
        """Test execute method with course name filtering"""

        course_titles = self.vector_store.get_existing_course_titles()
        if len(course_titles) > 1:
            # Test with existing course
            target_course = course_titles[0]
            result1 = self.search_tool.execute("data", course_name=target_course)

            print(f"\\n🎯 Filtered search in '{target_course}':")
            print(f"   Result contains course name: {target_course in result1}")

            if "No relevant content found" not in result1:
                assert (
                    target_course in result1
                ), f"Result should mention course '{target_course}'"

            # Test with nonexistent course
            result2 = self.search_tool.execute(
                "data", course_name="Nonexistent Course XYZ"
            )
            print(f"\\n❌ Search in nonexistent course:")
            print(f"   Result: {result2}")

            assert (
                "No course found matching" in result2
            ), "Should indicate course not found"

        print("✅ Course filtering tests passed")

    def test_sources_tracking(self):
        """Test that sources are properly tracked"""

        # Clear any existing sources
        self.search_tool.last_sources = []

        # Perform a search that should return results
        result = self.search_tool.execute("retrieval")

        print(f"\\n📎 Sources tracking test:")
        print(f"   Found {len(self.search_tool.last_sources)} sources")

        if len(self.search_tool.last_sources) > 0:
            for i, source in enumerate(self.search_tool.last_sources[:3]):
                print(f"   Source {i+1}: {source}")

                assert "text" in source, "Source should have 'text' field"
                assert isinstance(source["text"], str), "Source text should be string"

                # Link may or may not be present
                if "link" in source:
                    print(f"     Link: {source['link']}")

        print("✅ Sources tracking test passed")


class TestToolManagerWithRealData:
    """Test ToolManager with real CourseSearchTool"""

    def setup_method(self):
        """Setup tool manager with real search tool"""
        self.vector_store = VectorStore(
            chroma_path=config.CHROMA_PATH,
            embedding_model=config.EMBEDDING_MODEL,
            max_results=config.MAX_RESULTS,
        )
        self.search_tool = CourseSearchTool(self.vector_store)
        self.tool_manager = ToolManager()
        self.tool_manager.register_tool(self.search_tool)

    def test_tool_registration(self):
        """Test that tools are properly registered"""

        definitions = self.tool_manager.get_tool_definitions()
        print(f"\\n🛠️  Registered tools: {len(definitions)}")

        assert len(definitions) == 1, "Should have 1 registered tool"
        assert definitions[0]["name"] == "search_course_content"

        print("✅ Tool registration test passed")

    def test_tool_execution_through_manager(self):
        """Test executing tools through the manager"""

        # Test successful execution
        result = self.tool_manager.execute_tool(
            "search_course_content", query="retrieval"
        )
        print(f"\\n⚙️  Tool execution through manager:")
        print(f"   Result length: {len(result)} characters")
        print(f"   Preview: {result[:150]}...")

        assert isinstance(result, str), "Should return string result"
        assert len(result) > 0, "Should return non-empty result"

        # Test execution of nonexistent tool
        error_result = self.tool_manager.execute_tool("nonexistent_tool", query="test")
        print(f"\\n❌ Nonexistent tool execution:")
        print(f"   Result: {error_result}")

        assert "not found" in error_result, "Should indicate tool not found"

        print("✅ Tool execution tests passed")

    def test_sources_management(self):
        """Test sources management through tool manager"""

        # Execute a search
        result = self.tool_manager.execute_tool(
            "search_course_content", query="embedding"
        )

        # Get sources
        sources = self.tool_manager.get_last_sources()
        print(f"\\n📋 Sources management:")
        print(f"   Found {len(sources)} sources through manager")

        if len(sources) > 0:
            print(f"   First source: {sources[0]}")

        # Reset sources
        self.tool_manager.reset_sources()
        sources_after_reset = self.tool_manager.get_last_sources()

        print(f"   Sources after reset: {len(sources_after_reset)}")
        assert len(sources_after_reset) == 0, "Sources should be empty after reset"

        print("✅ Sources management test passed")


def run_real_data_tests():
    """Run all tests with real data"""
    print("🔬 TESTING COURSE SEARCH TOOL WITH REAL DATA")
    print("=" * 60)

    # Test CourseSearchTool
    search_tester = TestCourseSearchToolRealData()
    search_tester.setup_method()

    try:
        search_tester.test_course_search_tool_definition()
        search_tester.test_execute_with_existing_courses()
        search_tester.test_execute_with_nonexistent_query()
        search_tester.test_execute_with_course_filter()
        search_tester.test_sources_tracking()

        print("\\n✅ All CourseSearchTool tests passed!")

    except Exception as e:
        print(f"\\n❌ CourseSearchTool test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test ToolManager
    manager_tester = TestToolManagerWithRealData()
    manager_tester.setup_method()

    try:
        manager_tester.test_tool_registration()
        manager_tester.test_tool_execution_through_manager()
        manager_tester.test_sources_management()

        print("\\n✅ All ToolManager tests passed!")

    except Exception as e:
        print(f"\\n❌ ToolManager test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\\n" + "=" * 60)
    print("🎯 CONCLUSION: CourseSearchTool and ToolManager are working correctly!")
    print("   - Tool definitions are proper")
    print("   - Execute method works with real data")
    print("   - Sources are tracked correctly")
    print("   - Tool registration and execution work through manager")
    print("   - The issue is NOT in the search tool components")
    print("=" * 60)

    return True


if __name__ == "__main__":
    run_real_data_tests()
