#!/usr/bin/env python3
"""
Live System Integration Tests

These tests work with the actual RAG system components to identify
real failure points in the "query failed" issue.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from ai_generator import AIGenerator
from config import config
from rag_system import RAGSystem
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults, VectorStore

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestLiveSystemComponents:
    """Test actual system components with real data"""

    def setup_method(self):
        """Setup test environment"""
        self.test_db_path = tempfile.mkdtemp()
        print(f"Using test database: {self.test_db_path}")

    def teardown_method(self):
        """Cleanup test environment"""
        if os.path.exists(self.test_db_path):
            shutil.rmtree(self.test_db_path)

    def test_config_api_key_status(self):
        """Test if API key is configured"""
        print(f"\n🔑 API Key Status:")
        print(f"   Has API key: {bool(config.ANTHROPIC_API_KEY)}")
        print(
            f"   Key length: {len(config.ANTHROPIC_API_KEY) if config.ANTHROPIC_API_KEY else 0}"
        )
        print(
            f"   Key preview: {config.ANTHROPIC_API_KEY[:10] + '...' if config.ANTHROPIC_API_KEY and len(config.ANTHROPIC_API_KEY) > 10 else 'EMPTY'}"
        )

        return bool(config.ANTHROPIC_API_KEY)

    def test_vector_store_real_initialization(self):
        """Test vector store initialization with real ChromaDB"""
        print(f"\n📚 Testing Vector Store Initialization:")

        try:
            # Test with temporary database
            vector_store = VectorStore(
                chroma_path=self.test_db_path,
                embedding_model="all-MiniLM-L6-v2",
                max_results=5,
            )

            print("   ✅ Vector store initialized successfully")
            print(f"   📊 Collections created: course_catalog, course_content")

            # Test basic operations
            course_count = vector_store.get_course_count()
            print(f"   📈 Current course count: {course_count}")

            return True, vector_store

        except Exception as e:
            print(f"   ❌ Vector store initialization failed: {e}")
            return False, None

    def test_search_tool_with_real_vector_store(self):
        """Test CourseSearchTool with actual vector store"""
        print(f"\n🔍 Testing CourseSearchTool with Real Vector Store:")

        # Get vector store
        vs_success, vector_store = self.test_vector_store_real_initialization()
        if not vs_success:
            return False

        try:
            # Create search tool
            search_tool = CourseSearchTool(vector_store)

            # Test tool definition
            tool_def = search_tool.get_tool_definition()
            print(f"   ✅ Tool definition: {tool_def['name']}")

            # Test execution with empty database
            result = search_tool.execute("Python programming")
            print(f"   📝 Search result: {result[:100]}...")

            # Should return "No relevant content found" since DB is empty
            if "No relevant content found" in result:
                print("   ✅ Search tool correctly handles empty database")
                return True
            else:
                print(f"   ❌ Unexpected result from empty database: {result}")
                return False

        except Exception as e:
            print(f"   ❌ Search tool test failed: {e}")
            return False

    def test_ai_generator_with_real_api_key_status(self):
        """Test AI generator with current API key configuration"""
        print(f"\n🤖 Testing AI Generator:")

        try:
            # Test initialization
            ai_gen = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)
            print(f"   ✅ AI Generator initialized")
            print(f"   🔧 Model: {config.ANTHROPIC_MODEL}")

            # If no API key, this should fail
            if not config.ANTHROPIC_API_KEY:
                print("   ⚠️  No API key configured - testing failure mode")

                try:
                    result = ai_gen.generate_response("Test query")
                    print(f"   ❌ Unexpected success: {result}")
                    return False
                except Exception as e:
                    if "authentication" in str(e).lower():
                        print(
                            f"   ✅ Correctly failed with auth error: {str(e)[:100]}..."
                        )
                        return "NO_API_KEY"
                    else:
                        print(f"   ❌ Failed with unexpected error: {e}")
                        return False
            else:
                print("   ✅ API key is configured")
                # Don't make actual API call in tests, just verify setup
                return "API_KEY_OK"

        except Exception as e:
            print(f"   ❌ AI Generator initialization failed: {e}")
            return False

    def test_rag_system_initialization(self):
        """Test RAG system initialization"""
        print(f"\n🎯 Testing RAG System Initialization:")

        try:
            # Create RAG system with actual config
            rag_system = RAGSystem(config)

            print("   ✅ RAG system initialized")
            print(f"   🔧 Components created successfully")

            # Check if tools are registered
            tool_definitions = rag_system.tool_manager.get_tool_definitions()
            print(f"   🛠️  Registered tools: {len(tool_definitions)}")

            for tool_def in tool_definitions:
                print(f"      - {tool_def['name']}: {tool_def['description'][:50]}...")

            return True, rag_system

        except Exception as e:
            print(f"   ❌ RAG system initialization failed: {e}")
            import traceback

            traceback.print_exc()
            return False, None

    def test_rag_system_query_flow(self):
        """Test the complete query flow through RAG system"""
        print(f"\n🚀 Testing Complete RAG Query Flow:")

        # Initialize RAG system
        init_success, rag_system = self.test_rag_system_initialization()
        if not init_success:
            return False

        try:
            # Test query without session
            print("   📤 Testing query: 'What is Python programming?'")

            response, sources = rag_system.query("What is Python programming?")

            print(f"   📥 Response received: {len(response)} characters")
            print(f"   📎 Sources count: {len(sources)}")
            print(f"   💬 Response preview: {response[:100]}...")

            if sources:
                print("   📋 Sources:")
                for i, source in enumerate(sources[:3]):
                    print(f"      {i+1}. {source}")

            return True, response, sources

        except Exception as e:
            print(f"   ❌ RAG query failed: {e}")
            print(f"   🔍 Error type: {type(e).__name__}")

            # Check if it's an authentication error
            if "authentication" in str(e).lower():
                print("   🎯 ROOT CAUSE: Authentication error - missing API key")
                return "AUTH_ERROR", str(e), []
            else:
                print("   🔍 Unexpected error - investigating...")
                import traceback

                traceback.print_exc()
                return False, str(e), []


class TestRealDataScenarios:
    """Test with actual course data if available"""

    def test_existing_course_data(self):
        """Test if there's existing course data in the system"""
        print(f"\n📊 Testing Existing Course Data:")

        try:
            # Use actual config to connect to real database
            vector_store = VectorStore(
                chroma_path=config.CHROMA_PATH,
                embedding_model=config.EMBEDDING_MODEL,
                max_results=config.MAX_RESULTS,
            )

            course_count = vector_store.get_course_count()
            print(f"   📈 Courses in database: {course_count}")

            if course_count > 0:
                course_titles = vector_store.get_existing_course_titles()
                print(f"   📚 Available courses:")
                for title in course_titles[:5]:  # Show first 5
                    print(f"      - {title}")

                # Test search with real data
                search_tool = CourseSearchTool(vector_store)
                result = search_tool.execute(
                    "Python", course_name=course_titles[0] if course_titles else None
                )

                print(f"   🔍 Sample search result: {result[:200]}...")
                return True, course_count, course_titles
            else:
                print("   📭 No courses found in database")
                return False, 0, []

        except Exception as e:
            print(f"   ❌ Failed to access existing data: {e}")
            return False, 0, []


def run_comprehensive_live_tests():
    """Run all live system tests"""
    print("🔬 COMPREHENSIVE LIVE SYSTEM ANALYSIS")
    print("=" * 60)

    tester = TestLiveSystemComponents()
    tester.setup_method()

    try:
        # Test 1: Check API key
        has_api_key = tester.test_config_api_key_status()

        # Test 2: Vector store
        vs_success = tester.test_search_tool_with_real_vector_store()

        # Test 3: AI Generator
        ai_status = tester.test_ai_generator_with_real_api_key_status()

        # Test 4: RAG System Query Flow
        query_result = tester.test_rag_system_query_flow()

        # Test 5: Real data
        data_tester = TestRealDataScenarios()
        data_result = data_tester.test_existing_course_data()

        # Analysis
        print("\n" + "=" * 60)
        print("📊 LIVE SYSTEM ANALYSIS RESULTS")
        print("=" * 60)

        print(f"API Key Configured: {'✅' if has_api_key else '❌'}")
        print(f"Vector Store: {'✅' if vs_success else '❌'}")
        print(
            f"AI Generator: {'✅' if ai_status == 'API_KEY_OK' else '❌ (No API Key)' if ai_status == 'NO_API_KEY' else '❌'}"
        )

        if query_result == "AUTH_ERROR":
            print(f"Query Flow: ❌ (Authentication Error)")
            print(f"\n🎯 ROOT CAUSE CONFIRMED:")
            print(
                f"   The system fails because the Anthropic API key is not configured."
            )
            print(f"   All other components are working correctly.")
        elif query_result[0] == True:
            print(f"Query Flow: ✅")
            print(f"   Response: {query_result[1][:100]}...")
        else:
            print(f"Query Flow: ❌")
            print(f"   Error: {query_result[1][:100]}...")

        print(f"Existing Data: {'✅' if data_result[0] else '❌'}")
        if data_result[0]:
            print(f"   Courses available: {data_result[1]}")

    finally:
        tester.teardown_method()


if __name__ == "__main__":
    run_comprehensive_live_tests()
