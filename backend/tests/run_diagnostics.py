#!/usr/bin/env python3
"""
RAG System Diagnostic Test Runner

This script runs comprehensive tests to identify why the RAG system
returns 'query failed' for content-related questions.

Usage:
    cd backend
    python tests/run_diagnostics.py
"""

import os
import subprocess
import sys
from pathlib import Path

# Add parent directory to Python path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


def check_environment():
    """Check if the environment is properly configured"""
    print("🔍 CHECKING ENVIRONMENT CONFIGURATION")
    print("-" * 50)

    issues = []

    # Check if .env file exists and has content
    env_file = parent_dir.parent / ".env"
    if not env_file.exists():
        issues.append("❌ .env file does not exist")
    else:
        env_size = env_file.stat().st_size
        if env_size == 0:
            issues.append("❌ .env file exists but is EMPTY (0 bytes)")
            print(f"   📁 .env file location: {env_file}")
            print(f"   📏 File size: {env_size} bytes")
        else:
            print(f"✅ .env file exists and has content ({env_size} bytes)")

    # Check if required Python packages are available
    try:
        import chromadb

        print("✅ chromadb package available")
    except ImportError:
        issues.append("❌ chromadb package not installed")

    try:
        import anthropic

        print("✅ anthropic package available")
    except ImportError:
        issues.append("❌ anthropic package not installed")

    try:
        import sentence_transformers

        print("✅ sentence_transformers package available")
    except ImportError:
        issues.append("❌ sentence_transformers package not installed")

    return issues


def test_config_loading():
    """Test if configuration loads correctly"""
    print("\n🔧 TESTING CONFIGURATION LOADING")
    print("-" * 50)

    try:
        from config import config

        print(f"✅ Config loaded successfully")
        print(f"   📊 Chunk size: {config.CHUNK_SIZE}")
        print(f"   📊 Max results: {config.MAX_RESULTS}")
        print(f"   🤖 Model: {config.ANTHROPIC_MODEL}")

        # Check API key (without revealing it)
        if hasattr(config, "ANTHROPIC_API_KEY") and config.ANTHROPIC_API_KEY:
            key_preview = (
                config.ANTHROPIC_API_KEY[:10] + "..."
                if len(config.ANTHROPIC_API_KEY) > 10
                else "short"
            )
            print(f"   🔑 API key configured: {key_preview}")
            return True
        else:
            print(f"   ❌ API key NOT configured")
            return False

    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False


def test_vector_store_mock():
    """Test vector store with mocked dependencies"""
    print("\n📚 TESTING VECTOR STORE (MOCKED)")
    print("-" * 50)

    try:
        from unittest.mock import Mock, patch

        with (
            patch("vector_store.chromadb.PersistentClient") as mock_client_class,
            patch(
                "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
        ):

            mock_client = Mock()
            mock_client_class.return_value = mock_client

            mock_collection = Mock()
            mock_client.get_or_create_collection.return_value = mock_collection

            # Mock successful search
            mock_collection.query.return_value = {
                "documents": [["Test content"]],
                "metadatas": [[{"course_title": "Test Course", "lesson_number": 1}]],
                "distances": [[0.1]],
            }

            from vector_store import VectorStore

            vector_store = VectorStore("./test_db", "test-model", 5)
            results = vector_store.search("test query")

            if not results.error and len(results.documents) > 0:
                print("✅ Vector store search working (mocked)")
                return True
            else:
                print("❌ Vector store search failed (mocked)")
                return False

    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        return False


def test_search_tool_mock():
    """Test search tool with mocked dependencies"""
    print("\n🔍 TESTING COURSE SEARCH TOOL (MOCKED)")
    print("-" * 50)

    try:
        from unittest.mock import Mock

        from search_tools import CourseSearchTool
        from vector_store import SearchResults

        # Create mock vector store
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = SearchResults(
            documents=["Python is a programming language"],
            metadata=[{"course_title": "Python Course", "lesson_number": 1}],
            distances=[0.1],
            error=None,
        )

        # Test search tool
        search_tool = CourseSearchTool(mock_vector_store)
        result = search_tool.execute("What is Python?")

        if "Python Course" in result and "programming language" in result:
            print("✅ CourseSearchTool working (mocked)")
            return True
        else:
            print(f"❌ CourseSearchTool failed: {result}")
            return False

    except Exception as e:
        print(f"❌ CourseSearchTool test failed: {e}")
        return False


def test_ai_generator_mock():
    """Test AI generator with mocked Anthropic API"""
    print("\n🤖 TESTING AI GENERATOR (MOCKED)")
    print("-" * 50)

    try:
        from unittest.mock import Mock, patch

        from ai_generator import AIGenerator

        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            # Mock successful response
            mock_response = Mock()
            mock_response.stop_reason = "end_turn"
            mock_content = Mock()
            mock_content.text = "Python is a programming language"
            mock_response.content = [mock_content]
            mock_client.messages.create.return_value = mock_response

            # Test AI generator
            ai_gen = AIGenerator("test_key", "claude-3-sonnet-20240229")
            result = ai_gen.generate_response("What is Python?")

            if "programming language" in result:
                print("✅ AI Generator working (mocked)")
                return True
            else:
                print(f"❌ AI Generator failed: {result}")
                return False

    except Exception as e:
        print(f"❌ AI Generator test failed: {e}")
        return False


def test_ai_generator_real_empty_key():
    """Test AI generator with real (empty) API key to reproduce error"""
    print("\n🚨 TESTING AI GENERATOR WITH EMPTY API KEY (REAL)")
    print("-" * 50)

    try:
        from ai_generator import AIGenerator

        # Test with empty API key (this should fail like in production)
        ai_gen = AIGenerator("", "claude-3-sonnet-20240229")
        result = ai_gen.generate_response("What is Python?")

        print(f"❌ Unexpected success with empty key: {result}")
        return False

    except Exception as e:
        error_msg = str(e)
        if "authentication" in error_msg.lower():
            print(f"✅ Correctly failed with authentication error:")
            print(f"   📝 Error: {error_msg}")
            return True
        else:
            print(f"❌ Failed with unexpected error: {error_msg}")
            return False


def diagnose_issue():
    """Run comprehensive diagnosis"""
    print("\n" + "=" * 70)
    print("🔬 RAG SYSTEM COMPREHENSIVE DIAGNOSIS")
    print("=" * 70)

    # Step 1: Environment check
    env_issues = check_environment()

    # Step 2: Config loading
    config_ok = test_config_loading()

    # Step 3: Component tests (mocked)
    vector_store_ok = test_vector_store_mock()
    search_tool_ok = test_search_tool_mock()
    ai_generator_mock_ok = test_ai_generator_mock()

    # Step 4: Real API test with empty key
    ai_generator_real_fail = test_ai_generator_real_empty_key()

    # Analysis
    print("\n" + "=" * 70)
    print("📊 DIAGNOSIS RESULTS")
    print("=" * 70)

    if env_issues:
        print("🚨 ENVIRONMENT ISSUES FOUND:")
        for issue in env_issues:
            print(f"   {issue}")

    print(f"\n📋 COMPONENT STATUS:")
    print(f"   Config Loading: {'✅' if config_ok else '❌'}")
    print(f"   Vector Store: {'✅' if vector_store_ok else '❌'}")
    print(f"   Search Tool: {'✅' if search_tool_ok else '❌'}")
    print(f"   AI Generator (mocked): {'✅' if ai_generator_mock_ok else '❌'}")
    print(
        f"   AI Generator (real empty key): {'✅ Fails as expected' if ai_generator_real_fail else '❌ Unexpected behavior'}"
    )

    # Root cause analysis
    print(f"\n🎯 ROOT CAUSE ANALYSIS:")

    if not config_ok and any(
        "env file exists but is EMPTY" in issue for issue in env_issues
    ):
        print("   🔴 PRIMARY ISSUE: Empty .env file")
        print("      - The .env file exists but contains no API key")
        print("      - AI Generator cannot authenticate with Anthropic")
        print("      - This causes all queries to fail with authentication error")
        print("\n   💡 SOLUTION:")
        print("      1. Add ANTHROPIC_API_KEY=your_actual_api_key to .env file")
        print("      2. Restart the server")
        print("      3. Test with a sample query")

    elif not vector_store_ok:
        print("   🔴 SECONDARY ISSUE: Vector store problems")
        print("      - ChromaDB connection or embedding issues")

    elif not search_tool_ok:
        print("   🔴 SECONDARY ISSUE: Search tool problems")
        print("      - Tool execution logic issues")

    else:
        print("   🟡 Components appear functional individually")
        print("      - Issue may be in integration or real API calls")


def main():
    """Main diagnostic function"""
    print("🚀 Starting RAG System Diagnostics...")

    try:
        diagnose_issue()

        print("\n" + "=" * 70)
        print("✅ DIAGNOSTIC COMPLETE")
        print("=" * 70)
        print("💡 Check the analysis above for the root cause and solution.")

    except Exception as e:
        print(f"\n❌ Diagnostic failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
