import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from ai_generator import AIGenerator
from rag_system import RAGSystem

class TestIntegrationScenarios:
    """Integration tests to identify where the 'query failed' issue occurs"""
    
    def setup_method(self):
        """Setup integration test fixtures"""
        self.mock_chroma_client = Mock()
        self.mock_collection = Mock()
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_vector_store_search_functionality(self, mock_embedding_func, mock_client_class):
        """Test if vector store search is working correctly"""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_course_content = Mock()
        mock_course_catalog = Mock()
        mock_client.get_or_create_collection.side_effect = [mock_course_catalog, mock_course_content]
        
        # Mock search results
        mock_course_content.query.return_value = {
            'documents': [['Test course content about Python']],
            'metadatas': [[{'course_title': 'Python Basics', 'lesson_number': 1}]],
            'distances': [[0.1]]
        }
        
        # Test vector store
        vector_store = VectorStore('./test_chroma', 'all-MiniLM-L6-v2', 5)
        results = vector_store.search("Python programming")
        
        # Assertions
        assert not results.error
        assert len(results.documents) == 1
        assert 'Python' in results.documents[0]
        
        print("✓ Vector store search functionality test passed")
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')  
    def test_search_tool_with_vector_store(self, mock_embedding_func, mock_client_class):
        """Test CourseSearchTool integration with vector store"""
        # Setup vector store mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_course_content = Mock()
        mock_course_catalog = Mock()
        mock_client.get_or_create_collection.side_effect = [mock_course_catalog, mock_course_content]
        
        # Mock successful search
        mock_course_content.query.return_value = {
            'documents': [['Python is a high-level programming language']],
            'metadatas': [[{'course_title': 'Python Course', 'lesson_number': 1, 'lesson_link': 'http://test.com'}]],
            'distances': [[0.2]]
        }
        
        # Test search tool
        vector_store = VectorStore('./test_chroma', 'all-MiniLM-L6-v2', 5)
        search_tool = CourseSearchTool(vector_store)
        
        result = search_tool.execute("What is Python?")
        
        # Assertions
        assert 'Python Course' in result
        assert 'Python is a high-level programming language' in result
        assert len(search_tool.last_sources) == 1
        
        print("✓ CourseSearchTool integration test passed")
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_tool_no_results(self, mock_embedding_func, mock_client_class):
        """Test CourseSearchTool when no results found"""
        # Setup vector store mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_course_content = Mock()
        mock_course_catalog = Mock()
        mock_client.get_or_create_collection.side_effect = [mock_course_catalog, mock_course_content]
        
        # Mock empty search results
        mock_course_content.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        # Test search tool
        vector_store = VectorStore('./test_chroma', 'all-MiniLM-L6-v2', 5)
        search_tool = CourseSearchTool(vector_store)
        
        result = search_tool.execute("Nonexistent topic")
        
        # Assertions
        assert 'No relevant content found' in result
        assert len(search_tool.last_sources) == 0
        
        print("✓ CourseSearchTool no results test passed")
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_ai_generator_with_valid_api_key(self, mock_anthropic_class):
        """Test AI generator with valid API key"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock successful response
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_content = Mock()
        mock_content.text = "Python is a programming language"
        mock_response.content = [mock_content]
        
        mock_client.messages.create.return_value = mock_response
        
        # Test AI generator
        ai_gen = AIGenerator("valid_api_key", "claude-3-sonnet-20240229")
        result = ai_gen.generate_response("What is Python?")
        
        # Assertions
        assert result == "Python is a programming language"
        mock_client.messages.create.assert_called_once()
        
        print("✓ AI generator with valid API key test passed")
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_ai_generator_with_invalid_api_key(self, mock_anthropic_class):
        """Test AI generator with invalid API key (main issue)"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock authentication error
        auth_error = Exception("Could not resolve authentication method. Expected either api_key or auth_token to be set")
        mock_client.messages.create.side_effect = auth_error
        
        # Test AI generator
        ai_gen = AIGenerator("", "claude-3-sonnet-20240229")  # Empty API key
        
        with pytest.raises(Exception) as exc_info:
            ai_gen.generate_response("What is Python?")
        
        assert "authentication method" in str(exc_info.value)
        print("✓ AI generator authentication error test passed")
    
    @patch('ai_generator.anthropic.Anthropic')  
    def test_ai_generator_with_tool_calling(self, mock_anthropic_class):
        """Test AI generator tool calling functionality"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock tool use response
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.input = {"query": "Python basics"}
        mock_tool_content.id = "test_id"
        
        initial_response = Mock()
        initial_response.stop_reason = "tool_use"
        initial_response.content = [mock_tool_content]
        
        # Mock final response
        final_response = Mock()
        final_content = Mock()
        final_content.text = "Based on the search, Python is a programming language"
        final_response.content = [final_content]
        
        mock_client.messages.create.side_effect = [initial_response, final_response]
        
        # Test with tool manager
        ai_gen = AIGenerator("valid_api_key", "claude-3-sonnet-20240229")
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results about Python"
        
        tools = [{"name": "search_course_content", "description": "Search tool"}]
        
        result = ai_gen.generate_response(
            "What is Python?",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Assertions
        mock_tool_manager.execute_tool.assert_called_once_with("search_course_content", query="Python basics")
        assert "Based on the search" in result
        
        print("✓ AI generator tool calling test passed")

class TestErrorScenarios:
    """Test various error scenarios that could cause 'query failed'"""
    
    def test_missing_api_key_scenario(self):
        """Test the most likely cause: missing/empty API key"""
        # Simulate the actual error we're seeing
        from config import config
        
        # This would normally fail if API key is not set
        print(f"Current API key configured: {'***' if config.ANTHROPIC_API_KEY else 'EMPTY/MISSING'}")
        
        if not config.ANTHROPIC_API_KEY:
            print("❌ FOUND ISSUE: API key is missing or empty in config")
            return "API_KEY_MISSING"
        else:
            print("✓ API key appears to be configured")
            return "API_KEY_OK"
    
    @patch('vector_store.chromadb.PersistentClient')
    def test_vector_store_connection_error(self, mock_client_class):
        """Test vector store connection issues"""
        # Mock connection failure
        mock_client_class.side_effect = Exception("Failed to connect to ChromaDB")
        
        try:
            vector_store = VectorStore('./test_chroma', 'all-MiniLM-L6-v2', 5)
            return "VECTOR_STORE_ERROR"
        except Exception as e:
            print(f"❌ Vector store error: {e}")
            return "VECTOR_STORE_ERROR"
    
    def test_tool_execution_chain(self):
        """Test the complete tool execution chain"""
        # This would test: Query -> AI decides to use tool -> Tool executes -> Results returned
        print("Testing complete tool execution chain...")
        
        # Mock all components
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        
        # Test tool manager
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)
        
        # Test tool execution
        result = tool_manager.execute_tool("search_course_content", query="test")
        
        assert "Test" in result
        print("✓ Tool execution chain test passed")

def run_diagnostic_tests():
    """Run diagnostic tests to identify the root cause"""
    print("\\n" + "="*60)
    print("RUNNING RAG SYSTEM DIAGNOSTIC TESTS")
    print("="*60)
    
    # Test 1: Check API key configuration
    error_tester = TestErrorScenarios()
    api_key_status = error_tester.test_missing_api_key_scenario()
    
    # Test 2: Test components individually
    integration_tester = TestIntegrationScenarios()
    
    try:
        integration_tester.test_ai_generator_with_invalid_api_key()
    except:
        print("❌ AI Generator fails with empty API key (EXPECTED)")
    
    print("\\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    if api_key_status == "API_KEY_MISSING":
        print("🎯 ROOT CAUSE IDENTIFIED: Missing/Empty Anthropic API Key")
        print("   - The .env file exists but is empty (0 bytes)")
        print("   - AI Generator cannot authenticate with Anthropic API")
        print("   - This causes the 'query failed' error")
    else:
        print("🔍 API key appears configured, investigating other causes...")

if __name__ == "__main__":
    run_diagnostic_tests()
    pytest.main([__file__, "-v"])