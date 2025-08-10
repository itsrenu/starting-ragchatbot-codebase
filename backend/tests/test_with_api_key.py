#!/usr/bin/env python3
"""
Test the system with a mock valid API key to see if there are any other issues
"""

import sys
import os
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from config import config

def test_with_mock_valid_api_key():
    """Test the system behavior with a mock valid API key"""
    
    print("🔍 TESTING SYSTEM WITH MOCK API KEY")
    print("=" * 60)
    
    # Mock the config to have an API key
    original_api_key = config.ANTHROPIC_API_KEY
    config.ANTHROPIC_API_KEY = "test_valid_api_key_mock"
    
    try:
        print(f"✅ Mock API key set: {config.ANTHROPIC_API_KEY[:10]}...")
        
        # Initialize RAG system
        rag_system = RAGSystem(config)
        print("✅ RAG system initialized with mock API key")
        
        # Check tool registration
        tools = rag_system.tool_manager.get_tool_definitions()
        print(f"✅ Tools registered: {len(tools)}")
        for tool in tools:
            print(f"   - {tool['name']}")
        
        # Test with mocked Anthropic API
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client
            
            # Mock successful response
            mock_response = Mock()
            mock_response.stop_reason = "end_turn"
            mock_content = Mock()
            mock_content.text = "Based on the course content, retrieval systems help find relevant information..."
            mock_response.content = [mock_content]
            mock_client.messages.create.return_value = mock_response
            
            # Test query
            response, sources = rag_system.query("What is retrieval in AI?")
            
            print(f"✅ Query executed successfully")
            print(f"   Response length: {len(response)}")
            print(f"   Response preview: {response[:100]}...")
            print(f"   Sources count: {len(sources)}")
            
            # Verify API was called
            print(f"   API calls made: {mock_client.messages.create.call_count}")
            
            if mock_client.messages.create.call_count > 0:
                call_args = mock_client.messages.create.call_args[1]
                print(f"   Tools provided to AI: {'tools' in call_args}")
                if 'tools' in call_args:
                    print(f"   Number of tools: {len(call_args['tools'])}")
            
        print("✅ System works correctly with valid API key")
        return True
        
    except Exception as e:
        print(f"❌ System failed even with mock API key: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Restore original API key
        config.ANTHROPIC_API_KEY = original_api_key

def test_server_startup_error_handling():
    """Test what happens during server startup"""
    
    print(f"\n🚀 TESTING SERVER STARTUP BEHAVIOR")
    print("=" * 60)
    
    try:
        from app import rag_system as app_rag_system
        print("✅ App's RAG system imported successfully")
        
        # Check the app's RAG system
        tools = app_rag_system.tool_manager.get_tool_definitions()
        print(f"✅ App RAG system has {len(tools)} tools")
        
        # Try a direct query to see the actual error
        try:
            response, sources = app_rag_system.query("test query")
            print(f"✅ Query succeeded: {response[:50]}...")
        except Exception as e:
            print(f"❌ Query failed with error: {e}")
            print(f"   Error type: {type(e).__name__}")
            
            # Check if it's still an auth error
            if "authentication" in str(e).lower():
                print("   🎯 Confirmed: Still an authentication error")
            else:
                print("   🔍 Different error - investigating...")
                import traceback
                traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"❌ Server startup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_direct_api_call():
    """Test making a direct API call to Anthropic"""
    
    print(f"\n🔑 TESTING DIRECT ANTHROPIC API ACCESS")
    print("=" * 60)
    
    # Test with empty key (current situation)
    try:
        import anthropic
        
        print("✅ Anthropic library imported")
        
        # Test with current config
        if config.ANTHROPIC_API_KEY:
            print(f"   Config has API key: {config.ANTHROPIC_API_KEY[:10]}...")
            client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
            
            try:
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Hello"}]
                )
                print("✅ Direct API call succeeded")
                return True
            except Exception as api_error:
                print(f"❌ Direct API call failed: {api_error}")
                if "authentication" in str(api_error).lower():
                    print("   🎯 API key is invalid or expired")
                else:
                    print("   🔍 Different API error")
                return False
        else:
            print("❌ No API key in config")
            return False
            
    except ImportError:
        print("❌ Anthropic library not available")
        return False
    except Exception as e:
        print(f"❌ Direct API test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔬 COMPREHENSIVE SYSTEM TESTING WITH API KEY INVESTIGATION")
    print("=" * 70)
    
    test1_result = test_with_mock_valid_api_key()
    test2_result = test_server_startup_error_handling()  
    test3_result = test_direct_api_call()
    
    print(f"\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    print(f"Mock API key test: {'✅' if test1_result else '❌'}")
    print(f"Server startup test: {'✅' if test2_result else '❌'}")  
    print(f"Direct API call test: {'✅' if test3_result else '❌'}")
    
    if not any([test1_result, test2_result, test3_result]):
        print("\n❌ All tests indicate API key issues remain")
        print("💡 Recommendation: Verify the API key is valid and properly formatted")
    elif test1_result and not test3_result:
        print("\n✅ System logic works, but API key is the issue")
        print("💡 Recommendation: Check API key validity and format")
    else:
        print("\n🔍 Mixed results - may need additional investigation")