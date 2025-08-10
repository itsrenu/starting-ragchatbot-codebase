import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import asyncio
from fastapi.testclient import TestClient
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Add backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from config import config
from rag_system import RAGSystem


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config(temp_dir):
    """Create a mock config for testing."""
    from types import SimpleNamespace
    
    test_config = SimpleNamespace()
    test_config.anthropic_api_key = "test-key-12345"
    test_config.model_name = "claude-sonnet-4-20250514"
    test_config.max_tokens = 1000
    test_config.temperature = 0.0
    test_config.chunk_size = 800
    test_config.chunk_overlap = 100
    test_config.embedding_model_name = "all-MiniLM-L6-v2"
    test_config.chroma_db_path = os.path.join(temp_dir, "test_chroma_db")
    test_config.collection_name = "test_course_chunks"
    test_config.course_collection_name = "test_courses"
    test_config.search_results_limit = 5
    
    return test_config


@pytest.fixture
def mock_rag_system(mock_config):
    """Create a mock RAG system for testing."""
    mock_rag = Mock()
    mock_rag.query.return_value = ("Test answer", [{"text": "Test source", "link": None}])
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Test Course 1", "Test Course 2"]
    }
    mock_rag.session_manager.create_session.return_value = "test-session-123"
    return mock_rag


@pytest.fixture
def test_app():
    """Create a clean test FastAPI app without static file mounting."""
    return create_test_app()


@pytest.fixture 
def client(test_app, mock_rag_system):
    """Create a test client for the FastAPI app."""
    # Inject the mock RAG system into the app
    test_app.state.rag_system = mock_rag_system
    return TestClient(test_app)


def create_test_app():
    """Factory function to create a test app without import issues."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List, Optional
    
    # Create test app
    app = FastAPI(title="Test Course Materials RAG System")
    
    # Add middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # Pydantic models
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class Source(BaseModel):
        text: str
        link: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Source]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    # API endpoints
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            rag_system = app.state.rag_system
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()
            
            answer, raw_sources = rag_system.query(request.query, session_id)
            
            sources = []
            for source in raw_sources:
                if isinstance(source, dict):
                    sources.append(Source(text=source["text"], link=source.get("link")))
                else:
                    sources.append(Source(text=str(source), link=None))
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            rag_system = app.state.rag_system
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def read_root():
        return {"message": "Test app running"}
    
    return app


@pytest.fixture
def sample_query_data():
    """Sample query data for testing."""
    return {
        "valid_query": {
            "query": "What is machine learning?",
            "session_id": "test-session-123"
        },
        "query_without_session": {
            "query": "Explain neural networks"
        },
        "empty_query": {
            "query": ""
        }
    }


@pytest.fixture
def sample_course_data():
    """Sample course data for testing."""
    return {
        "courses": [
            {
                "title": "Introduction to Machine Learning",
                "instructor": "Dr. Smith",
                "description": "Basic concepts of ML"
            },
            {
                "title": "Deep Learning Fundamentals",
                "instructor": "Prof. Johnson",
                "description": "Neural networks and deep learning"
            }
        ],
        "chunks": [
            {
                "content": "Machine learning is a subset of artificial intelligence...",
                "course_title": "Introduction to Machine Learning",
                "chunk_id": "chunk-1"
            },
            {
                "content": "Neural networks are computational models...",
                "course_title": "Deep Learning Fundamentals", 
                "chunk_id": "chunk-2"
            }
        ]
    }


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing."""
    with patch('anthropic.Anthropic') as mock_client:
        mock_response = Mock()
        mock_response.content = [Mock(text="Mocked AI response")]
        mock_response.usage = Mock(input_tokens=10, output_tokens=20)
        
        mock_client.return_value.messages.create.return_value = mock_response
        yield mock_client


@pytest.fixture
def mock_environment_variables():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'test-api-key-12345',
        'MODEL_NAME': 'claude-sonnet-4-20250514'
    }):
        yield


@pytest.fixture(autouse=True)
def suppress_warnings():
    """Suppress common warnings during testing."""
    import warnings
    warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)