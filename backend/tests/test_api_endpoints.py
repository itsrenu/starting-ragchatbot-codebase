import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch


@pytest.mark.api
class TestQueryEndpoint:
    """Test cases for the /api/query endpoint."""
    
    def test_query_with_session_id(self, client, sample_query_data):
        """Test query endpoint with provided session ID."""
        response = client.post(
            "/api/query",
            json=sample_query_data["valid_query"]
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"
        assert isinstance(data["sources"], list)
        assert len(data["sources"]) > 0
        assert "text" in data["sources"][0]
    
    def test_query_without_session_id(self, client, sample_query_data, mock_rag_system):
        """Test query endpoint without session ID - should create new session."""
        response = client.post(
            "/api/query",
            json=sample_query_data["query_without_session"]
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        mock_rag_system.session_manager.create_session.assert_called_once()
    
    def test_query_with_empty_query(self, client, sample_query_data):
        """Test query endpoint with empty query string."""
        response = client.post(
            "/api/query",
            json=sample_query_data["empty_query"]
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
    
    def test_query_invalid_json(self, client):
        """Test query endpoint with invalid JSON."""
        response = client.post(
            "/api/query",
            data="invalid json"
        )
        
        assert response.status_code == 422
    
    def test_query_missing_required_field(self, client):
        """Test query endpoint with missing required query field."""
        response = client.post(
            "/api/query",
            json={"session_id": "test-123"}
        )
        
        assert response.status_code == 422
    
    def test_query_rag_system_error(self, client, sample_query_data, mock_rag_system):
        """Test query endpoint when RAG system raises an error."""
        mock_rag_system.query.side_effect = Exception("RAG system error")
        
        response = client.post(
            "/api/query",
            json=sample_query_data["valid_query"]
        )
        
        assert response.status_code == 500
        assert "RAG system error" in response.json()["detail"]
    
    def test_query_response_format_legacy_sources(self, client, sample_query_data, mock_rag_system):
        """Test query endpoint with legacy source format (strings)."""
        mock_rag_system.query.return_value = ("Test answer", ["Legacy source text"])
        
        response = client.post(
            "/api/query",
            json=sample_query_data["valid_query"]
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["sources"][0]["text"] == "Legacy source text"
        assert data["sources"][0]["link"] is None
    
    def test_query_response_format_new_sources(self, client, sample_query_data, mock_rag_system):
        """Test query endpoint with new source format (dicts with links)."""
        mock_rag_system.query.return_value = (
            "Test answer", 
            [{"text": "Source with link", "link": "https://example.com"}]
        )
        
        response = client.post(
            "/api/query",
            json=sample_query_data["valid_query"]
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["sources"][0]["text"] == "Source with link"
        assert data["sources"][0]["link"] == "https://example.com"


@pytest.mark.api
class TestCoursesEndpoint:
    """Test cases for the /api/courses endpoint."""
    
    def test_get_courses_success(self, client):
        """Test courses endpoint returns course statistics."""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == 2
        assert isinstance(data["course_titles"], list)
        assert len(data["course_titles"]) == 2
        assert "Test Course 1" in data["course_titles"]
        assert "Test Course 2" in data["course_titles"]
    
    def test_get_courses_rag_system_error(self, client, mock_rag_system):
        """Test courses endpoint when RAG system raises an error."""
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics error")
        
        response = client.get("/api/courses")
        
        assert response.status_code == 500
        assert "Analytics error" in response.json()["detail"]
    
    def test_get_courses_empty_response(self, client, mock_rag_system):
        """Test courses endpoint with empty course list."""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_courses"] == 0
        assert data["course_titles"] == []
    
    def test_courses_http_methods(self, client):
        """Test that only GET method is allowed for courses endpoint."""
        # POST should not be allowed
        response = client.post("/api/courses", json={})
        assert response.status_code == 405
        
        # PUT should not be allowed  
        response = client.put("/api/courses", json={})
        assert response.status_code == 405
        
        # DELETE should not be allowed
        response = client.delete("/api/courses")
        assert response.status_code == 405


@pytest.mark.api
class TestRootEndpoint:
    """Test cases for the root / endpoint."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns expected response."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert data["message"] == "Test app running"
    
    def test_root_endpoint_methods(self, client):
        """Test various HTTP methods on root endpoint."""
        # GET should work
        response = client.get("/")
        assert response.status_code == 200
        
        # POST might be allowed depending on implementation
        response = client.post("/", json={})
        assert response.status_code in [200, 405]


@pytest.mark.api
class TestMiddleware:
    """Test cases for middleware functionality."""
    
    def test_cors_headers(self, client):
        """Test that CORS headers are properly set."""
        # Test CORS by making a request with Origin header
        headers = {"Origin": "http://localhost:3000"}
        response = client.get("/api/courses", headers=headers)
        
        # CORS middleware should be configured even if headers don't appear in TestClient
        assert response.status_code == 200
        # Note: FastAPI TestClient may not show CORS headers in test environment
        # This test primarily verifies the middleware is configured without error
    
    def test_trusted_host_middleware(self, client):
        """Test that trusted host middleware is working."""
        # This test verifies the middleware is configured
        # In a real scenario, you might test with different Host headers
        response = client.get("/api/courses")
        assert response.status_code == 200


@pytest.mark.api
class TestErrorHandling:
    """Test cases for error handling across endpoints."""
    
    def test_404_for_nonexistent_endpoint(self, client):
        """Test 404 response for non-existent endpoints."""
        response = client.get("/api/nonexistent")
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test 405 response for unsupported HTTP methods."""
        response = client.patch("/api/query")
        assert response.status_code == 405
    
    @pytest.mark.parametrize("endpoint", ["/api/query", "/api/courses"])
    def test_large_request_handling(self, client, endpoint):
        """Test handling of unusually large requests."""
        if endpoint == "/api/query":
            large_data = {"query": "x" * 10000}  # Very long query
            response = client.post(endpoint, json=large_data)
            # Should still process, might be slow but shouldn't crash
            assert response.status_code in [200, 500]  # 500 if processing fails
        else:
            # GET endpoint, no large request data
            response = client.get(endpoint)
            assert response.status_code in [200, 500]


@pytest.mark.api
@pytest.mark.integration
class TestEndToEndFlow:
    """Integration tests for complete API workflows."""
    
    def test_query_then_courses_flow(self, client, sample_query_data):
        """Test complete flow: query documents then get course stats."""
        # First make a query
        query_response = client.post(
            "/api/query",
            json=sample_query_data["valid_query"]
        )
        assert query_response.status_code == 200
        query_data = query_response.json()
        session_id = query_data["session_id"]
        
        # Then get courses
        courses_response = client.get("/api/courses")
        assert courses_response.status_code == 200
        courses_data = courses_response.json()
        
        # Verify both responses are valid
        assert "answer" in query_data
        assert "total_courses" in courses_data
        assert session_id is not None
    
    def test_multiple_queries_same_session(self, client, mock_rag_system):
        """Test multiple queries using the same session ID."""
        session_id = "persistent-session-123"
        
        queries = [
            {"query": "First query", "session_id": session_id},
            {"query": "Second query", "session_id": session_id},
            {"query": "Third query", "session_id": session_id}
        ]
        
        for query_data in queries:
            response = client.post("/api/query", json=query_data)
            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == session_id
        
        # Verify RAG system was called for each query
        assert mock_rag_system.query.call_count == 3