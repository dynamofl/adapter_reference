import pytest
from fastapi.testclient import TestClient
import json
import os
from unittest.mock import MagicMock, patch

# Import app module - the path may need adjustment
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app, ENABLE_AUTH, API_KEY_SECRET

# Create test client
client = TestClient(app)

# Mock OpenAI API responses
MOCK_COMPLETION_RESPONSE = {
    "id": "cmpl-mock123",
    "object": "text_completion",
    "created": 1677858242,
    "model": "gpt-3.5-turbo-instruct",
    "choices": [
        {
            "text": "This is a test response",
            "index": 0,
            "logprobs": None,
            "finish_reason": "length"
        }
    ],
    "usage": {
        "prompt_tokens": 5,
        "completion_tokens": 7,
        "total_tokens": 12
    }
}

MOCK_CHAT_COMPLETION_RESPONSE = {
    "id": "chatcmpl-mock123",
    "object": "chat.completion",
    "created": 1677858242,
    "model": "gpt-3.5-turbo",
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "This is a test response"
            },
            "index": 0,
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 14,
        "completion_tokens": 7,
        "total_tokens": 21
    }
}

MOCK_EMBEDDINGS_RESPONSE = {
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "embedding": [0.1, 0.2, 0.3],
            "index": 0
        }
    ],
    "model": "text-embedding-ada-002",
    "usage": {
        "prompt_tokens": 5,
        "total_tokens": 5
    }
}


# Setup environment for tests
@pytest.fixture(autouse=True)
def setup_environment():
    """Set up test environment variables"""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key", "ENABLE_AUTH": "false"}):
        yield


# Tests for basic endpoints
def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


# Tests for API endpoints
@patch("app.client")
def test_chat_completions(mock_client):
    """Test chat completions endpoint"""
    # Mock the OpenAI client response
    mock_response = MagicMock()
    mock_response.model_dump.return_value = MOCK_CHAT_COMPLETION_RESPONSE
    mock_client.chat.completions.create.return_value = mock_response
    
    # Make request to endpoint
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}]
        }
    )
    
    # Assertions
    assert response.status_code == 200
    
    # Verify OpenAI client was called
    mock_client.chat.completions.create.assert_called_once()


@patch("app.client")
def test_completions(mock_client):
    """Test completions endpoint"""
    # Mock the OpenAI client response
    mock_response = MagicMock()
    mock_response.model_dump.return_value = MOCK_COMPLETION_RESPONSE
    mock_client.completions.create.return_value = mock_response
    
    # Make request to endpoint
    response = client.post(
        "/v1/completions",
        json={
            "model": "gpt-3.5-turbo-instruct",
            "prompt": "Hello"
        }
    )
    
    # Assertions
    assert response.status_code == 200
    
    # Verify OpenAI client was called
    mock_client.completions.create.assert_called_once()


@patch("app.client")
def test_embeddings(mock_client):
    """Test embeddings endpoint"""
    # Mock the OpenAI client response
    mock_response = MagicMock()
    mock_response.model_dump.return_value = MOCK_EMBEDDINGS_RESPONSE
    mock_client.embeddings.create.return_value = mock_response
    
    # Make request to endpoint
    response = client.post(
        "/v1/embeddings",
        json={
            "model": "text-embedding-ada-002",
            "input": "Hello"
        }
    )
    
    # Assertions
    assert response.status_code == 200
    
    # Verify OpenAI client was called
    mock_client.embeddings.create.assert_called_once()


# Error handling tests
def test_validation_error():
    """Test validation error handling"""
    # Make request with invalid data
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": []  # Empty messages array should fail validation
        }
    )
    
    # Assertions
    assert response.status_code == 422
    assert "error" in response.json()
    assert response.json()["error"] == "ValidationError"


@patch("app.ENABLE_AUTH", True)
def test_authentication():
    """Test authentication requirement when enabled"""
    with patch.dict(os.environ, {"ENABLE_AUTH": "true"}):
        # Make request without API key
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        
        # Assertions
        assert response.status_code == 401
        
        # Make request with valid API key
        response = client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": API_KEY_SECRET},
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        
        # This might still fail if the OpenAI client is not properly mocked
        # But it should pass authentication check
        assert response.status_code != 401


# Cache and rate limiting tests can be added as well