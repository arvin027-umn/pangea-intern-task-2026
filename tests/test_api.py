import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

"""API testing at the HTTP level using FastAPI's TestClient
- Verify the HTTP endpoints, not just internal functions, and catches issues
such as incorrect response status codes, missing headers, validation errors from FastAPI,
and serialization problems that unit tests miss
- LLM calls are mocked for deterministic testing
"""

def _mock_completion(response_data: dict) -> MagicMock:
    """Build a mock OpenAI response object."""
    choice = MagicMock()
    choice.message.content = json.dumps(response_data)
    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = MagicMock()
    completion.usage.prompt_tokens = 100
    completion.usage.completion_tokens = 50
    completion.usage.prompt_tokens_details = None
    return completion


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clears cache before each test"""
    from app.cache import clear_cache
    clear_cache()
    yield
    clear_cache()

@pytest.fixture
def client():
    """Creates a test client for the FastAPI app"""
    return TestClient(app)

def test_health_returns_200(client):
    """GET /health check should return 200 with status ok"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_health_is_get_only(client):
    """POST /health should return 405 Method Not Allowed"""
    response = client.post("/health")
    assert response.status_code == 405

def test_feedback_rejects_empty_body(client):
    """POST /feedback with no request body should return a 422 error"""
    response = client.post("/feedback")
    assert response.status_code == 422

def test_feedback_rejects_missing_sentence(client):
    """POST /feedback without sentence should return a 422 error"""
    response = client.post("/feedback", json={
        "target_language": "Spanish",
        "native_language": "English",
    })
    assert response.status_code == 422

def test_feedback_rejects_empty_sentence(client):
    """POST /feedback with empty sentence should return a 422 error"""
    response = client.post("/feedback", json={
        "sentence": "",
        "target_language": "Spanish",
        "native_language": "English",
    })
    assert response.status_code == 422

def test_feedback_rejects_missing_target_language(client):
    """POST /feedback without target_language should return a 422 error"""
    response = client.post("/feedback", json={
        "sentence": "Hola mundo.",
        "native_language": "English",
    })
    assert response.status_code == 422

def test_feedback_rejects_missing_native_language(client):
    """POST /feedback without native_language should return a 422 error"""
    response = client.post("/feedback", json={
        "sentence": "Hola mundo.",
        "target_language": "Spanish",
    })
    assert response.status_code == 422

def test_feedback_rejects_short_language_code(client):
    """POST /feedback with a 1 character language should return a 422 error"""
    response = client.post("/feedback", json={
        "sentence": "Hola mundo.",
        "target_language": "S",
        "native_language": "English",
    })
    assert response.status_code == 422

def test_feedback_returns_valid_json_for_error_sentence(client):
    """POST /feedback with a valid request should return 200 with correct schema"""
    mock_data = {
        "corrected_sentence": "Yo fui al mercado ayer.",
        "is_correct": False,
        "errors": [
            {
                "original": "soy fue",
                "correction": "fui",
                "error_type": "conjugation",
                "explanation": "You mixed two verb forms.",
            }
        ],
        "difficulty": "A2",
    }

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_completion(mock_data)
    )

    with patch("app.feedback._get_client", return_value=mock_client):
        response = client.post("/feedback", json={
            "sentence": "Yo soy fue al mercado ayer.",
            "target_language": "Spanish",
            "native_language": "English",
        })

    assert response.status_code == 200
    data = response.json()

    # Verify all required fields are present
    assert "corrected_sentence" in data
    assert "is_correct" in data
    assert "errors" in data
    assert "difficulty" in data

    # Verify types
    assert isinstance(data["corrected_sentence"], str)
    assert isinstance(data["is_correct"], bool)
    assert isinstance(data["errors"], list)
    assert isinstance(data["difficulty"], str)

    # Verify values
    assert data["is_correct"] is False
    assert len(data["errors"]) == 1
    assert data["errors"][0]["error_type"] == "conjugation"
    assert data["difficulty"] in ["A1", "A2", "B1", "B2", "C1", "C2"]

def test_feedback_returns_valid_json_for_correct_sentence(client):
    """POST /feedback for a correct sentence should return is_correct=True"""
    mock_data = {
        "corrected_sentence": "Ich bin hier.",
        "is_correct": True,
        "errors": [],
        "difficulty": "A1",
    }

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_completion(mock_data)
    )

    with patch("app.feedback._get_client", return_value=mock_client):
        response = client.post("/feedback", json={
            "sentence": "Ich bin hier.",
            "target_language": "German",
            "native_language": "English",
        })

    assert response.status_code == 200
    data = response.json()
    assert data["is_correct"] is True
    assert data["errors"] == []


def test_feedback_content_type_is_json(client):
    """The response Content-Type should be application/json"""
    mock_data = {
        "corrected_sentence": "Hola.",
        "is_correct": True,
        "errors": [],
        "difficulty": "A1",
    }

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_completion(mock_data)
    )

    with patch("app.feedback._get_client", return_value=mock_client):
        response = client.post("/feedback", json={
            "sentence": "Hola.",
            "target_language": "Spanish",
            "native_language": "English",
        })

    assert "application/json" in response.headers["content-type"]

def test_nonexistent_endpoint_returns_404(client):
    """Requesting a path that doesn't exist should return 404 not found error"""
    response = client.get("/nonexistent")
    assert response.status_code == 404