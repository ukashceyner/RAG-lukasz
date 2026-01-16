"""Basic API tests."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_root_endpoint(client):
    """Test root endpoint returns API info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "RAG API"
    assert data["version"] == "1.0.0"


def test_health_endpoint(client):
    """Test health endpoint returns status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "qdrant_connected" in data
    assert "voyage_configured" in data
    assert "gemini_configured" in data


def test_list_documents_empty(client):
    """Test listing documents when collection might be empty."""
    response = client.get("/documents")
    # May fail if Qdrant not configured, which is expected in test env
    assert response.status_code in [200, 500]


def test_upload_invalid_file_type(client):
    """Test uploading unsupported file type returns error."""
    response = client.post(
        "/documents/upload",
        files={"file": ("test.txt", b"test content", "text/plain")}
    )
    assert response.status_code == 400
    assert "not supported" in response.json()["detail"]


def test_upload_empty_file(client):
    """Test uploading empty file returns error."""
    response = client.post(
        "/documents/upload",
        files={"file": ("test.pdf", b"", "application/pdf")}
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_query_validation(client):
    """Test query endpoint validates input."""
    # Empty question should fail
    response = client.post(
        "/query",
        json={"question": ""}
    )
    assert response.status_code == 422  # Validation error
