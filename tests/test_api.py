"""
test_api.py
===========
Pytest suite for the FastAPI application.

Tests:
    - Health check endpoint
    - Metrics endpoint
    - Query endpoint (mocked RAG pipeline)
    - Error handling
    - Request validation

Run:
    pytest tests/test_api.py -v
"""

from __future__ import annotations

import sys
import os
import pytest
from fastapi.testclient import TestClient

# Path bootstrap
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Mock the heavy imports before they're loaded by main.py
from unittest.mock import Mock, MagicMock, patch

# ---- Mock ChromaDB and sentence-transformers BEFORE importing main.py ----
mock_chroma = MagicMock()
mock_chroma.HttpClient = Mock(return_value=Mock())
sys.modules["chromadb"] = mock_chroma

mock_sentence_transformers = MagicMock()
sys.modules["sentence_transformers"] = mock_sentence_transformers

# Now safe to import
from api.main import app
from generation.rag_pipeline import RAGResponse, RetrievedChunk


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_rag_pipeline(monkeypatch):
    """Mock the RAG pipeline to return a canned response."""
    def mock_run(self, query: str) -> RAGResponse:
        return RAGResponse(
            request_id="test-request-id",
            query=query,
            answer="This is a test answer from the mocked RAG pipeline.",
            raw_answer="This is a test answer from the mocked RAG pipeline.",
            chunks_used=[
                RetrievedChunk(
                    chunk_id="chunk_0",
                    text="Test chunk content.",
                    score=0.85,
                    metadata={"source": "test_doc.txt"},
                )
            ],
            faithfulness_score=0.92,
            passed_faithfulness=True,
            retries_used=0,
            latency_ms=123.45,
            audit_log=[],
            metadata={"timestamp": "2026-02-03T00:00:00Z"},
        )

    from generation import rag_pipeline
    monkeypatch.setattr(rag_pipeline.RAGPipeline, "run", mock_run)


# ===========================================================================
# Tests
# ===========================================================================


class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        """Health endpoint should return 200 when services are healthy."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """Health response must have required fields."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "chromadb_connected" in data
        assert "model_loaded" in data

    def test_health_status_healthy(self, client):
        """Health status should be 'healthy' when all checks pass."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"


class TestMetricsEndpoint:

    def test_metrics_returns_200(self, client):
        """Metrics endpoint should return 200."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_content_type(self, client):
        """Metrics should return Prometheus plain text format."""
        response = client.get("/metrics")
        assert "text/plain" in response.headers["content-type"]

    def test_metrics_contains_expected_metrics(self, client):
        """Metrics output should contain our custom metric names."""
        response = client.get("/metrics")
        text = response.text
        assert "rag_requests_total" in text
        assert "rag_request_duration_seconds" in text
        assert "rag_faithfulness_score" in text


class TestQueryEndpoint:

    def test_query_requires_body(self, client):
        """POST /query with no body should return 422."""
        response = client.post("/query")
        assert response.status_code == 422

    def test_query_requires_query_field(self, client):
        """POST /query with empty JSON should return 422."""
        response = client.post("/query", json={})
        assert response.status_code == 422

    def test_query_validates_min_length(self, client):
        """Query string must be at least 1 character."""
        response = client.post("/query", json={"query": ""})
        assert response.status_code == 422

    def test_query_validates_max_length(self, client):
        """Query string cannot exceed 1000 characters."""
        response = client.post("/query", json={"query": "x" * 1001})
        assert response.status_code == 422

    def test_query_success(self, client, mock_rag_pipeline):
        """Valid query should return 200 with structured response."""
        response = client.post("/query", json={"query": "What is Python?"})
        assert response.status_code == 200

    def test_query_response_structure(self, client, mock_rag_pipeline):
        """Query response must have all required fields."""
        response = client.post("/query", json={"query": "Test query"})
        data = response.json()

        assert "request_id" in data
        assert "query" in data
        assert "answer" in data
        assert "faithfulness_score" in data
        assert "passed_faithfulness" in data
        assert "chunks_used" in data
        assert "retries_used" in data
        assert "latency_ms" in data
        assert "metadata" in data

    def test_query_returns_mocked_answer(self, client, mock_rag_pipeline):
        """Query should return the mocked RAG pipeline answer."""
        response = client.post("/query", json={"query": "Any question"})
        data = response.json()
        assert "test answer" in data["answer"].lower()

    def test_query_chunks_have_correct_structure(self, client, mock_rag_pipeline):
        """Each chunk in the response must have required fields."""
        response = client.post("/query", json={"query": "Any question"})
        data = response.json()
        chunks = data["chunks_used"]

        assert len(chunks) > 0
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "text" in chunk
            assert "score" in chunk
            assert "metadata" in chunk

    def test_query_with_optional_params(self, client, mock_rag_pipeline):
        """Query accepts optional top_k, user_id, clearance_level."""
        response = client.post(
            "/query",
            json={
                "query": "Test",
                "top_k": 10,
                "user_id": "user_123",
                "clearance_level": "internal",
            },
        )
        assert response.status_code == 200


class TestRootEndpoint:

    def test_root_returns_200(self, client):
        """Root endpoint should return 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_json(self, client):
        """Root endpoint should return JSON with message."""
        response = client.get("/")
        data = response.json()
        assert "message" in data
        assert "docs" in data


class TestRequestID:

    def test_response_includes_request_id_header(self, client):
        """Every response should include X-Request-ID header."""
        response = client.get("/health")
        assert "x-request-id" in response.headers

    def test_request_id_is_unique(self, client):
        """Each request should get a unique request ID."""
        r1 = client.get("/health")
        r2 = client.get("/health")
        assert r1.headers["x-request-id"] != r2.headers["x-request-id"]