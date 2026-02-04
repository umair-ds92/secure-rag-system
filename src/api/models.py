"""
models.py
=========
Pydantic schemas for request/response validation and OpenAPI documentation.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """POST /query request body."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User question",
        examples=["What is Python used for?"],
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve",
    )
    user_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="User identifier for access control",
    )
    clearance_level: Optional[str] = Field(
        default="public",
        description="RBAC clearance level",
        examples=["public", "internal", "confidential", "restricted"],
    )

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        """Strip leading/trailing whitespace."""
        return v.strip()


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class RetrievedChunk(BaseModel):
    """A single retrieved chunk with metadata."""
    chunk_id: str
    text: str
    score: float = Field(ge=0.0, le=1.0)
    metadata: dict = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """POST /query response body."""
    request_id: str
    query: str
    answer: str
    faithfulness_score: float = Field(ge=0.0, le=1.0)
    passed_faithfulness: bool
    chunks_used: list[RetrievedChunk]
    retries_used: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    metadata: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """GET /health response."""
    status: str = Field(examples=["healthy"])
    version: str = Field(examples=["1.0.0"])
    chromadb_connected: bool
    model_loaded: bool


class MetricsResponse(BaseModel):
    """GET /metrics response (Prometheus-compatible plain text returned separately)."""
    total_requests: int = Field(ge=0)
    total_errors: int = Field(ge=0)
    avg_latency_ms: float = Field(ge=0.0)
    faithfulness_pass_rate: float = Field(ge=0.0, le=1.0)