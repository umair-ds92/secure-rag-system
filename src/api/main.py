"""
main.py
=======
Production FastAPI application for the Secure RAG system.

Endpoints:
    POST   /query    → RAG question answering
    GET    /health   → health check (Docker + ECS probes)
    GET    /metrics  → Prometheus-compatible metrics

Middleware:
    - CORS (configurable origins)
    - Request ID injection
    - Structured logging
    - Error handling
"""

from __future__ import annotations

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import chromadb

# Local imports
from api.models import QueryRequest, QueryResponse, HealthResponse, RetrievedChunk
from generation.rag_pipeline import RAGPipeline
from retrieval.vector_store import VectorStoreManager
from security.input_sanitizer import InputSanitizer
from security.prompt_guard import PromptGuard

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "secure_documents")
FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.70"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
AUDIT_LOG_PATH = os.getenv("AUDIT_LOG_PATH", "/app/logs/audit.jsonl")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

REQUEST_COUNT = Counter(
    "rag_requests_total",
    "Total number of RAG requests",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "rag_request_duration_seconds",
    "RAG request latency in seconds",
    ["endpoint"],
)
FAITHFULNESS_SCORE = Histogram(
    "rag_faithfulness_score",
    "Distribution of faithfulness scores",
    buckets=[0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0],
)
ACTIVE_REQUESTS = Gauge(
    "rag_active_requests",
    "Number of active RAG requests",
)

# ---------------------------------------------------------------------------
# Global state (initialized at startup)
# ---------------------------------------------------------------------------

vector_store: Optional[VectorStoreManager] = None
rag_pipeline: Optional[RAGPipeline] = None
input_sanitizer: Optional[InputSanitizer] = None
prompt_guard: Optional[PromptGuard] = None

# ---------------------------------------------------------------------------
# Lifespan context manager (startup/shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, clean up on shutdown."""
    global vector_store, rag_pipeline, input_sanitizer, prompt_guard

    logger.info("=== Application startup ===")

    # ---- 1. Initialize Vector Store ------------------------------------
    try:
        vector_store = VectorStoreManager(
            collection_name=COLLECTION_NAME,
            persist_directory="/app/data/chroma_db",
        )
        logger.info(f"✓ Vector store initialized: {COLLECTION_NAME}")
    except Exception as exc:
        logger.error(f"✗ ChromaDB connection failed: {exc}")
        raise

    # ---- 2. Initialize security layer ----------------------------------
    input_sanitizer = InputSanitizer()
    prompt_guard = PromptGuard()
    logger.info("✓ Security layer initialized")

    # ---- 3. Initialize RAG pipeline ------------------------------------
    def _llm_wrapper(system_prompt: str, user_prompt: str) -> str:
        """
        Placeholder LLM wrapper.  In production this calls OpenAI/Anthropic/etc.
        For now it returns a static string so the /health check passes.
        """
        if not OPENAI_API_KEY:
            return (
                "LLM not configured. Set OPENAI_API_KEY environment variable. "
                "This is a placeholder response for health checks."
            )
        # Real implementation would be:
        # from openai import OpenAI
        # client = OpenAI(api_key=OPENAI_API_KEY)
        # response = client.chat.completions.create(...)
        # return response.choices[0].message.content
        return "Placeholder LLM response. Integrate OpenAI/Anthropic client here."

    rag_pipeline = RAGPipeline(
        llm_fn=_llm_wrapper,
        vector_store=vector_store.collection,
        faithfulness_threshold=FAITHFULNESS_THRESHOLD,
        max_retries=MAX_RETRIES,
        audit_log_path=AUDIT_LOG_PATH,
    )
    logger.info("✓ RAG pipeline ready")

    logger.info("=== Application ready ===")

    yield  # application runs

    logger.info("=== Application shutdown ===")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Secure RAG API",
    description="Production RAG system with hallucination mitigation and security controls",
    version="1.0.0",
    lifespan=lifespan,
)

# ---- CORS middleware ---------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware: request ID + logging
# ---------------------------------------------------------------------------


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Inject request ID into every request for tracing."""
    import uuid
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all for unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    REQUEST_COUNT.labels(endpoint=request.url.path, status="error").inc()
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "request_id": getattr(request.state, "request_id", "unknown")},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.  Used by:
      - Docker HEALTHCHECK
      - ECS target group health probes
      - Kubernetes liveness/readiness probes
    """
    chromadb_ok = False
    model_ok = False

    try:
        if vector_store and vector_store.collection:
            vector_store.collection.count()  # simple operation to verify connection
            chromadb_ok = True
    except Exception as exc:
        logger.warning(f"ChromaDB health check failed: {exc}")

    try:
        if rag_pipeline and rag_pipeline._faithfulness:
            model_ok = True
    except Exception as exc:
        logger.warning(f"Model health check failed: {exc}")

    if not (chromadb_ok and model_ok):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy",
        )

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        chromadb_connected=chromadb_ok,
        model_loaded=model_ok,
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus-compatible metrics endpoint.
    Scrape this with Prometheus or CloudWatch Container Insights.
    """
    return PlainTextResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_endpoint(request: Request, body: QueryRequest):
    """
    Main RAG endpoint.  Processes a user query through the full pipeline:
        1. Input sanitization
        2. Prompt injection detection
        3. Retrieval
        4. Generation with faithfulness check
        5. Audit logging

    Returns a structured response with answer + faithfulness score + metadata.
    """
    ACTIVE_REQUESTS.inc()
    start = time.perf_counter()

    try:
        # ---- 1. Security: sanitize input -------------------------------
        sanitization_result = input_sanitizer.sanitize(body.query)
        sanitized_query = sanitization_result.sanitized_text

        # ---- 2. Security: prompt injection guard -----------------------
        if not prompt_guard.is_safe_query(sanitized_query):
            REQUEST_COUNT.labels(endpoint="/query", status="blocked").inc()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Prompt injection detected",
            )

        # ---- 3. Run RAG pipeline ---------------------------------------
        result = rag_pipeline.run(sanitized_query)

        # ---- 4. Convert to API response schema -------------------------
        response = QueryResponse(
            request_id=result.request_id,
            query=result.query,
            answer=result.answer,
            faithfulness_score=result.faithfulness_score,
            passed_faithfulness=result.passed_faithfulness,
            chunks_used=[
                RetrievedChunk(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    score=c.score,
                    metadata=c.metadata,
                )
                for c in result.chunks_used
            ],
            retries_used=result.retries_used,
            latency_ms=result.latency_ms,
            metadata=result.metadata,
        )

        # ---- 5. Metrics ------------------------------------------------
        REQUEST_COUNT.labels(endpoint="/query", status="success").inc()
        FAITHFULNESS_SCORE.observe(result.faithfulness_score)

        return response

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Query processing failed: {exc}", exc_info=True)
        REQUEST_COUNT.labels(endpoint="/query", status="error").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    finally:
        ACTIVE_REQUESTS.dec()
        latency = time.perf_counter() - start
        REQUEST_LATENCY.labels(endpoint="/query").observe(latency)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint — redirects to /docs."""
    return {"message": "Secure RAG API v1.0.0", "docs": "/docs"}