# ===========================================================================
# Dockerfile — secure-rag-system  (app container)
# ===========================================================================
# Stage 1 : builder   – installs Python deps into a venv
# Stage 2 : runner    – copies only the venv + app code into a slim image
#
# Build:   docker build -t secure-rag-app .
# ===========================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder  –  dependency installation
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# System deps needed to compile some wheels (cryptography, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only requirements first  –  layer caching: pip install is expensive
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: Runner  –  final production image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runner

# Minimal runtime system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd -r raguser && useradd -r -g raguser -d /app raguser

WORKDIR /app

# Copy the venv from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application source
COPY src/           /app/src/
COPY config.yaml    /app/config.yaml
COPY scripts/       /app/scripts/

# Create writable directories for logs, data, and audit files
RUN mkdir -p /app/logs /app/data/chroma_db /app/monitoring \
    && chown -R raguser:raguser /app

# Activate venv via PATH (no need to source it in every RUN)
ENV PATH="/opt/venv/bin:${PATH}"

# Pre-download the sentence-transformer model into the image so the first
# request doesn't stall.  Runs as root so it can write to the model cache.
# Falls back silently if network is unavailable at build time.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" 2>/dev/null || true

# Switch to non-root
USER raguser

# ---------------------------------------------------------------------------
# Health-check  –  used by docker-compose & ECS health probes
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=15s \
    CMD curl -sf http://localhost:8000/health || exit 1

# ---------------------------------------------------------------------------
# Expose & entrypoint
# ---------------------------------------------------------------------------
EXPOSE 8000

# Default command — uvicorn FastAPI server on port 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]