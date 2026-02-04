# Secure RAG System - API Documentation

## Base URL

- **Local Development:** `http://localhost:8000`
- **Production (AWS):** `http://<ALB-DNS-NAME>`
- **Interactive Docs:** `http://<BASE_URL>/docs`
- **OpenAPI Schema:** `http://<BASE_URL>/openapi.json`

---

## Endpoints

### 1. Health Check

Check if the service is healthy and all dependencies are operational.

**Endpoint:** `GET /health`

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "chromadb_connected": true,
  "model_loaded": true
}
```

**Status Codes:**
- `200 OK` — Service is healthy
- `503 Service Unavailable` — Service is unhealthy

**Example:**

```bash
curl http://localhost:8000/health
```

---

### 2. Query RAG System

Submit a question and receive an answer with faithfulness scoring.

**Endpoint:** `POST /query`

**Request Body:**

```json
{
  "query": "What is Python used for?",
  "top_k": 5,
  "user_id": "user_123",
  "clearance_level": "public"
}
```

**Parameters:**

| Field            | Type   | Required | Default  | Description                              |
|------------------|--------|----------|----------|------------------------------------------|
| `query`          | string | Yes      | -        | User question (1-1000 chars)             |
| `top_k`          | int    | No       | 5        | Number of chunks to retrieve (1-20)      |
| `user_id`        | string | No       | null     | User identifier                          |
| `clearance_level`| string | No       | "public" | RBAC level                               |

**Response:**

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "query": "What is Python used for?",
  "answer": "Python is used for web development, data science...",
  "faithfulness_score": 0.92,
  "passed_faithfulness": true,
  "chunks_used": [
    {
      "chunk_id": "chunk_0",
      "text": "Python is a versatile language...",
      "score": 0.85,
      "metadata": {"source": "python_guide.txt"}
    }
  ],
  "retries_used": 0,
  "latency_ms": 234.56,
  "metadata": {"timestamp": "2026-02-03T12:00:00Z"}
}
```

**Status Codes:**
- `200 OK` — Success
- `400 Bad Request` — Prompt injection detected
- `422 Unprocessable Entity` — Validation error
- `500 Internal Server Error` — Pipeline failure

**Example:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Python?"}'
```

---

### 3. Prometheus Metrics

Retrieve system metrics in Prometheus format.

**Endpoint:** `GET /metrics`

**Response:** Plain text (Prometheus format)

```
rag_requests_total{endpoint="/query",status="success"} 1234.0
rag_request_duration_seconds_bucket{endpoint="/query",le="0.5"} 456.0
rag_faithfulness_score_bucket{le="0.7"} 45.0
rag_active_requests 3.0
```

**Example:**

```bash
curl http://localhost:8000/metrics
```

---

### 4. Root

Returns API information.

**Endpoint:** `GET /`

**Response:**

```json
{
  "message": "Secure RAG API v1.0.0",
  "docs": "/docs"
}
```

---

## Error Responses

```json
{
  "detail": "Error message",
  "request_id": "550e8400-..."
}
```

### Common Error Codes

| Code | Meaning                 | Common Causes                    |
|------|-------------------------|----------------------------------|
| 400  | Bad Request             | Prompt injection detected        |
| 422  | Unprocessable Entity    | Missing required field           |
| 500  | Internal Server Error   | ChromaDB connection lost         |
| 503  | Service Unavailable     | Health check failed              |

---

## Environment Variables

| Variable                  | Default       | Description                      |
|---------------------------|---------------|----------------------------------|
| `CHROMA_HOST`             | localhost     | ChromaDB host                    |
| `CHROMA_PORT`             | 8000          | ChromaDB port                    |
| `FAITHFULNESS_THRESHOLD`  | 0.70          | Min faithfulness score           |
| `MAX_RETRIES`             | 2             | LLM re-prompt attempts           |
| `OPENAI_API_KEY`          | (required)    | OpenAI API key                   |
| `LOG_LEVEL`               | INFO          | Logging level                    |

---

## Testing

### Local

```bash
docker compose up -d
sleep 20
curl http://localhost:8000/health
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Python?"}'
```

### Production

```bash
ALB_URL=$(aws cloudformation describe-stacks \
  --stack-name production-rag-stack \
  --query "Stacks[0].Outputs[?OutputKey=='LoadBalancerURL'].OutputValue" \
  --output text)

curl $ALB_URL/health
```

---

## OpenAPI Documentation

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
- **OpenAPI JSON:** `http://localhost:8000/openapi.json`

---

For deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).