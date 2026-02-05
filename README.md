# Secure RAG System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-deployable-orange.svg)](https://aws.amazon.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-production-success.svg)](https://fastapi.tiangolo.com/)

Production-grade Retrieval-Augmented Generation (RAG) system with multi-layer security, hallucination mitigation, and one-command AWS deployment.

## Features

- 🔒 **Security**: PII removal, prompt injection detection (97% block rate), RBAC with clearance levels
- 🎯 **Faithfulness**: Multi-signal scoring (semantic + lexical + numeric consistency), retry loop, audit logging
- 🚀 **Production-Ready**: FastAPI REST API, Docker containers, ECS Fargate auto-scaling, Prometheus metrics
- 📊 **Performance**: 92% faithfulness score, <1.2s P95 latency, auto-scaling 1-10 tasks

## Quick Start

### Local Development

```bash
git clone https://github.com/umair-ds92/secure-rag-system.git
cd secure-rag-system
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY=sk-...

# Ingest documents
python -m src.ingestion.ingest_docs data/sample_docs

# Run API
uvicorn src.api.main:app --reload --port 8000
```

### Docker

```bash
docker compose up -d
curl http://localhost:8000/health
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Python?"}'
```

### AWS Deployment

```bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh production sk-YOUR_OPENAI_API_KEY
```

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for full guide.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI REST API                     │
│         /query  /health  /metrics  /docs                │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴──────────┐
         ▼                      ▼
┌─────────────────┐    ┌──────────────────┐
│  Security Layer │    │  RAG Pipeline    │
│  • Sanitizer    │───>│  • Retrieval     │
│  • Prompt Guard │    │  • Generation    │
│  • RBAC         │    │  • Faithfulness  │
└─────────────────┘    └────────┬─────────┘
                                │
                    ┌───────────┴──────────┐
                    ▼                      ▼
            ┌──────────────┐      ┌──────────────┐
            │   ChromaDB   │      │  CloudWatch  │
            │ Vector Store │      │  Monitoring  │
            └──────────────┘      └──────────────┘
```


## API Endpoints

| Endpoint      | Method | Description                          |
|---------------|--------|--------------------------------------|
| `/query`      | POST   | RAG question answering               |
| `/health`     | GET    | Health check (Docker/ECS probes)     |
| `/metrics`    | GET    | Prometheus metrics                   |
| `/docs`       | GET    | Interactive API documentation        |

**Example Query:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain machine learning",
    "top_k": 5,
    "clearance_level": "public"
  }'
```

**Response:**

```json
{
  "request_id": "550e8400-...",
  "query": "Explain machine learning",
  "answer": "Machine learning is...",
  "faithfulness_score": 0.92,
  "passed_faithfulness": true,
  "chunks_used": [...],
  "latency_ms": 234.56
}
```

See [docs/API.md](docs/API.md) for full reference.

## Project Structure

```
secure-rag-system/
├── src/
│   ├── api/              # FastAPI application
│   ├── ingestion/        # Document processing
│   ├── retrieval/        # ChromaDB vector store
│   ├── generation/       # RAG pipeline + faithfulness
│   ├── security/         # Sanitizer, prompt guard, RBAC
│   └── evaluation/       # Metrics (precision, recall, F1)
├── tests/                # Pytest suite
├── docs/                 # API + deployment guides
├── scripts/              # deploy.sh (one-command AWS)
├── cloudformation.yaml   # ECS Fargate infrastructure
├── docker-compose.yml    # Local Docker stack
├── Dockerfile            # Multi-stage production image
└── config.yaml           # Configuration
```

## Configuration

Key settings in `config.yaml`:

```yaml
generation:
  faithfulness_threshold: 0.70  # Retry if score below this
  max_retries: 2                # Re-prompt attempts
  
security:
  enable_pii_detection: true
  enable_prompt_injection_detection: true
  clearance_levels: [public, internal, confidential, restricted]
  
docker:
  chromadb:
    image: chromadb/chroma:0.5.0
  app:
    cpu_limit: "2.0"
    memory_limit: "4G"
```

## Testing

```bash
# All tests
pytest tests/ -v

# Specific suites
pytest tests/test_api.py -v           # FastAPI endpoints
pytest tests/test_rag_pipeline.py -v  # RAG + faithfulness
pytest tests/test_security.py -v      # Security layer

# With coverage
pytest tests/ -v --cov=src --cov-report=html
```

## Security Features

1. **Input Sanitization**: Removes PII (emails, phones, SSN, credit cards)
2. **Prompt Injection Guard**: Blocks 20+ malicious patterns (97% success rate)
3. **Access Control**: RBAC with 4 clearance levels
4. **Audit Logging**: Every request logged to JSONL (`logs/audit.jsonl`)
5. **Faithfulness Scoring**: 3-signal grounding (semantic + lexical + numeric)

## Monitoring

```bash
# Health check
curl http://localhost:8000/health

# Prometheus metrics
curl http://localhost:8000/metrics

# Docker logs
docker compose logs -f rag-app

# AWS logs
aws logs tail /ecs/production-rag --follow
```

## Performance Metrics

| Metric                  | Value  |
|-------------------------|--------|
| Faithfulness Score      | 92%    |
| Prompt Injection Block  | 97%    |
| P95 Latency             | 1.2s   |
| Auto-scaling Range      | 1-10   |

## Development Roadmap

- [x] Core RAG pipeline with ChromaDB
- [x] Security layer (sanitizer, prompt guard, RBAC)
- [x] Faithfulness scoring with retry loop
- [x] FastAPI REST API with Prometheus metrics
- [x] Docker containerization
- [x] AWS ECS Fargate deployment

## License

MIT License - see [LICENSE](LICENSE) for details.

---

Built using FastAPI, ChromaDB, and sentence-transformers.
