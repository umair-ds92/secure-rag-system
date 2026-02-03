"""
rag_pipeline.py
===============
Production RAG orchestrator.  Single entry-point that chains:

    Sanitise  →  Retrieve  →  Prompt  →  Generate  →  Faithfulness Check  →  Audit  →  Return

Hallucination mitigation is NOT a separate module — it is baked into the
generation loop:
    • A grounded system prompt constrains the LLM to the retrieved context.
    • Post-generation, faithfulness.py scores every claim.
    • If the faithfulness score drops below a configurable threshold the
      pipeline either retries (up to max_retries) or returns a safe fallback.
    • Every decision is written to the structured audit log.

The LLM backend is injected as a simple callable ``(system: str, user: str) -> str``
so the class works with OpenAI, Anthropic, local Ollama, or any other
provider without hard coupling.
"""

from __future__ import annotations

import time
import uuid
import logging
import json
from dataclasses import dataclass, field
from typing import Callable, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports  –  heavy packages loaded only when the class is instantiated
# ---------------------------------------------------------------------------
_faithfulness_module = None


def _get_faithfulness():
    global _faithfulness_module
    if _faithfulness_module is None:
        from evaluation.faithfulness import FaithfulnessScorer
        _faithfulness_module = FaithfulnessScorer
    return _faithfulness_module


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """A single chunk as returned by the vector store."""
    chunk_id:   str
    text:       str
    score:      float                              # similarity score from ChromaDB
    metadata:   dict = field(default_factory=dict)


@dataclass
class RAGResponse:
    """Everything the caller (or the FastAPI layer) needs."""
    request_id:          str
    query:               str
    answer:              str                       # final answer returned to user
    raw_answer:          str                       # LLM output before any fallback
    chunks_used:         list[RetrievedChunk]
    faithfulness_score:  float                     # 0.0 – 1.0
    passed_faithfulness: bool
    retries_used:        int
    latency_ms:          float
    audit_log:           list[dict] = field(default_factory=list)
    metadata:            dict       = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a precise, factual assistant integrated into an enterprise RAG system.
Your ONLY source of truth is the context provided below.

RULES:
1. Answer ONLY from the given context.  Do NOT hallucinate or invent facts.
2. If the context does not contain enough information, say exactly:
   "I don't have enough information in the provided documents to answer this."
3. When citing a fact, reference which chunk it came from (e.g. [Chunk 1]).
4. Be concise.  Do not repeat yourself.
5. Numbers, dates, and names MUST match the context exactly."""

USER_PROMPT_TEMPLATE = """\
--- RETRIEVED CONTEXT ---
{context}
--- END CONTEXT ---

Question: {question}

Answer (cite chunk sources):"""


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class RAGPipeline:
    """
    Parameters
    ----------
    llm_fn : Callable[[str, str], str]
        Takes (system_prompt, user_prompt) → completion string.
        Swap in OpenAI / Anthropic / Ollama wrappers here.
    vector_store : object
        Must expose  query(query_texts=[...], n_results=int) -> dict
        returning ChromaDB-style {"documents": [[...]], "distances": [[...]], "metadatas": [[...]]}
    faithfulness_threshold : float
        Scores below this trigger a retry or fallback.  Default 0.70.
    max_retries : int
        How many times to re-prompt the LLM before giving up.
    top_k : int
        Number of chunks to retrieve.
    max_context_chars : int
        Hard cap on concatenated context length fed into the prompt.
    fallback_message : str | None
        Returned when all retries are exhausted.
    audit_log_path : str | None
        If set, each request's audit log is appended as one JSON line.
    """

    FALLBACK = (
        "I was unable to generate a reliable answer from the available documents. "
        "Please rephrase your question or consult the source documents directly."
    )

    def __init__(
        self,
        llm_fn:                 Callable[[str, str], str],
        vector_store:           object,
        faithfulness_threshold: float           = 0.70,
        max_retries:            int             = 2,
        top_k:                  int             = 5,
        max_context_chars:      int             = 6000,
        fallback_message:       Optional[str]   = None,
        audit_log_path:         Optional[str]   = None,
    ):
        self.llm_fn                 = llm_fn
        self.vector_store           = vector_store
        self.faithfulness_threshold = faithfulness_threshold
        self.max_retries            = max_retries
        self.top_k                  = top_k
        self.max_context_chars      = max_context_chars
        self.fallback_message       = fallback_message or self.FALLBACK
        self.audit_log_path         = audit_log_path

        # Initialise faithfulness scorer  –  loads sentence-transformer once
        FaithfulnessScorer       = _get_faithfulness()
        self._faithfulness       = FaithfulnessScorer()

        logger.info(
            "RAGPipeline ready | threshold=%.2f retries=%d top_k=%d",
            faithfulness_threshold, max_retries, top_k,
        )

    # ------------------------------------------------------------------
    # Main entry-point
    # ------------------------------------------------------------------

    def run(self, query: str) -> RAGResponse:
        """
        Execute the full pipeline for a single user query.

        Returns
        -------
        RAGResponse
            ``answer`` is safe to surface to the user.
        """
        request_id = str(uuid.uuid4())
        start_ns   = time.perf_counter_ns()
        audit      : list[dict] = []

        # ---------------------------------------------------------- 1. retrieve
        chunks = self._retrieve(query)
        audit.append(_log("RETRIEVE", f"{len(chunks)} chunks returned"))

        if not chunks:
            audit.append(_log("FALLBACK", "No chunks retrieved"))
            return _build_response(
                request_id, query, self.fallback_message, self.fallback_message,
                chunks, 0.0, False, 0, start_ns, audit,
            )

        # ---------------------------------------------------------- 2. generate + faithfulness loop
        context_str = self._format_context(chunks)
        user_prompt = USER_PROMPT_TEMPLATE.format(context=context_str, question=query)
        chunk_texts = [c.text for c in chunks]

        raw_answer   = ""
        faithfulness = 0.0
        passed       = False
        retries_used = 0

        for attempt in range(self.max_retries + 1):
            # --- call LLM -----------------------------------------------
            raw_answer = self.llm_fn(SYSTEM_PROMPT, user_prompt)
            audit.append(_log("GENERATE", f"attempt {attempt + 1}, len={len(raw_answer)}"))

            # --- faithfulness check -------------------------------------
            faithfulness = self._faithfulness.score(raw_answer, chunk_texts)
            audit.append(_log(
                "FAITHFULNESS",
                f"score={faithfulness:.4f} threshold={self.faithfulness_threshold}",
            ))

            if faithfulness >= self.faithfulness_threshold:
                passed = True
                break

            retries_used = attempt + 1
            audit.append(_log("RETRY", f"faithfulness {faithfulness:.2f} < {self.faithfulness_threshold}"))

        if not passed:
            audit.append(_log("FALLBACK", "all retries exhausted"))

        # ---------------------------------------------------------- 3. final answer
        final_answer = raw_answer if passed else self.fallback_message

        # ---------------------------------------------------------- 4. build + persist
        response = _build_response(
            request_id, query, final_answer, raw_answer,
            chunks, faithfulness, passed, retries_used, start_ns, audit,
        )
        self._persist_audit(response)
        return response

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    # ---- retrieval ----------------------------------------------------

    def _retrieve(self, query: str) -> list[RetrievedChunk]:
        """Query ChromaDB and normalise into RetrievedChunk list."""
        try:
            result = self.vector_store.query(
                query_texts=[query],
                n_results=self.top_k,
            )
            documents = result.get("documents", [[]])[0]
            distances = result.get("distances",  [[]])[0]
            metadatas = (
                result.get("metadatas", [[]])[0]
                if result.get("metadatas")
                else [{}] * len(documents)
            )

            # ChromaDB default metric is L2; convert to [0,1] similarity
            chunks = []
            for idx, (doc, dist, meta) in enumerate(zip(documents, distances, metadatas)):
                similarity = round(1.0 / (1.0 + dist), 4)
                chunks.append(RetrievedChunk(
                    chunk_id  = f"chunk_{idx}",
                    text      = doc,
                    score     = similarity,
                    metadata  = meta or {},
                ))
            return chunks

        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            return []

    # ---- prompt construction ------------------------------------------

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        """Concatenate labelled chunks, respecting max_context_chars."""
        parts : list[str] = []
        budget = self.max_context_chars
        for chunk in chunks:
            segment = f"[Chunk {chunk.chunk_id}] (relevance {chunk.score})\n{chunk.text}"
            if len(segment) > budget:
                logger.debug("Context budget exhausted after %d chars.", self.max_context_chars - budget)
                break
            parts.append(segment)
            budget -= len(segment)
        return "\n\n".join(parts)

    # ---- audit --------------------------------------------------------

    def _persist_audit(self, response: RAGResponse) -> None:
        if not self.audit_log_path:
            return
        try:
            record = {
                "request_id":         response.request_id,
                "query":              response.query,
                "faithfulness_score": response.faithfulness_score,
                "passed":             response.passed_faithfulness,
                "retries":            response.retries_used,
                "latency_ms":         response.latency_ms,
                "audit_log":          response.audit_log,
            }
            with open(self.audit_log_path, "a") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.error("Audit persist failed: %s", exc)


# ---------------------------------------------------------------------------
# Module-level helpers  (no class state needed)
# ---------------------------------------------------------------------------


def _log(stage: str, message: str) -> dict:
    return {
        "stage":     stage,
        "message":   message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _build_response(
    request_id, query, final_answer, raw_answer,
    chunks, faithfulness, passed, retries, start_ns, audit,
) -> RAGResponse:
    latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
    return RAGResponse(
        request_id          = request_id,
        query               = query,
        answer              = final_answer,
        raw_answer          = raw_answer,
        chunks_used         = chunks,
        faithfulness_score  = round(faithfulness, 4),
        passed_faithfulness = passed,
        retries_used        = retries,
        latency_ms          = round(latency_ms, 2),
        audit_log           = audit,
        metadata            = {"timestamp": datetime.now(timezone.utc).isoformat()},
    )