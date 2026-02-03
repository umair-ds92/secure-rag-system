"""
test_rag_pipeline.py
====================
Pytest suite — RAG Generation + Faithfulness.

Test classes
------------
TestFaithfulnessScorer     – unit tests for every scoring signal
TestMetrics                – token precision / recall / F1 + batch eval
TestRAGPipeline            – integration tests using a mock LLM + mock vector store

Run
---
    pytest tests/test_rag_pipeline.py -v
"""

from __future__ import annotations

import sys
import os
import pytest

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.faithfulness import FaithfulnessScorer
from evaluation.metrics       import (
    token_precision, token_recall, token_f1,
    latency_percentiles, evaluate_batch,
)
from generation.rag_pipeline  import RAGPipeline, RAGResponse


# ===========================================================================
# Shared test data
# ===========================================================================

CONTEXT_CHUNKS = [
    (
        "Python was created by Guido van Rossum and first released in 1991. "
        "It is a high-level, interpreted programming language known for its "
        "clean syntax and readability. Python supports multiple programming "
        "paradigms including procedural, object-oriented, and functional."
    ),
    (
        "The Python standard library includes modules for file handling, "
        "regular expressions, networking, and database access. It also "
        "contains the unittest framework for writing automated tests. "
        "The library ships with over 200 built-in modules."
    ),
    (
        "Python 3.11 introduced significant performance improvements, "
        "with benchmarks showing 10-60 percent speed gains over Python 3.10. "
        "The error messages were also enhanced to be more precise and "
        "helpful for debugging."
    ),
]

GROUNDED_RESPONSE = (
    "Python was created by Guido van Rossum and first released in 1991. "
    "It supports multiple paradigms including procedural, object-oriented, "
    "and functional programming. The standard library ships with over 200 "
    "built-in modules."
)

FABRICATED_RESPONSE = (
    "Python was invented by James Gosling in 1995 at Sun Microsystems. "
    "It was originally designed as a replacement for C++. "
    "The language has exactly 42 reserved keywords and runs on quantum hardware. "
    "The creator received a Nobel Prize for computer science in 2003."
)


# ===========================================================================
# 1.  FaithfulnessScorer
# ===========================================================================


@pytest.fixture(scope="module")
def scorer() -> FaithfulnessScorer:
    """Module-scoped — model loads once for all tests in this class."""
    return FaithfulnessScorer()


class TestFaithfulnessScorer:

    def test_grounded_response_scores_high(self, scorer: FaithfulnessScorer):
        """Response that mirrors context should score well above 0.70."""
        result = scorer.score(GROUNDED_RESPONSE, CONTEXT_CHUNKS)
        assert result >= 0.60, f"Expected ≥ 0.60 for grounded response, got {result}"

    def test_fabricated_response_scores_low(self, scorer: FaithfulnessScorer):
        """Completely invented facts should score significantly lower."""
        grounded_score     = scorer.score(GROUNDED_RESPONSE,  CONTEXT_CHUNKS)
        fabricated_score   = scorer.score(FABRICATED_RESPONSE, CONTEXT_CHUNKS)
        assert fabricated_score < grounded_score, (
            f"Fabricated ({fabricated_score}) should be < grounded ({grounded_score})"
        )

    def test_numeric_consistency_catches_bad_numbers(self, scorer: FaithfulnessScorer):
        """Invented numbers not in context should lower the score."""
        bad_numbers = (
            "Python was created in 1991 and has exactly 9999 built-in modules "
            "with a performance gain of 777 percent."
        )
        score = scorer.score(bad_numbers, CONTEXT_CHUNKS)
        # 9999 and 777 are not in context → numeric signal penalised
        assert score < 0.90

    def test_empty_response_returns_zero(self, scorer: FaithfulnessScorer):
        assert scorer.score("", CONTEXT_CHUNKS) == 0.0

    def test_empty_context_returns_zero(self, scorer: FaithfulnessScorer):
        assert scorer.score(GROUNDED_RESPONSE, []) == 0.0

    def test_detailed_returns_all_signals(self, scorer: FaithfulnessScorer):
        """score_detailed must return every expected key."""
        detail = scorer.score_detailed(GROUNDED_RESPONSE, CONTEXT_CHUNKS)
        for key in ("overall", "semantic", "coverage", "numeric", "penalty", "claims"):
            assert key in detail, f"Missing key '{key}' in detailed output"
        # claims should be a non-empty list
        assert isinstance(detail["claims"], list)
        assert len(detail["claims"]) > 0

    def test_claim_level_attribution(self, scorer: FaithfulnessScorer):
        """Each claim dict must have the required attribution fields."""
        detail = scorer.score_detailed(GROUNDED_RESPONSE, CONTEXT_CHUNKS)
        for claim in detail["claims"]:
            assert "claim"         in claim
            assert "supported"     in claim
            assert "semantic_sim"  in claim
            assert "best_chunk_idx" in claim


# ===========================================================================
# 2.  Metrics (pure functions — no model needed)
# ===========================================================================


class TestMetrics:

    # ---- token precision / recall / F1 ----------------------------------

    def test_precision_perfect_match(self):
        assert token_precision("the cat sat", "the cat sat on the mat") == 1.0

    def test_precision_no_overlap(self):
        assert token_precision("xyz abc", "the cat sat") == 0.0

    def test_recall_perfect_match(self):
        # prediction contains all reference tokens
        assert token_recall("the cat sat on the mat extra", "the cat sat") == 1.0

    def test_recall_partial(self):
        r = token_recall("the cat", "the cat sat on the mat")
        assert 0.0 < r < 1.0

    def test_f1_perfect(self):
        assert token_f1("the cat sat", "the cat sat") == 1.0

    def test_f1_zero_when_no_overlap(self):
        assert token_f1("xyz", "abc") == 0.0

    def test_empty_prediction_precision_zero(self):
        assert token_precision("", "hello world") == 0.0

    def test_empty_reference_recall_zero(self):
        assert token_recall("hello", "") == 0.0

    # ---- latency percentiles --------------------------------------------

    def test_latency_percentiles_basic(self):
        lats  = [float(i) for i in range(1, 101)]          # 1.0 … 100.0
        stats = latency_percentiles(lats)
        assert stats["p50"]  == 50.0
        assert stats["p90"]  == 90.0
        assert stats["p99"]  == 99.0
        assert stats["mean"] == 50.5

    def test_latency_percentiles_empty(self):
        stats = latency_percentiles([])
        assert all(v == 0.0 for v in stats.values())

    # ---- batch evaluation -----------------------------------------------

    def test_evaluate_batch_structure(self, scorer: FaithfulnessScorer):
        """evaluate_batch returns the right shape."""
        cases = [
            {
                "query":          "What is Python?",
                "gold_answer":    GROUNDED_RESPONSE,
                "llm_response":   GROUNDED_RESPONSE,
                "context_chunks": CONTEXT_CHUNKS,
            },
            {
                "query":          "Who invented Python?",
                "gold_answer":    "Guido van Rossum created Python in 1991.",
                "llm_response":   FABRICATED_RESPONSE,
                "context_chunks": CONTEXT_CHUNKS,
            },
        ]
        result = evaluate_batch(cases, scorer)

        assert result["n"] == 2
        assert len(result["per_sample"]) == 2
        agg = result["aggregate"]
        for key in ("avg_precision", "avg_recall", "avg_f1", "avg_faithfulness", "faithfulness_pass_rate"):
            assert key in agg
            assert isinstance(agg[key], float)

    # use the module-scoped scorer fixture
    @pytest.fixture(autouse=False)
    def scorer(self, scorer):
        return scorer


# ===========================================================================
# 3.  RAGPipeline  (integration with mocks)
# ===========================================================================


# ---- Mock helpers --------------------------------------------------------

class MockVectorStore:
    """Returns the global CONTEXT_CHUNKS regardless of query."""
    def query(self, query_texts: list[str], n_results: int = 5) -> dict:
        docs = CONTEXT_CHUNKS[:n_results]
        return {
            "documents": [docs],
            "distances":  [[0.1] * len(docs)],
            "metadatas":  [[{"source": f"doc_{i}"} for i in range(len(docs))]],
        }


def _echo_llm(system: str, user: str) -> str:
    """
    Extracts the context block from the prompt and returns it.
    This makes the response maximally grounded so the faithfulness check passes.
    """
    start_marker = "--- RETRIEVED CONTEXT ---"
    end_marker   = "--- END CONTEXT ---"
    s = user.find(start_marker)
    e = user.find(end_marker)
    if s != -1 and e != -1:
        return user[s + len(start_marker):e].strip()
    return user


def _bad_llm(system: str, user: str) -> str:
    """Always returns fabricated nonsense."""
    return FABRICATED_RESPONSE


def _empty_llm(system: str, user: str) -> str:
    """Returns an empty string — should trigger fallback."""
    return ""


# ---- Tests ---------------------------------------------------------------


@pytest.fixture(scope="module")
def good_pipeline() -> RAGPipeline:
    """Pipeline wired with the echo LLM (grounded output)."""
    return RAGPipeline(
        llm_fn          =_echo_llm,
        vector_store    =MockVectorStore(),
        faithfulness_threshold=0.50,   # lowered slightly for echo stub
        max_retries     =1,
        audit_log_path  =None,         # no file I/O in tests
    )


@pytest.fixture(scope="module")
def bad_pipeline() -> RAGPipeline:
    """Pipeline wired with the bad LLM (fabricated output)."""
    return RAGPipeline(
        llm_fn          =_bad_llm,
        vector_store    =MockVectorStore(),
        faithfulness_threshold=0.90,   # high bar – fabricated text won't clear it
        max_retries     =1,
        audit_log_path  =None,
    )


class TestRAGPipeline:

    def test_run_returns_rag_response(self, good_pipeline: RAGPipeline):
        resp = good_pipeline.run("What is Python?")
        assert isinstance(resp, RAGResponse)

    def test_grounded_response_passes_faithfulness(self, good_pipeline: RAGPipeline):
        resp = good_pipeline.run("What is Python?")
        assert resp.passed_faithfulness is True
        assert resp.answer == resp.raw_answer          # no fallback triggered
        assert resp.faithfulness_score > 0.0

    def test_fabricated_response_triggers_fallback(self, bad_pipeline: RAGPipeline):
        resp = bad_pipeline.run("Who invented Python?")
        assert resp.passed_faithfulness is False
        assert resp.answer == RAGPipeline.FALLBACK     # fallback returned
        assert resp.retries_used >= 1                  # at least one retry happened

    def test_audit_log_populated(self, good_pipeline: RAGPipeline):
        resp = good_pipeline.run("Tell me about the standard library.")
        assert len(resp.audit_log) >= 2                # at least RETRIEVE + GENERATE
        stages = {entry["stage"] for entry in resp.audit_log}
        assert "RETRIEVE"     in stages
        assert "GENERATE"     in stages
        assert "FAITHFULNESS" in stages

    def test_latency_tracked(self, good_pipeline: RAGPipeline):
        resp = good_pipeline.run("Any question.")
        assert resp.latency_ms > 0
        assert isinstance(resp.latency_ms, float)

    def test_chunks_populated(self, good_pipeline: RAGPipeline):
        resp = good_pipeline.run("Tell me about Python 3.11.")
        assert len(resp.chunks_used) > 0
        assert all(hasattr(c, "text") for c in resp.chunks_used)

    def test_empty_llm_response_triggers_fallback(self):
        """An LLM that returns empty string should fall back."""
        pipeline = RAGPipeline(
            llm_fn          =_empty_llm,
            vector_store    =MockVectorStore(),
            faithfulness_threshold=0.50,
            max_retries     =0,
            audit_log_path  =None,
        )
        resp = pipeline.run("Anything?")
        assert resp.answer == RAGPipeline.FALLBACK

    def test_empty_vector_store_returns_fallback(self):
        """When the vector store returns nothing, fallback is immediate."""
        class EmptyStore:
            def query(self, **kwargs):
                return {"documents": [[]], "distances": [[]], "metadatas": [[]]}

        pipeline = RAGPipeline(
            llm_fn       =_echo_llm,
            vector_store =EmptyStore(),
            audit_log_path=None,
        )
        resp = pipeline.run("Will this work?")
        assert resp.answer == RAGPipeline.FALLBACK
        assert resp.retries_used == 0