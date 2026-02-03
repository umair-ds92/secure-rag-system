"""
metrics.py
==========
Quantitative evaluation harness for the RAG pipeline.

Provides:
    • Precision / Recall / F1 at the token level (response vs gold answer).
    • Faithfulness aggregates over a batch of (query, gold, response, chunks).
    • Latency statistics (p50, p90, p95, p99) from a list of latency values.
    • ``evaluate_batch()`` — single function that takes a list of test cases,
      runs faithfulness scoring on each, and returns a structured summary.

All functions are stateless and importable individually — no class required.
"""

from __future__ import annotations

import re
import logging
import statistics
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token-level precision / recall / F1
# ---------------------------------------------------------------------------


def _tokenise(text: str) -> list[str]:
    """Lowercase word tokens (no punctuation)."""
    return re.findall(r"[a-z0-9]+", text.lower())


def token_precision(prediction: str, reference: str) -> float:
    """Fraction of tokens in *prediction* that also appear in *reference*."""
    pred_tokens = _tokenise(prediction)
    ref_tokens  = set(_tokenise(reference))
    if not pred_tokens:
        return 0.0
    hits = sum(1 for t in pred_tokens if t in ref_tokens)
    return round(hits / len(pred_tokens), 4)


def token_recall(prediction: str, reference: str) -> float:
    """Fraction of tokens in *reference* that also appear in *prediction*."""
    pred_set   = set(_tokenise(prediction))
    ref_tokens = _tokenise(reference)
    if not ref_tokens:
        return 0.0
    hits = sum(1 for t in ref_tokens if t in pred_set)
    return round(hits / len(ref_tokens), 4)


def token_f1(prediction: str, reference: str) -> float:
    """Harmonic mean of token precision and recall."""
    p = token_precision(prediction, reference)
    r = token_recall(prediction, reference)
    if p + r == 0:
        return 0.0
    return round(2 * p * r / (p + r), 4)


# ---------------------------------------------------------------------------
# Latency statistics
# ---------------------------------------------------------------------------


def latency_percentiles(latencies_ms: list[float]) -> dict[str, float]:
    """
    Given a list of latency values (ms), return p50 / p90 / p95 / p99 / mean.
    Returns all zeros when the list is empty.
    """
    if not latencies_ms:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}

    sorted_lat = sorted(latencies_ms)
    n          = len(sorted_lat)

    def _percentile(pct: float) -> float:
        idx = min(int(pct / 100.0 * (n-1)), n - 1)
        return round(sorted_lat[idx], 2)

    return {
        "p50":  _percentile(50),
        "p90":  _percentile(90),
        "p95":  _percentile(95),
        "p99":  _percentile(99),
        "mean": round(statistics.mean(sorted_lat), 2),
    }


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


def evaluate_batch(
    test_cases: list[dict],
    faithfulness_scorer: object,
) -> dict:
    """
    Run a full evaluation over a list of test cases.

    Each element of *test_cases* must be a dict:
        {
            "query":          str,    – the user question
            "gold_answer":    str,    – ground-truth answer
            "llm_response":   str,    – what the LLM returned
            "context_chunks": list,   – retrieved chunks
        }

    Parameters
    ----------
    faithfulness_scorer : FaithfulnessScorer
        Already-instantiated scorer (avoids reloading the model per call).

    Returns
    -------
    dict:
        {
            "n": int,
            "per_sample": [ { …per-case metrics… }, … ],
            "aggregate": {
                "avg_precision":            float,
                "avg_recall":               float,
                "avg_f1":                   float,
                "avg_faithfulness":         float,
                "faithfulness_pass_rate":   float,
            }
        }
    """
    FAITHFULNESS_THRESHOLD = 0.70

    per_sample : list[dict]  = []
    precisions : list[float] = []
    recalls    : list[float] = []
    f1s        : list[float] = []
    faiths     : list[float] = []

    for i, tc in enumerate(test_cases):
        query    = tc["query"]
        gold     = tc["gold_answer"]
        response = tc["llm_response"]
        chunks   = tc["context_chunks"]

        # --- token metrics ---
        p = token_precision(response, gold)
        r = token_recall(response, gold)
        f = token_f1(response, gold)

        # --- faithfulness ---
        faith = faithfulness_scorer.score(response, chunks)

        per_sample.append({
            "idx":          i,
            "query":        query[:80],
            "precision":    p,
            "recall":       r,
            "f1":           f,
            "faithfulness": faith,
            "faith_passed": faith >= FAITHFULNESS_THRESHOLD,
        })

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        faiths.append(faith)

        logger.debug("eval[%d] P=%.2f R=%.2f F1=%.2f faith=%.2f", i, p, r, f, faith)

    # --- aggregates ---
    n          = len(test_cases)
    pass_count = sum(1 for fv in faiths if fv >= FAITHFULNESS_THRESHOLD)

    def _avg(lst: list[float]) -> float:
        return round(statistics.mean(lst), 4) if lst else 0.0

    return {
        "n": n,
        "per_sample": per_sample,
        "aggregate": {
            "avg_precision":          _avg(precisions),
            "avg_recall":             _avg(recalls),
            "avg_f1":                 _avg(f1s),
            "avg_faithfulness":       _avg(faiths),
            "faithfulness_pass_rate": round(pass_count / max(n, 1), 4),
        },
    }