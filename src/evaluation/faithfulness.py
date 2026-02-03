"""
faithfulness.py
===============
Faithfulness scorer.  Answers the question:

    "How much of what the LLM said is actually supported by the retrieved context?"

Three independent signals are computed and combined into a single [0, 1] score:

    1. Semantic Similarity  (weight 0.40)
           Cosine similarity between the response embedding and the
           best-matching chunk embedding.  Uses all-MiniLM-L6-v2.

    2. Claim Coverage       (weight 0.40)
           The response is split into claims (declarative sentences).
           Each claim is scored against every chunk.  The fraction of
           claims that are supported (sim ≥ claim_threshold OR token
           overlap ≥ lexical_threshold) becomes this signal.

    3. Numeric Consistency  (weight 0.20)
           Every number that appears in the response must also appear in
           the context.  If the response contains no numbers this signal
           is 1.0 (no numeric claims to contradict).

Penalty: if the response contains hedging language AND the semantic
similarity is weak, a small penalty (≤ 0.10) is subtracted.  This
catches the pattern where the LLM "guesses confidently" while actually
having thin evidence.

The model is loaded once and cached for the lifetime of the scorer
instance — safe to keep as a module-level singleton in long-running
services (FastAPI workers, etc.).
"""

from __future__ import annotations

import re
import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL       = "all-MiniLM-L6-v2"
CLAIM_SEM_THRESHOLD = 0.55      # per-claim semantic threshold
CLAIM_LEX_THRESHOLD = 0.35      # per-claim lexical fallback threshold
NUMBER_RE           = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b")
SENTENCE_RE         = re.compile(r"(?<=[.!?])\s+")
HEDGE_WORDS         = {
    "perhaps", "maybe", "possibly", "might", "could", "probably",
    "likely", "seems", "appears", "suggests", "uncertain", "unclear",
    "not sure", "unsure",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tokenise(text: str) -> set[str]:
    """Lowercase word-token set."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _extract_numbers(text: str) -> set[str]:
    """Normalised numeric strings (commas removed)."""
    return {m.replace(",", "") for m in NUMBER_RE.findall(text)}


def _split_claims(text: str) -> list[str]:
    """
    Split into declarative sentences.  Drop questions and fragments < 4 words.
    """
    raw = SENTENCE_RE.split(text.strip())
    return [
        s.strip() for s in raw
        if s.strip() and not s.strip().endswith("?") and len(s.split()) >= 4
    ]


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class FaithfulnessScorer:
    """
    Parameters
    ----------
    model_name : str
        Sentence-transformer model identifier.
    weights : dict | None
        Must contain keys ``semantic``, ``coverage``, ``numeric`` and sum to 1.0.
    hedge_max_penalty : float
        Maximum penalty subtracted when hedging + weak evidence detected.
    """

    DEFAULT_WEIGHTS = {
        "semantic":  0.40,
        "coverage":  0.40,
        "numeric":   0.20,
    }

    def __init__(
        self,
        model_name:        str                          = DEFAULT_MODEL,
        weights:           Optional[dict[str, float]]  = None,
        hedge_max_penalty: float                       = 0.10,
    ):
        self.weights           = weights or self.DEFAULT_WEIGHTS
        self.hedge_max_penalty = hedge_max_penalty

        # ---- validate weights ------------------------------------------
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

        # ---- load model ------------------------------------------------
        logger.info("FaithfulnessScorer: loading '%s' …", model_name)
        self._model = SentenceTransformer(model_name)
        logger.info("FaithfulnessScorer: model ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, response: str, context_chunks: list[str]) -> float:
        """
        Return a single faithfulness score in [0.0, 1.0].

        Parameters
        ----------
        response       : str   – the LLM-generated answer.
        context_chunks : list  – the chunks that were fed into the prompt.

        Returns
        -------
        float
            0.0 = completely unfaithful / fabricated.
            1.0 = fully grounded in the context.
        """
        if not response or not context_chunks:
            logger.warning("score() called with empty input – returning 0.0")
            return 0.0

        combined = "\n".join(context_chunks)

        # --- signal 1: semantic similarity ------------------------------
        sem   = self._semantic_similarity(response, context_chunks)

        # --- signal 2: claim coverage ----------------------------------
        cov   = self._claim_coverage(response, context_chunks, combined)

        # --- signal 3: numeric consistency ------------------------------
        num   = self._numeric_consistency(response, combined)

        # --- weighted sum ----------------------------------------------
        raw   = (
            self.weights["semantic"] * sem +
            self.weights["coverage"] * cov +
            self.weights["numeric"]  * num
        )

        # --- hedge penalty ---------------------------------------------
        penalty = self._hedge_penalty(response, sem)

        final = max(0.0, min(1.0, raw - penalty))
        logger.debug(
            "faithfulness | sem=%.3f cov=%.3f num=%.3f penalty=%.3f → %.4f",
            sem, cov, num, penalty, final,
        )
        return round(final, 4)

    # ------------------------------------------------------------------
    # Detailed scoring  (useful for evaluation dashboards)
    # ------------------------------------------------------------------

    def score_detailed(self, response: str, context_chunks: list[str]) -> dict:
        """
        Same as score() but returns every sub-signal for logging / dashboards.
        """
        if not response or not context_chunks:
            return {"overall": 0.0, "semantic": 0.0, "coverage": 0.0,
                    "numeric": 0.0, "penalty": 0.0, "claims": []}

        combined = "\n".join(context_chunks)

        sem     = self._semantic_similarity(response, context_chunks)
        cov     = self._claim_coverage(response, context_chunks, combined)
        num     = self._numeric_consistency(response, combined)
        penalty = self._hedge_penalty(response, sem)

        raw     = (
            self.weights["semantic"] * sem +
            self.weights["coverage"] * cov +
            self.weights["numeric"]  * num
        )
        overall = round(max(0.0, min(1.0, raw - penalty)), 4)

        # per-claim detail
        claims = self._score_claims(response, context_chunks, combined)

        return {
            "overall":  overall,
            "semantic": round(sem, 4),
            "coverage": round(cov, 4),
            "numeric":  round(num, 4),
            "penalty":  round(penalty, 4),
            "claims":   claims,
        }

    # ------------------------------------------------------------------
    # Signal implementations
    # ------------------------------------------------------------------

    # ---- 1. Semantic Similarity ----------------------------------------

    def _semantic_similarity(self, response: str, chunks: list[str]) -> float:
        """Max cosine similarity between the response and any single chunk."""
        try:
            texts      = [response] + chunks
            embeddings = self._model.encode(texts, normalize_embeddings=True)
            resp_emb   = embeddings[0]
            sims       = [float(np.dot(resp_emb, c)) for c in embeddings[1:]]
            return max(max(sims), 0.0) if sims else 0.0
        except Exception as exc:                           # pragma: no cover
            logger.error("Semantic similarity failed: %s", exc)
            return 0.0

    # ---- 2. Claim Coverage ---------------------------------------------

    def _claim_coverage(
        self, response: str, chunks: list[str], combined: str
    ) -> float:
        """Fraction of declarative claims that are supported."""
        claims = _split_claims(response)
        if not claims:
            return 1.0                                     # nothing to contradict

        ctx_tokens = _tokenise(combined)

        try:
            claim_embs = self._model.encode(claims,  normalize_embeddings=True)
            chunk_embs = self._model.encode(chunks,  normalize_embeddings=True)
        except Exception as exc:                           # pragma: no cover
            logger.error("Claim encoding failed: %s", exc)
            return 0.0

        supported = 0
        for claim, c_emb in zip(claims, claim_embs):
            sims     = [float(np.dot(c_emb, ch)) for ch in chunk_embs]
            best_sim = max(sims) if sims else 0.0

            # lexical fallback
            claim_toks  = _tokenise(claim)
            lex_overlap = len(claim_toks & ctx_tokens) / max(len(claim_toks), 1)

            if best_sim >= CLAIM_SEM_THRESHOLD or lex_overlap >= CLAIM_LEX_THRESHOLD:
                supported += 1

        return supported / len(claims)

    # ---- 3. Numeric Consistency ----------------------------------------

    @staticmethod
    def _numeric_consistency(response: str, combined_context: str) -> float:
        """Fraction of response numbers that also appear in the context."""
        resp_nums = _extract_numbers(response)
        if not resp_nums:
            return 1.0                                     # no numeric claims
        ctx_nums  = _extract_numbers(combined_context)
        matched   = resp_nums & ctx_nums
        return len(matched) / len(resp_nums)

    # ---- 4. Hedge Penalty ----------------------------------------------

    def _hedge_penalty(self, response: str, semantic_sim: float) -> float:
        """
        Small penalty when hedging words appear AND semantic evidence is weak.
        Catches confident fabrication with thin grounding.
        """
        if semantic_sim >= 0.55:                           # evidence is decent
            return 0.0
        words      = set(re.findall(r"[a-z]+", response.lower()))
        hedge_hits = len(words & HEDGE_WORDS)
        hedge_ratio = hedge_hits / max(len(words), 1)
        # scale 0 → 0, full hedge → hedge_max_penalty
        return round(min(self.hedge_max_penalty, hedge_ratio * 0.25 * (1.0 - semantic_sim)), 4)

    # ------------------------------------------------------------------
    # Per-claim detail  (for dashboards / audit)
    # ------------------------------------------------------------------

    def _score_claims(
        self, response: str, chunks: list[str], combined: str
    ) -> list[dict]:
        """Return per-claim grounding detail."""
        claims = _split_claims(response)
        if not claims:
            return []

        ctx_tokens = _tokenise(combined)
        try:
            claim_embs = self._model.encode(claims,  normalize_embeddings=True)
            chunk_embs = self._model.encode(chunks,  normalize_embeddings=True)
        except Exception as exc:                           # pragma: no cover
            logger.error("Claim detail encoding failed: %s", exc)
            return [{"claim": c, "supported": False, "error": str(exc)} for c in claims]

        out = []
        for claim, c_emb in zip(claims, claim_embs):
            sims     = [float(np.dot(c_emb, ch)) for ch in chunk_embs]
            best_idx = int(np.argmax(sims))
            best_sim = sims[best_idx]

            claim_toks  = _tokenise(claim)
            lex_overlap = len(claim_toks & ctx_tokens) / max(len(claim_toks), 1)

            supported = best_sim >= CLAIM_SEM_THRESHOLD or lex_overlap >= CLAIM_LEX_THRESHOLD

            out.append({
                "claim":          claim[:120],
                "supported":      supported,
                "best_chunk_idx": best_idx,
                "semantic_sim":   round(best_sim, 4),
                "lexical_overlap":round(lex_overlap, 4),
                "source_snippet": chunks[best_idx][:100] if supported else None,
            })
        return out