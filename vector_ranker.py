"""
vector_ranker.py
==================
PlanBridge — Intelligent Planning-to-Execution Reconciliation Middleware
SIH 2026 | Problem Statement SIH26122 (Oil India Limited)

PHASE 4 DELIVERABLE — Stage 2: Semantic Vector Ranking
--------------------------------------------------------------
``VectorRanker`` computes semantic similarity between a DPR observation's
raw text and the candidate activity names produced by Phase 3's
``CandidateNarrower``. This is where PlanBridge captures matches that pure
entity filtering misses — e.g. "excavation" vs "trenching" sharing meaning
without sharing any of Stage 1's regex-extracted entities.

Backend selection (in priority order):
    1. Sentence-BERT via ``sentence-transformers`` — tries
       ``all-MiniLM-L6-v2`` first, then ``paraphrase-MiniLM-L6-v2`` as a
       second attempt, computing true dense-embedding cosine similarity.
    2. TF-IDF + cosine similarity (``scikit-learn``) — a lexical-overlap
       fallback when neither transformer model can be loaded (no network
       access to download model weights, package not installed, etc).
    3. ``difflib.SequenceMatcher`` character-level ratio — the final,
       zero-dependency fallback if even scikit-learn is unavailable.

This cascade is not theoretical: in the actual development/CI environment
this module was built and tested in, outbound network access to
huggingface.co is blocked, so ``SentenceTransformer(...)`` genuinely fails
to load and every test in ``test_phase4.py`` exercises the TF-IDF fallback
path for real, not just in a mocked unit test. Whichever backend is active
is recorded in ``VectorRanker.backend_name`` for observability/logging.

Every code path returns similarity scores clipped to [0.0, 1.0], per the
required contract of ``calculate_semantic_similarity``.
"""

from __future__ import annotations

import difflib
import logging
from typing import Optional

log = logging.getLogger("planbridge.vector_ranker")

# --------------------------------------------------------------------------
# Defensive imports — every optional backend degrades gracefully.
# --------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised in minimal environments
    SentenceTransformer = None  # type: ignore[assignment,misc]
    _SENTENCE_TRANSFORMERS_AVAILABLE = False
    try:
        import numpy as np  # numpy may still be present even without sentence-transformers
    except ImportError:  # pragma: no cover
        np = None  # type: ignore[assignment]

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    TfidfVectorizer = None  # type: ignore[assignment,misc]
    cosine_similarity = None  # type: ignore[assignment]
    _SKLEARN_AVAILABLE = False


PRIMARY_MODEL_NAME = "all-MiniLM-L6-v2"
FALLBACK_MODEL_NAME = "paraphrase-MiniLM-L6-v2"


class VectorRanker:
    """
    Computes semantic similarity between a query text and a list of
    candidate texts, using Sentence-BERT embeddings when available and
    degrading gracefully through TF-IDF to raw string similarity otherwise.

    Usage
    -----
        ranker = VectorRanker()
        scores = ranker.calculate_semantic_similarity(
            "150m HDD drilling finished near KP 24+600",
            ["HDD River Crossing Execution at KP 24+600", "Trenching & Backfilling — Section 4B"],
        )
        # scores[0] should score noticeably higher than scores[1]
    """

    def __init__(
        self,
        primary_model_name: str = PRIMARY_MODEL_NAME,
        fallback_model_name: str = FALLBACK_MODEL_NAME,
        use_semantic_model: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        primary_model_name  : first Sentence-BERT model to attempt loading.
        fallback_model_name : second Sentence-BERT model to attempt if the
                               primary fails to load (still offline-safe —
                               if this also fails, we drop to TF-IDF).
        use_semantic_model   : if False, skip Sentence-BERT entirely and go
                               straight to the TF-IDF/SequenceMatcher path.
                               Useful for fast tests or known-offline runs.
        """
        self._model = None
        self.backend_name = "UNINITIALIZED"

        if use_semantic_model and _SENTENCE_TRANSFORMERS_AVAILABLE:
            for model_name in (primary_model_name, fallback_model_name):
                try:
                    self._model = SentenceTransformer(model_name)
                    self.backend_name = f"SENTENCE_TRANSFORMER:{model_name}"
                    log.info("VectorRanker: loaded Sentence-BERT model '%s'.", model_name)
                    break
                except Exception as exc:
                    log.warning(
                        "VectorRanker: could not load Sentence-BERT model '%s' (%s).",
                        model_name, exc,
                    )
        elif use_semantic_model and not _SENTENCE_TRANSFORMERS_AVAILABLE:
            log.warning(
                "VectorRanker: sentence-transformers is not installed — "
                "falling back to TF-IDF/SequenceMatcher similarity."
            )

        if self._model is None:
            if _SKLEARN_AVAILABLE:
                self.backend_name = "TFIDF_COSINE_FALLBACK"
                log.warning(
                    "VectorRanker: no Sentence-BERT model could be loaded "
                    "(offline or package unavailable) — using TF-IDF cosine "
                    "similarity as the semantic-score fallback."
                )
            else:
                self.backend_name = "SEQUENCEMATCHER_FALLBACK"
                log.warning(
                    "VectorRanker: neither Sentence-BERT nor scikit-learn are "
                    "available — using stdlib difflib.SequenceMatcher as the "
                    "last-resort similarity fallback."
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def calculate_semantic_similarity(self, query_text: str, candidate_texts: list[str]) -> list[float]:
        """
        Compute semantic similarity between ``query_text`` and each entry
        in ``candidate_texts``, returning one float in [0.0, 1.0] per
        candidate, in the same order as ``candidate_texts``.

        Edge cases handled without raising:
          * ``candidate_texts`` is empty -> returns ``[]``.
          * ``query_text`` is blank/whitespace-only -> returns all zeros
            (no meaningful comparison is possible).
          * An individual candidate text is blank -> that entry scores
            0.0 regardless of backend, rather than trusting
            SequenceMatcher's quirky "two empty strings are 100% similar"
            behavior.
        """
        if not candidate_texts:
            return []

        if not query_text or not query_text.strip():
            log.warning("VectorRanker: blank query_text supplied; returning all-zero similarity scores.")
            return [0.0] * len(candidate_texts)

        # Track which candidates are blank so every backend treats them
        # identically (score 0.0), regardless of how that backend would
        # otherwise handle an empty string.
        blank_mask = [not (text and text.strip()) for text in candidate_texts]
        safe_candidates = [text if text and text.strip() else " " for text in candidate_texts]

        if self._model is not None:
            try:
                scores = self._sentence_bert_similarity(query_text, safe_candidates)
            except Exception as exc:
                log.error(
                    "VectorRanker: Sentence-BERT encoding failed at runtime (%s); "
                    "falling back to TF-IDF/SequenceMatcher for this call only.",
                    exc,
                )
                scores = self._fallback_similarity(query_text, safe_candidates)
        else:
            scores = self._fallback_similarity(query_text, safe_candidates)

        return [0.0 if is_blank else score for is_blank, score in zip(blank_mask, scores)]

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------
    def _sentence_bert_similarity(self, query_text: str, candidate_texts: list[str]) -> list[float]:
        """Dense-embedding cosine similarity via Sentence-BERT."""
        texts = [query_text] + candidate_texts
        embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        query_vec = embeddings[0]
        candidate_vecs = embeddings[1:]
        # Embeddings are L2-normalized, so dot product == cosine similarity.
        raw_scores = candidate_vecs @ query_vec
        return [float(max(0.0, min(1.0, score))) for score in raw_scores]

    def _fallback_similarity(self, query_text: str, candidate_texts: list[str]) -> list[float]:
        """TF-IDF cosine similarity, degrading further to SequenceMatcher
        if scikit-learn itself is unavailable or fails at runtime (e.g. a
        pathological all-stopword corpus producing an empty vocabulary)."""
        if _SKLEARN_AVAILABLE:
            try:
                return self._tfidf_similarity(query_text, candidate_texts)
            except Exception as exc:
                log.warning(
                    "VectorRanker: TF-IDF similarity failed (%s); falling back to SequenceMatcher.",
                    exc,
                )
        return self._sequence_matcher_similarity(query_text, candidate_texts)

    @staticmethod
    def _tfidf_similarity(query_text: str, candidate_texts: list[str]) -> list[float]:
        corpus = [query_text] + candidate_texts
        vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(corpus)
        query_vec = tfidf_matrix[0:1]
        candidate_vecs = tfidf_matrix[1:]
        sims = cosine_similarity(query_vec, candidate_vecs)[0]
        return [float(max(0.0, min(1.0, s))) for s in sims]

    @staticmethod
    def _sequence_matcher_similarity(query_text: str, candidate_texts: list[str]) -> list[float]:
        query_lower = query_text.lower()
        scores = []
        for text in candidate_texts:
            ratio = difflib.SequenceMatcher(None, query_lower, text.lower()).ratio()
            scores.append(float(max(0.0, min(1.0, ratio))))
        return scores
