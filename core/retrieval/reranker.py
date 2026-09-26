"""Cross-encoder reranking engine for retrieved legal contexts.

Reranks candidate chunks by cross-encoding query and document text pairs,
sharply boosting precision for both statutory sections and judicial holdings.
"""

import logging
import math
import threading
from typing import Any, List, Optional, Protocol, Sequence, Tuple

from solvemycase.config.settings import Settings, get_settings
from solvemycase.data.ingestion.schema import RetrievedContext

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    """Map an unconstrained logit into [0.0, 1.0] without overflow."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class CrossEncoderLike(Protocol):
    """Anything that scores (query, document) pairs, e.g. sentence_transformers.CrossEncoder."""

    def predict(self, pairs: Sequence[Sequence[str]]) -> Sequence[float]:
        ...


_CACHE_LOCK = threading.Lock()
_ENCODER_CACHE: dict = {}


class LegalCrossEncoderReranker:
    """Reranks candidate legal chunks using cross-encoder architecture."""

    # Upper bound on (scenario + sub-queries) scored per candidate.
    MAX_RERANK_QUERIES = 12
    # Upper bound on (query, document) pairs per predict call, to keep CPU latency bounded (~1.5 s for 200 pairs).
    MAX_RERANK_PAIRS = 256

    def __init__(
        self,
        settings: Optional[Settings] = None,
        encoder: Optional[CrossEncoderLike] = None,
        lexical_only: bool = False,
    ):
        """Create a reranker.

        Args:
            settings: Application settings (cross-encoder model id, default top_k).
            encoder: Optional pre-built cross-encoder (tests inject fakes here). Loaded lazily when omitted.
            lexical_only: Skip the cross-encoder and always use the lexical scorer (deterministic, offline).
        """
        self.settings = settings or get_settings()
        self._cross_encoder: Optional[Any] = encoder
        self._load_attempted = encoder is not None or lexical_only
        self._lock = threading.Lock()

    def _get_encoder(self):
        """Lazy load the cross encoder (cached per model name across instances) to avoid repeated weight loads."""
        with self._lock:
            if not self._load_attempted:
                self._load_attempted = True
                model_name = self.settings.cross_encoder_model
                with _CACHE_LOCK:
                    if model_name in _ENCODER_CACHE:
                        self._cross_encoder = _ENCODER_CACHE[model_name]
                    else:
                        try:
                            from sentence_transformers import CrossEncoder
                            self._cross_encoder = CrossEncoder(model_name)
                            _ENCODER_CACHE[model_name] = self._cross_encoder
                            logger.info("Loaded CrossEncoder model: %s", model_name)
                        except Exception as err:
                            _ENCODER_CACHE[model_name] = None
                            logger.warning("CrossEncoder could not be loaded (%s); using lexical cross-scoring.", err)
            return self._cross_encoder

    @classmethod
    def _dedupe_queries(cls, query: str, extra_queries: Optional[List[str]] = None) -> List[str]:
        """Return [query] + unique non-empty extra queries (case-insensitive), capped at MAX_RERANK_QUERIES."""
        queries: List[str] = []
        seen = set()
        for q in [query, *(extra_queries or [])]:
            q_clean = (q or "").strip()
            key = q_clean.lower()
            if not q_clean or key in seen:
                continue
            seen.add(key)
            queries.append(q_clean)
        return queries[: cls.MAX_RERANK_QUERIES] or [query or ""]

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedContext],
        top_k: Optional[int] = None,
        extra_queries: Optional[List[str]] = None,
    ) -> List[RetrievedContext]:
        """Rerank candidates based on deep cross-attention similarity to the query.

        Args:
            query: The scenario or targeted search query.
            candidates: List of candidate RetrievedContext items from hybrid retrieval.
            top_k: Number of highest-ranked documents to return.
            extra_queries: Optional decontextualized sub-queries. A candidate's score is the maximum
                calibrated score over the query and these sub-queries, so a provision that answers one
                legal aspect of the facts ranks well even if the raw narrative is phrased differently.
                When candidates x queries exceeds MAX_RERANK_PAIRS, the last sub-queries are dropped
                (the main query is always kept).

        Returns:
            Sorted list of RetrievedContext items with updated reranker scores.
        """
        if not candidates:
            return []

        top_k = top_k or self.settings.rerank_top_k
        queries = self._dedupe_queries(query, extra_queries)
        max_queries = max(1, self.MAX_RERANK_PAIRS // len(candidates))
        queries = queries[:max_queries]
        encoder = self._get_encoder()

        if encoder is not None:
            try:
                docs = [f"{doc.title} {doc.citation_or_section}\n{doc.text}" for doc in candidates]
                # Query-major layout: pair index = query_index * len(docs) + doc_index.
                pairs = [[q, d] for q in queries for d in docs]
                with self._lock:
                    scores = encoder.predict(pairs)
                scores = [float(s) for s in scores]
                if len(scores) != len(pairs):
                    raise ValueError(f"expected {len(pairs)} scores, got {len(scores)}")

                n_docs = len(docs)
                scored_candidates: List[Tuple[RetrievedContext, float]] = []
                for i, doc in enumerate(candidates):
                    best = max(_sigmoid(scores[qi * n_docs + i]) for qi in range(len(queries)))
                    calibrated = round(best, 4)
                    cloned = doc.model_copy()
                    cloned.score = calibrated
                    scored_candidates.append((cloned, calibrated))

                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                return [item[0] for item in scored_candidates[:top_k]]
            except Exception as err:
                logger.warning("Cross-encoder prediction failed (%s); falling back to lexical cross-scoring.", err)

        # Lexical relevance score fallback
        return self._lexical_rerank(queries, candidates, top_k)

    @staticmethod
    def _lexical_score(query: str, doc: RetrievedContext) -> float:
        """Word-overlap relevance of one document to one query (offline fallback)."""
        query_words = set(query.lower().split())
        doc_text = f"{doc.title} {doc.citation_or_section} {doc.text}".lower()
        doc_words = set(doc_text.split())

        overlap = len(query_words.intersection(doc_words))
        ratio = overlap / max(len(query_words), 1)

        # Boost exact section number matches (an empty section string would be "in" every query).
        section = (doc.citation_or_section or "").strip().lower()
        section_boost = 0.5 if section and section in query.lower() else 0.0
        return doc.score * 0.4 + ratio * 0.4 + section_boost * 0.2

    def _lexical_rerank(
        self, queries, candidates: List[RetrievedContext], top_k: int
    ) -> List[RetrievedContext]:
        """Fast fallback lexical overlap scoring (max over the query and sub-queries)."""
        if isinstance(queries, str):
            queries = [queries]

        scored: List[Tuple[RetrievedContext, float]] = []
        for doc in candidates:
            final_score = max(self._lexical_score(q, doc) for q in queries)
            cloned = doc.model_copy()
            cloned.score = round(final_score, 4)
            scored.append((cloned, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in scored[:top_k]]
