"""Cross-encoder reranking engine for retrieved legal contexts.

Reranks candidate chunks by cross-encoding query and document text pairs,
sharply boosting precision for both statutory sections and judicial holdings.
"""

import math
from typing import List, Optional, Tuple
from solvemycase.config.settings import Settings, get_settings
from solvemycase.data.ingestion.schema import RetrievedContext


def _sigmoid(x: float) -> float:
    """Map an unconstrained logit into [0.0, 1.0] without overflow."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)



class LegalCrossEncoderReranker:
    """Reranks candidate legal chunks using cross-encoder architecture."""

    def __init__(self, settings: Optional[Settings] = None):
        import threading
        self.settings = settings or get_settings()
        self._cross_encoder = None
        self._load_attempted = False
        self._lock = threading.Lock()

    def _get_encoder(self):
        """Lazy load the cross encoder to avoid slowing down startup."""
        with self._lock:
            if not self._load_attempted:
                self._load_attempted = True
                try:
                    from sentence_transformers import CrossEncoder
                    self._cross_encoder = CrossEncoder(self.settings.cross_encoder_model)
                    print(f"[Reranker] Loaded CrossEncoder model: {self.settings.cross_encoder_model}")
                except Exception as err:
                    print(f"[Reranker] Note: sentence-transformers CrossEncoder could not be loaded ({err}). Using lexical cross-scoring.")
            return self._cross_encoder

    def rerank(
        self, query: str, candidates: List[RetrievedContext], top_k: Optional[int] = None
    ) -> List[RetrievedContext]:
        """Rerank candidates based on deep cross-attention similarity to the query.

        Args:
            query: The scenario or targeted search query.
            candidates: List of candidate RetrievedContext items from hybrid retrieval.
            top_k: Number of highest-ranked documents to return.

        Returns:
            Sorted list of RetrievedContext items with updated reranker scores.
        """
        if not candidates:
            return []

        top_k = top_k or self.settings.rerank_top_k
        encoder = self._get_encoder()

        if encoder is not None:
            try:
                pairs = [[query, f"{doc.title} {doc.citation_or_section}\n{doc.text}"] for doc in candidates]
                with self._lock:
                    scores = encoder.predict(pairs)

                scored_candidates: List[Tuple[RetrievedContext, float]] = []
                for doc, score in zip(candidates, scores):
                    calibrated = round(_sigmoid(float(score)), 4)
                    cloned = doc.model_copy()
                    cloned.score = calibrated
                    scored_candidates.append((cloned, calibrated))

                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                return [item[0] for item in scored_candidates[:top_k]]
            except Exception as err:
                print(f"[Reranker] Prediction failed ({err}), falling back to lexical cross-scoring.")

        # Lexical relevance score fallback
        return self._lexical_rerank(query, candidates, top_k)

    def _lexical_rerank(
        self, query: str, candidates: List[RetrievedContext], top_k: int
    ) -> List[RetrievedContext]:
        """Fast fallback lexical overlap scoring."""
        query_words = set(query.lower().split())

        scored: List[Tuple[RetrievedContext, float]] = []
        for doc in candidates:
            doc_text = f"{doc.title} {doc.citation_or_section} {doc.text}".lower()
            doc_words = set(doc_text.split())

            overlap = len(query_words.intersection(doc_words))
            ratio = overlap / max(len(query_words), 1)

            # Boost exact section number matches
            section_boost = 0.5 if (doc.citation_or_section.lower() in query.lower()) else 0.0
            final_score = doc.score * 0.4 + ratio * 0.4 + section_boost * 0.2

            cloned = doc.model_copy()
            cloned.score = round(final_score, 4)
            scored.append((cloned, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in scored[:top_k]]
