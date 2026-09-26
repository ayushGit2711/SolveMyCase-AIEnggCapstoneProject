"""Hybrid vector store and retrieval engine utilizing Qdrant and BM25.

Provides dense vector similarity search combined with sparse BM25 keyword matching
and Reciprocal Rank Fusion (RRF) for high-precision statutory and precedent retrieval.
Supports embedded in-process disk mode (no Docker needed) or remote Qdrant instances.
"""

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Set
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from rank_bm25 import BM25Okapi

from solvemycase.config.settings import Settings, get_settings
from solvemycase.data.ingestion.schema import (
    DocumentType,
    LegalDomain,
    LegalPrecedent,
    LegalProvision,
    RetrievedContext,
    acts_share_significant_token,
    sections_match,
)


def tokenize_legal_text(text: str) -> List[str]:
    """Tokenize legal text for BM25 indexing, preserving alphanumeric identifiers like section numbers."""
    cleaned = re.sub(r"[^\w\s\(\)\.-]", " ", text.lower())
    return [token for token in cleaned.split() if len(token) > 1]


# Reciprocal Rank Fusion smoothing constant (Cormack et al., 2009).
DEFAULT_RRF_K = 60
CRIMINAL_ACT_CATEGORY = "criminal"


def reciprocal_rank_fusion(rankings: List[Dict[str, int]], rrf_k: int = DEFAULT_RRF_K) -> Dict[str, float]:
    """Fuse 1-based rank maps: score(doc) = sum over rankings of 1 / (rrf_k + rank)."""
    fused: Dict[str, float] = {}
    for ranks in rankings:
        for cid, rank in ranks.items():
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    return fused


def rank_bm25_hits(
    scores: Any,
    documents: List[RetrievedContext],
    limit: int,
    keep: Callable[[RetrievedContext], bool],
) -> Dict[str, int]:
    """1-based ranks of the best-scoring documents that pass ``keep``.

    Documents are filtered before truncating, so a filtered search still gets up to ``limit`` hits. A document
    with a zero or negative score shares no informative term with the query and is never ranked.
    """
    ranks: Dict[str, int] = {}
    for idx in np.argsort(scores)[::-1]:
        if len(ranks) >= limit or scores[idx] <= 0:
            break
        doc = documents[idx]
        if keep(doc):
            ranks[doc.chunk_id] = len(ranks) + 1
    return ranks


class QdrantLegalStore:
    """Manages legal vector collections, payload metadata, BM25 indexing, and hybrid search.

    Attributes:
        client: QdrantClient instance (local embedded or remote).
        collection_name: Name of Qdrant collection.
        vector_dim: Dimensionality of dense embedding vectors.
        corpus_documents: In-memory registry of documents for BM25 and fast payload lookup.
        bm25_index: BM25Okapi index over the corpus.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        vector_dim: int = 1536,
        in_memory: bool = False,
    ):
        """Initialize Qdrant connection and storage directories.

        Args:
            settings: Optional Settings instance. Defaults to singleton.
            vector_dim: Vector dimension (1536 for OpenAI text-embedding-3-small, 384 for MiniLM).
            in_memory: If True, forces in-memory database (useful for isolated tests).
        """
        self.settings = settings or get_settings()
        self.vector_dim = vector_dim
        self.collection_name = self.settings.qdrant_collection_name
        self.corpus_documents: List[RetrievedContext] = []
        self.bm25_index: Optional[BM25Okapi] = None
        self.loaded_embedder_ids: Set[str] = set()
        self.loaded_fingerprint: Optional[str] = None
        import threading
        self._lock = threading.Lock()

        if in_memory:
            self.client = QdrantClient(location=":memory:")
        elif self.settings.qdrant_url:
            api_key = self.settings.qdrant_api_key.get_secret_value() if self.settings.qdrant_api_key else None
            self.client = QdrantClient(url=self.settings.qdrant_url, api_key=api_key)
        else:
            storage_path = self.settings.resolve_path(self.settings.qdrant_path)
            storage_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(storage_path))

        self._ensure_collection_exists()
        self._load_corpus_from_collection()

    def _ensure_collection_exists(self) -> None:
        """Create the target vector collection if it does not already exist."""
        collections = self.client.get_collections().collections
        existing_names = {c.name for c in collections}

        if self.collection_name not in existing_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.vector_dim,
                    distance=qmodels.Distance.COSINE,
                ),
            )

    def _load_corpus_from_collection(self, batch_size: int = 256) -> None:
        """Rebuild the in-memory document registry and BM25 index from persisted Qdrant payloads.

        The registry and BM25 index live only in memory, so a store opened by a different
        process than the indexer (e.g. the API or UI) would otherwise start empty. That makes
        hybrid search drop every dense hit (no chunk_id -> document mapping) and makes
        exact-section lookups impossible.
        """
        contexts: List[RetrievedContext] = []
        payloads: List[Dict[str, Any]] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                if point.payload:
                    payloads.append(dict(point.payload))
                    contexts.append(RetrievedContext(**point.payload, score=0.0))
            if offset is None:
                break

        self._set_registry(contexts, payloads)

    def _set_registry(self, contexts: List[RetrievedContext], payloads: List[Dict[str, Any]]) -> None:
        """Install the in-memory registry, BM25 index, corpus fingerprint and embedder ids."""
        self.corpus_documents = contexts
        self.loaded_embedder_ids = {str(p["embedder_id"]) for p in payloads if p.get("embedder_id")}
        self.loaded_fingerprint = self.corpus_fingerprint(payloads) if payloads else None
        if contexts:
            corpus_tokens = [
                tokenize_legal_text(f"{doc.title} {doc.citation_or_section} {doc.text}") for doc in contexts
            ]
            self.bm25_index = BM25Okapi(corpus_tokens)
        else:
            self.bm25_index = None

    @staticmethod
    def corpus_fingerprint(payloads: List[Dict[str, Any]]) -> str:
        """Stable hash over document payloads (excluding embedder_id), independent of order."""
        canonical = sorted(
            json.dumps({k: v for k, v in p.items() if k != "embedder_id"}, sort_keys=True, default=str)
            for p in payloads
        )
        return hashlib.sha256("\n".join(canonical).encode("utf-8")).hexdigest()

    @staticmethod
    def provision_payload(prov: LegalProvision, embedder_id: Optional[str] = None) -> Dict[str, Any]:
        """Qdrant payload for a statutory provision."""
        payload = {
            "chunk_id": prov.doc_id,
            "doc_type": DocumentType.STATUTE.value,
            "title": f"{prov.act_name} - Section {prov.section_number}: {prov.title or ''}".strip(),
            "citation_or_section": f"Section {prov.section_number}",
            "act_name": prov.act_name,
            "section_number": prov.section_number,
            "court": None,
            "domain": prov.domain.value,
            "act_category": prov.act_category,
            "text": prov.text,
            "source_url": prov.source_url,
        }
        if embedder_id:
            payload["embedder_id"] = embedder_id
        return payload

    @staticmethod
    def precedent_payload(prec: LegalPrecedent, embedder_id: Optional[str] = None) -> Dict[str, Any]:
        """Qdrant payload for a judgment chunk."""
        payload = {
            "chunk_id": prec.chunk_id,
            "doc_type": DocumentType.PRECEDENT.value,
            "title": f"{prec.title} ({prec.citation or prec.year})",
            "citation_or_section": prec.citation or f"Decision {prec.year}",
            "act_name": None,
            "section_number": None,
            "court": prec.court,
            "domain": prec.domain.value,
            "act_category": None,
            "text": prec.text,
            "source_url": prec.source_url,
        }
        if embedder_id:
            payload["embedder_id"] = embedder_id
        return payload

    def needs_reindex(
        self,
        provisions: List[LegalProvision],
        precedents: List[LegalPrecedent],
        embedder_id: Optional[str] = None,
    ) -> bool:
        """Return True if the index is empty, holds a different corpus, or was built with another embedder.

        Lets a long-running deployment (e.g. Streamlit Cloud) heal itself after corpus edits or a change of
        embedding model instead of serving stale text and links.
        """
        if not self.corpus_documents or not self.loaded_fingerprint:
            return True
        expected = [self.provision_payload(p) for p in provisions] + [self.precedent_payload(p) for p in precedents]
        if self.corpus_fingerprint(expected) != self.loaded_fingerprint:
            return True
        if embedder_id and self.loaded_embedder_ids != {embedder_id}:
            return True
        return False

    def reset_collection(self) -> None:
        """Delete and recreate the collection (used for clean re-indexing)."""
        self.client.delete_collection(collection_name=self.collection_name)
        self._ensure_collection_exists()
        self._set_registry([], [])

    def index_provisions_and_precedents(
        self,
        provisions: List[LegalProvision],
        precedents: List[LegalPrecedent],
        provision_embeddings: List[List[float]],
        precedent_embeddings: List[List[float]],
        embedder_id: Optional[str] = None,
    ) -> int:
        """Batch index both statutory provisions and judicial precedents into Qdrant and BM25.

        Args:
            provisions: List of LegalProvision models.
            precedents: List of LegalPrecedent models.
            provision_embeddings: Dense vector representations of provisions.
            precedent_embeddings: Dense vector representations of precedents.
            embedder_id: Optional id of the embedding space (e.g. 'openai:text-embedding-3-small'), stored in
                every payload so a later process can tell whether its query embeddings match the index.

        Returns:
            Total count of indexed records.
        """
        points: List[qmodels.PointStruct] = []
        contexts: List[RetrievedContext] = []
        payloads: List[Dict[str, Any]] = []
        point_id = 1

        # 1. Process Statutory Provisions
        for prov, emb in zip(provisions, provision_embeddings):
            payload = self.provision_payload(prov, embedder_id)
            points.append(qmodels.PointStruct(id=point_id, vector=emb, payload=payload))
            contexts.append(RetrievedContext(**payload, score=0.0))
            payloads.append(payload)
            point_id += 1

        # 2. Process Judicial Precedents
        for prec, emb in zip(precedents, precedent_embeddings):
            payload = self.precedent_payload(prec, embedder_id)
            points.append(qmodels.PointStruct(id=point_id, vector=emb, payload=payload))
            contexts.append(RetrievedContext(**payload, score=0.0))
            payloads.append(payload)
            point_id += 1

        # Upsert into Qdrant in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=self.collection_name, points=batch)

        # Build in-memory BM25 index, fingerprint and embedder ids
        self._set_registry(contexts, payloads)

        return len(points)

    @staticmethod
    def allowed_domains(domain_filter: Optional[Any]) -> Optional[List[str]]:
        """Domains a filtered search may return, or None for no filter.

        general_dispute (or no domain) means the question was not narrowed to one field, so nothing is
        filtered. A specific domain X also admits general_dispute documents, which hold cross-cutting law
        such as the criminal codes (BNS/BNSS) that apply in every field.
        """
        if domain_filter is None:
            return None
        value = domain_filter.value if isinstance(domain_filter, LegalDomain) else str(domain_filter or "")
        if not value or value == LegalDomain.GENERAL_DISPUTE.value:
            return None
        return [value, LegalDomain.GENERAL_DISPUTE.value]

    def embedding_space_matches(self, embedder_id: Optional[str]) -> bool:
        """Return False when query vectors from ``embedder_id`` can't be compared with the indexed vectors.

        A transient OpenAI failure makes the embedder fall back to hash vectors; searching those against an
        OpenAI-embedded index returns noise, so callers skip dense search then. Unknown ids count as a match.
        """
        if not embedder_id or not self.loaded_embedder_ids:
            return True
        return embedder_id in self.loaded_embedder_ids

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: Optional[List[float]],
        top_k: int = 8,
        domain_filter: Optional[LegalDomain] = None,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> List[RetrievedContext]:
        """Perform hybrid retrieval combining Qdrant dense vector search and BM25 sparse search.

        Combines rankings using Reciprocal Rank Fusion (RRF):
            Score(doc) = 1 / (k + rank_dense) + 1 / (k + rank_bm25)

        Args:
            query_text: Raw natural language legal query string.
            query_embedding: Dense embedding vector for query_text, or None to run BM25 only (e.g. when the
                query was embedded in a different space than the index).
            top_k: Number of combined results to return.
            domain_filter: Optional domain restriction (e.g. MOTOR_VEHICLE_ACCIDENT). Documents tagged
                general_dispute always pass; general_dispute itself applies no filter.
            rrf_k: RRF smoothing constant (default: 60).

        Returns:
            Ranked list of RetrievedContext items with combined fusion scores.
        """
        allowed = self.allowed_domains(domain_filter)
        query_filter = None
        if allowed:
            query_filter = qmodels.Filter(
                must=[qmodels.FieldCondition(key="domain", match=qmodels.MatchAny(any=allowed))]
            )
        return self._fused_search(
            query_text=query_text,
            query_embedding=query_embedding,
            top_k=top_k,
            query_filter=query_filter,
            keep=lambda doc: not allowed or self._domain_value(doc.domain) in allowed,
            rrf_k=rrf_k,
        )

    def criminal_code_search(
        self,
        query_text: str,
        query_embedding: Optional[List[float]],
        top_k: int = 5,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> List[RetrievedContext]:
        """Perform dedicated hybrid retrieval against criminal statutes (BNS, BNSS, IPC, CrPC, PCA 1960).

        Restricts both dense and BM25 search to act_category="criminal".

        Args:
            query_text: Natural language query string.
            query_embedding: Dense embedding vector for query_text, or None to run BM25 only.
            top_k: Number of combined results to return.
            rrf_k: RRF smoothing constant (default: 60).

        Returns:
            Ranked list of criminal statutory contexts.
        """
        query_filter = qmodels.Filter(
            must=[qmodels.FieldCondition(key="act_category", match=qmodels.MatchValue(value=CRIMINAL_ACT_CATEGORY))]
        )
        return self._fused_search(
            query_text=query_text,
            query_embedding=query_embedding,
            top_k=top_k,
            query_filter=query_filter,
            keep=lambda doc: doc.act_category == CRIMINAL_ACT_CATEGORY,
            rrf_k=rrf_k,
        )

    def _fused_search(
        self,
        query_text: str,
        query_embedding: Optional[List[float]],
        top_k: int,
        query_filter: Optional[qmodels.Filter],
        keep: Callable[[RetrievedContext], bool],
        rrf_k: int,
    ) -> List[RetrievedContext]:
        """Dense search plus BM25, fused with RRF. ``query_filter`` and ``keep`` must express the same restriction."""
        limit = top_k * 2
        dense_ranks: Dict[str, int] = {}
        if query_embedding is not None:
            dense_ranks = self._dense_ranks(query_embedding, query_filter, limit)

        bm25_ranks: Dict[str, int] = {}
        if self.bm25_index and self.corpus_documents:
            bm25_scores = self.bm25_index.get_scores(tokenize_legal_text(query_text))
            bm25_ranks = rank_bm25_hits(bm25_scores, self.corpus_documents, limit, keep)

        fused = reciprocal_rank_fusion([dense_ranks, bm25_ranks], rrf_k)
        doc_map = {doc.chunk_id: doc for doc in self.corpus_documents}
        scored_docs = [(doc_map[cid], score) for cid, score in fused.items() if cid in doc_map]
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        results: List[RetrievedContext] = []
        for doc, score in scored_docs[:top_k]:
            cloned = doc.model_copy()
            cloned.score = round(score, 6)
            results.append(cloned)
        return results

    def _dense_ranks(
        self, query_embedding: List[float], query_filter: Optional[qmodels.Filter], limit: int
    ) -> Dict[str, int]:
        """1-based ranks of the nearest indexed chunks to ``query_embedding`` (Qdrant filter applied)."""
        with self._lock:
            try:
                search_response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    query_filter=query_filter,
                    limit=limit,
                )
                dense_hits = search_response.points
            except Exception:
                # Fallback for qdrant-client versions without query_points
                dense_hits = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    query_filter=query_filter,
                    limit=limit,
                )

        ranks: Dict[str, int] = {}
        for rank, hit in enumerate(dense_hits):
            cid = (hit.payload or {}).get("chunk_id")
            if cid and cid not in ranks:
                ranks[cid] = rank + 1
        return ranks

    @staticmethod
    def _domain_value(domain: Any) -> str:
        return domain.value if isinstance(domain, LegalDomain) else str(domain or "")

    def exact_section_search(
        self,
        section_number: str,
        act_name_pattern: Optional[str] = None,
        top_k: int = 3,
    ) -> List[RetrievedContext]:
        """Directly look up provisions by exact section number and optional act pattern.

        Used by the evaluation grounding metric and tests. The verification node no longer re-grounds
        citations against the wider corpus; it only accepts sections that were retrieved for the facts.

        Args:
            section_number: Section number to search (e.g., '106', '166', '173').
            act_name_pattern: Optional partial string match for Act name.
            top_k: Maximum matches to return.

        Returns:
            List of matching RetrievedContext instances.
        """
        matched: List[RetrievedContext] = []

        for doc in self.corpus_documents:
            if doc.doc_type != DocumentType.STATUTE:
                continue
            if sections_match(section_number, doc.citation_or_section):
                if act_name_pattern:
                    doc_act = doc.act_name or doc.title
                    if acts_share_significant_token(act_name_pattern, doc_act) or act_name_pattern.lower() in doc_act.lower():
                        matched.append(doc)
                else:
                    matched.append(doc)

            if len(matched) >= top_k:
                break

        return matched
