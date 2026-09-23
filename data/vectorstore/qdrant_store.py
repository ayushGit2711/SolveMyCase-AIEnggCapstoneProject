"""Hybrid vector store and retrieval engine utilizing Qdrant and BM25.

Provides dense vector similarity search combined with sparse BM25 keyword matching
and Reciprocal Rank Fusion (RRF) for high-precision statutory and precedent retrieval.
Supports embedded in-process disk mode (no Docker needed) or remote Qdrant instances.
"""

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
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
)


def tokenize_legal_text(text: str) -> List[str]:
    """Tokenize legal text for BM25 indexing, preserving alphanumeric identifiers like section numbers."""
    cleaned = re.sub(r"[^\w\s\(\)\.-]", " ", text.lower())
    return [token for token in cleaned.split() if len(token) > 1]


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

    def reset_collection(self) -> None:
        """Delete and recreate the collection (used for clean re-indexing)."""
        self.client.delete_collection(collection_name=self.collection_name)
        self._ensure_collection_exists()
        self.corpus_documents = []
        self.bm25_index = None

    def index_provisions_and_precedents(
        self,
        provisions: List[LegalProvision],
        precedents: List[LegalPrecedent],
        provision_embeddings: List[List[float]],
        precedent_embeddings: List[List[float]],
    ) -> int:
        """Batch index both statutory provisions and judicial precedents into Qdrant and BM25.

        Args:
            provisions: List of LegalProvision models.
            precedents: List of LegalPrecedent models.
            provision_embeddings: Dense vector representations of provisions.
            precedent_embeddings: Dense vector representations of precedents.

        Returns:
            Total count of indexed records.
        """
        points: List[qmodels.PointStruct] = []
        contexts: List[RetrievedContext] = []
        point_id = 1

        # 1. Process Statutory Provisions
        for prov, emb in zip(provisions, provision_embeddings):
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
            points.append(qmodels.PointStruct(id=point_id, vector=emb, payload=payload))
            contexts.append(RetrievedContext(**payload, score=0.0))
            point_id += 1

        # 2. Process Judicial Precedents
        for prec, emb in zip(precedents, precedent_embeddings):
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
            points.append(qmodels.PointStruct(id=point_id, vector=emb, payload=payload))
            contexts.append(RetrievedContext(**payload, score=0.0))
            point_id += 1

        # Upsert into Qdrant in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=self.collection_name, points=batch)

        # Build in-memory BM25 index
        self.corpus_documents = contexts
        corpus_tokens = [tokenize_legal_text(f"{doc.title} {doc.citation_or_section} {doc.text}") for doc in contexts]
        self.bm25_index = BM25Okapi(corpus_tokens)

        return len(points)

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 8,
        domain_filter: Optional[LegalDomain] = None,
        rrf_k: int = 60,
    ) -> List[RetrievedContext]:
        """Perform hybrid retrieval combining Qdrant dense vector search and BM25 sparse search.

        Combines rankings using Reciprocal Rank Fusion (RRF):
            Score(doc) = 1 / (k + rank_dense) + 1 / (k + rank_bm25)

        Args:
            query_text: Raw natural language legal query string.
            query_embedding: Dense embedding vector for query_text.
            top_k: Number of combined results to return.
            domain_filter: Optional domain restriction (e.g. MOTOR_VEHICLE_ACCIDENT).
            rrf_k: RRF smoothing constant (default: 60).

        Returns:
            Ranked list of RetrievedContext items with combined fusion scores.
        """
        # 1. Dense Vector Search via Qdrant
        query_filter = None
        if domain_filter:
            query_filter = qmodels.Filter(
                must=[qmodels.FieldCondition(key="domain", match=qmodels.MatchValue(value=domain_filter.value))]
            )

        try:
            # Qdrant client query points
            search_response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=query_filter,
                limit=top_k * 2,
            )
            dense_hits = search_response.points
        except Exception:
            # Fallback for search API variant
            dense_hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=top_k * 2,
            )

        dense_ranks: Dict[str, int] = {}
        for rank, hit in enumerate(dense_hits):
            cid = hit.payload.get("chunk_id")
            if cid:
                dense_ranks[cid] = rank + 1

        # 2. Sparse BM25 Search
        bm25_ranks: Dict[str, int] = {}
        if self.bm25_index and self.corpus_documents:
            tokens = tokenize_legal_text(query_text)
            bm25_scores = self.bm25_index.get_scores(tokens)
            sorted_indices = np.argsort(bm25_scores)[::-1]

            valid_rank = 1
            for idx in sorted_indices[: top_k * 2]:
                doc = self.corpus_documents[idx]
                if domain_filter and doc.domain != domain_filter:
                    continue
                bm25_ranks[doc.chunk_id] = valid_rank
                valid_rank += 1

        # 3. Reciprocal Rank Fusion (RRF)
        all_chunk_ids = set(dense_ranks.keys()).union(set(bm25_ranks.keys()))
        doc_map = {doc.chunk_id: doc for doc in self.corpus_documents}

        scored_docs: List[Tuple[RetrievedContext, float]] = []
        for cid in all_chunk_ids:
            doc = doc_map.get(cid)
            if not doc:
                continue

            score = 0.0
            if cid in dense_ranks:
                score += 1.0 / (rrf_k + dense_ranks[cid])
            if cid in bm25_ranks:
                score += 1.0 / (rrf_k + bm25_ranks[cid])

            scored_docs.append((doc, score))

        # Sort descending by fused score
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        results: List[RetrievedContext] = []
        for doc, score in scored_docs[:top_k]:
            cloned = doc.model_copy()
            cloned.score = round(score, 6)
            results.append(cloned)

        return results

    def criminal_code_search(
        self,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> List[RetrievedContext]:
        """Perform dedicated hybrid retrieval against criminal code statutes (BNS, BNSS, IPC, CrPC).

        Applies a strict Qdrant payload filter on act_category="criminal" to guarantee
        zero hallucination on statutory provisions.

        Args:
            query_text: Natural language query string.
            query_embedding: Dense embedding vector for query_text.
            top_k: Number of combined results to return.
            rrf_k: RRF smoothing constant (default: 60).

        Returns:
            Ranked list of criminal statutory contexts.
        """
        # 1. Dense search filtered by act_category == "criminal"
        query_filter = qmodels.Filter(
            must=[qmodels.FieldCondition(key="act_category", match=qmodels.MatchValue(value="criminal"))]
        )

        try:
            search_response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=query_filter,
                limit=top_k * 2,
            )
            dense_hits = search_response.points
        except Exception:
            dense_hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=top_k * 2,
            )

        dense_ranks: Dict[str, int] = {}
        for rank, hit in enumerate(dense_hits):
            cid = hit.payload.get("chunk_id")
            if cid:
                dense_ranks[cid] = rank + 1

        # 2. Sparse BM25 filtered by act_category == "criminal"
        bm25_ranks: Dict[str, int] = {}
        if self.bm25_index and self.corpus_documents:
            tokens = tokenize_legal_text(query_text)
            bm25_scores = self.bm25_index.get_scores(tokens)
            sorted_indices = np.argsort(bm25_scores)[::-1]

            valid_rank = 1
            for idx in sorted_indices:
                doc = self.corpus_documents[idx]
                if doc.act_category != "criminal":
                    continue
                bm25_ranks[doc.chunk_id] = valid_rank
                valid_rank += 1
                if valid_rank > top_k * 2:
                    break

        # 3. Reciprocal Rank Fusion
        all_chunk_ids = set(dense_ranks.keys()).union(set(bm25_ranks.keys()))
        doc_map = {doc.chunk_id: doc for doc in self.corpus_documents}

        scored_docs: List[Tuple[RetrievedContext, float]] = []
        for cid in all_chunk_ids:
            doc = doc_map.get(cid)
            if not doc:
                continue

            score = 0.0
            if cid in dense_ranks:
                score += 1.0 / (rrf_k + dense_ranks[cid])
            if cid in bm25_ranks:
                score += 1.0 / (rrf_k + bm25_ranks[cid])

            scored_docs.append((doc, score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)

        results: List[RetrievedContext] = []
        for doc, score in scored_docs[:top_k]:
            cloned = doc.model_copy()
            cloned.score = round(score, 6)
            results.append(cloned)

        return results

    def exact_section_search(
        self,
        section_number: str,
        act_name_pattern: Optional[str] = None,
        top_k: int = 3,
    ) -> List[RetrievedContext]:
        """Directly look up provisions by exact section number and optional act pattern.

        Used by the VerificationNode to re-ground statutory citations before stripping them.

        Args:
            section_number: Section number to search (e.g., '106', '166', '173').
            act_name_pattern: Optional partial string match for Act name.
            top_k: Maximum matches to return.

        Returns:
            List of matching RetrievedContext instances.
        """
        sec_clean = section_number.strip().lower().replace("section", "").replace("sec", "").replace(".", "").strip()
        matched: List[RetrievedContext] = []

        for doc in self.corpus_documents:
            if doc.doc_type != DocumentType.STATUTE:
                continue
            doc_sec = doc.citation_or_section.lower().replace("section", "").replace("sec", "").replace(".", "").strip()
            if doc_sec == sec_clean:
                if act_name_pattern:
                    pattern = act_name_pattern.lower()
                    doc_act = (doc.act_name or doc.title).lower()
                    if pattern in doc_act or any(word in doc_act for word in pattern.split() if len(word) > 3):
                        matched.append(doc)
                else:
                    matched.append(doc)

            if len(matched) >= top_k:
                break

        return matched
