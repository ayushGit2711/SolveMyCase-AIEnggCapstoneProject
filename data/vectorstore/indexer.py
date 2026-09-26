"""Indexer and embedding generation pipeline for solvemycase legal corpus.

Embeds statutory provisions and precedents using OpenAI text-embedding-3-small (or local
deterministic fallback for offline testing) and indexes them into Qdrant with BM25.
"""

import hashlib
import logging
import threading
from typing import Dict, List, Optional
import numpy as np
from openai import OpenAI

from solvemycase.config.openai_client import build_openai_client
from solvemycase.config.settings import Settings, get_settings
from solvemycase.data.ingestion.downloader import download_central_legislation
from solvemycase.data.ingestion.normalizer import load_unified_corpus
from solvemycase.data.ingestion.schema import LegalPrecedent, LegalProvision
from solvemycase.data.vectorstore.qdrant_store import QdrantLegalStore

logger = logging.getLogger(__name__)
_ENSURE_INDEX_LOCK = threading.Lock()


class EmbeddingProvider:
    """Generates dense vector embeddings via OpenAI API or deterministic offline fallback."""

    # OpenAI accepts large batches, but smaller requests keep payloads and retries manageable.
    OPENAI_BATCH_SIZE = 128

    def __init__(self, settings: Optional[Settings] = None, dim: int = 1536):
        self.settings = settings or get_settings()
        self.dim = dim
        self._openai_client: Optional[OpenAI] = build_openai_client(self.settings)
        self.last_embedder_id: Optional[str] = None

    @property
    def embedder_id(self) -> str:
        """Embedding space this provider targets: 'openai:<model>' or 'hash:<dim>' (offline)."""
        if self._openai_client:
            return f"openai:{self.settings.openai_embedding_model}"
        return self.offline_embedder_id

    @property
    def offline_embedder_id(self) -> str:
        return f"hash:{self.dim}"

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a batch of strings.

        Requests are sent in batches of at most OPENAI_BATCH_SIZE inputs. If any OpenAI request fails, the
        whole call falls back to offline embeddings so one call never mixes two embedding spaces;
        ``last_embedder_id`` records which space was used.

        Args:
            texts: List of text snippets.

        Returns:
            List of float vector embeddings of length self.dim.
        """
        if not texts:
            return []

        if self._openai_client:
            try:
                vectors: List[List[float]] = []
                for start in range(0, len(texts), self.OPENAI_BATCH_SIZE):
                    batch = texts[start : start + self.OPENAI_BATCH_SIZE]
                    response = self._openai_client.embeddings.create(
                        model=self.settings.openai_embedding_model,
                        input=batch,
                    )
                    vectors.extend(item.embedding for item in response.data)
                if len(vectors) != len(texts):
                    raise ValueError(f"expected {len(texts)} embeddings, got {len(vectors)}")
                self.last_embedder_id = self.embedder_id
                return vectors
            except Exception as err:
                logger.warning("OpenAI embedding API call failed (%s); falling back to offline embeddings.", err)

        # Offline deterministic fallback embedding based on hashed token frequencies
        self.last_embedder_id = self.offline_embedder_id
        return [self._offline_hash_embedding(text) for text in texts]

    def _offline_hash_embedding(self, text: str) -> List[float]:
        """Produce a normalized, deterministic embedding vector for offline testing."""
        vec = np.zeros(self.dim, dtype=np.float32)
        words = text.lower().split()
        if not words:
            vec[0] = 1.0
            return vec.tolist()

        for word in words:
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h % 2 == 0) else -1.0
            vec[idx] += sign

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            vec[0] = 1.0

        return vec.tolist()


def run_indexing_pipeline(
    store: Optional[QdrantLegalStore] = None,
    download_corpus: bool = False,
    force_reset: bool = True,
    settings: Optional[Settings] = None,
    embedder: Optional[EmbeddingProvider] = None,
    corpus_data: Optional[Dict[str, List]] = None,
) -> int:
    """Execute the full ingestion, embedding, and hybrid indexing pipeline.

    Args:
        store: Optional QdrantLegalStore instance. Defaults to embedded storage.
        download_corpus: If True, attempts to download the latest open-india-law parquet.
        force_reset: If True, clears existing vector collection before indexing.
        settings: Optional settings (defaults to the store's settings, then the global settings).
        embedder: Optional embedding provider (defaults to one built from settings).
        corpus_data: Optional pre-loaded corpus ({"provisions": [...], "precedents": [...]}).

    Returns:
        Number of items successfully indexed.
    """
    settings = settings or (store.settings if store is not None else None) or get_settings()

    if download_corpus:
        try:
            download_central_legislation()
        except Exception as e:
            logger.warning("Corpus download failed (%s); continuing with existing/curated corpus.", e)

    corpus_data = corpus_data or load_unified_corpus()
    provisions: List[LegalProvision] = corpus_data["provisions"]
    precedents: List[LegalPrecedent] = corpus_data["precedents"]

    logger.info("Loaded %d statutory provisions and %d case precedents.", len(provisions), len(precedents))

    embedder = embedder or EmbeddingProvider(settings=settings, dim=1536)

    # Prepare texts for dense embedding
    provision_texts = [f"{p.act_name} Section {p.section_number} {p.title or ''}: {p.text}" for p in provisions]
    precedent_texts = [f"{pr.title} {pr.citation or ''}: {pr.text}" for pr in precedents]

    # One call for the whole corpus so every vector comes from the same embedding space.
    logger.info("Generating dense embeddings...")
    all_embeddings = embedder.get_embeddings(provision_texts + precedent_texts)
    provision_embeddings = all_embeddings[: len(provision_texts)]
    precedent_embeddings = all_embeddings[len(provision_texts) :]
    embedder_id = embedder.last_embedder_id or embedder.embedder_id

    if store is None:
        store = QdrantLegalStore(settings=settings, vector_dim=1536)

    if force_reset:
        store.reset_collection()

    count = store.index_provisions_and_precedents(
        provisions=provisions,
        precedents=precedents,
        provision_embeddings=provision_embeddings,
        precedent_embeddings=precedent_embeddings,
        embedder_id=embedder_id,
    )

    logger.info(
        "Indexed %d items into Qdrant collection '%s' (embedder %s).",
        count,
        store.collection_name,
        embedder_id,
    )
    return count


def ensure_index_current(
    store: QdrantLegalStore,
    settings: Optional[Settings] = None,
    embedder: Optional[EmbeddingProvider] = None,
) -> bool:
    """Re-index when the store is empty, holds an outdated corpus, or was embedded with another model.

    Shared by the Streamlit UI and the FastAPI service so a deployment heals itself after corpus changes
    (replacing the old "fewer than 40 documents" rule, which missed edited or re-labelled documents).

    Returns:
        True if the store was re-indexed.
    """
    with _ENSURE_INDEX_LOCK:
        settings = settings or store.settings
        embedder = embedder or EmbeddingProvider(settings=settings, dim=getattr(store, "vector_dim", 1536))
        corpus_data = load_unified_corpus()
        needs_reindex = getattr(store, "needs_reindex", None)
        if needs_reindex is None:
            stale = len(getattr(store, "corpus_documents", []) or []) == 0
        else:
            stale = needs_reindex(corpus_data["provisions"], corpus_data["precedents"], embedder.embedder_id)
        if not stale:
            return False
        run_indexing_pipeline(
            store=store,
            download_corpus=False,
            force_reset=True,
            settings=settings,
            embedder=embedder,
            corpus_data=corpus_data,
        )
        return True


if __name__ == "__main__":
    run_indexing_pipeline()
