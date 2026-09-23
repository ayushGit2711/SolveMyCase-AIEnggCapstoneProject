"""Indexer and embedding generation pipeline for solvemycase legal corpus.

Embeds statutory provisions and precedents using OpenAI text-embedding-3-small (or local
deterministic fallback for offline testing) and indexes them into Qdrant with BM25.
"""

import hashlib
from typing import List, Optional
import numpy as np
from openai import OpenAI

from solvemycase.config.settings import Settings, get_settings
from solvemycase.data.ingestion.downloader import download_central_legislation
from solvemycase.data.ingestion.normalizer import load_unified_corpus
from solvemycase.data.ingestion.schema import LegalPrecedent, LegalProvision
from solvemycase.data.vectorstore.qdrant_store import QdrantLegalStore


class EmbeddingProvider:
    """Generates dense vector embeddings via OpenAI API or deterministic offline fallback."""

    def __init__(self, settings: Optional[Settings] = None, dim: int = 1536):
        self.settings = settings or get_settings()
        self.dim = dim
        self._openai_client: Optional[OpenAI] = None

        if self.settings.openai_api_key and self.settings.openai_api_key.get_secret_value():
            self._openai_client = OpenAI(api_key=self.settings.openai_api_key.get_secret_value())

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a batch of strings.

        Args:
            texts: List of text snippets.

        Returns:
            List of float vector embeddings of length self.dim.
        """
        if not texts:
            return []

        if self._openai_client:
            try:
                response = self._openai_client.embeddings.create(
                    model=self.settings.openai_embedding_model,
                    input=texts,
                )
                return [item.embedding for item in response.data]
            except Exception as err:
                print(f"[EmbeddingProvider] OpenAI API call failed ({err}). Falling back to local offline embeddings.")

        # Offline deterministic fallback embedding based on hashed token frequencies
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
) -> int:
    """Execute the full ingestion, embedding, and hybrid indexing pipeline.

    Args:
        store: Optional QdrantLegalStore instance. Defaults to embedded storage.
        download_corpus: If True, attempts to download the latest open-india-law parquet.
        force_reset: If True, clears existing vector collection before indexing.

    Returns:
        Number of items successfully indexed.
    """
    settings = get_settings()

    if download_corpus:
        try:
            download_central_legislation()
        except Exception as e:
            print(f"[Indexer] Warning: download failed ({e}), continuing with existing/curated corpus.")

    corpus_data = load_unified_corpus()
    provisions: List[LegalProvision] = corpus_data["provisions"]
    precedents: List[LegalPrecedent] = corpus_data["precedents"]

    print(f"[Indexer] Loaded {len(provisions)} statutory provisions and {len(precedents)} case precedents.")

    embedder = EmbeddingProvider(settings=settings, dim=1536)

    # Prepare texts for dense embedding
    provision_texts = [f"{p.act_name} Section {p.section_number} {p.title or ''}: {p.text}" for p in provisions]
    precedent_texts = [f"{pr.title} {pr.citation or ''}: {pr.text}" for pr in precedents]

    print("[Indexer] Generating dense embeddings...")
    provision_embeddings = embedder.get_embeddings(provision_texts)
    precedent_embeddings = embedder.get_embeddings(precedent_texts)

    if store is None:
        store = QdrantLegalStore(settings=settings, vector_dim=1536)

    if force_reset:
        store.reset_collection()

    count = store.index_provisions_and_precedents(
        provisions=provisions,
        precedents=precedents,
        provision_embeddings=provision_embeddings,
        precedent_embeddings=precedent_embeddings,
    )

    print(f"[Indexer] Successfully indexed {count} items into Qdrant collection '{store.collection_name}'.")
    return count


if __name__ == "__main__":
    run_indexing_pipeline()
