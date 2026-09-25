"""Unit tests for solvemycase configuration and settings."""

import os
from pathlib import Path
from solvemycase.config.settings import Settings, get_settings


def test_default_settings():
    """Verify default configuration values adhere to system specifications."""
    settings = Settings()

    assert settings.openai_model_primary == "gpt-4o"
    assert settings.openai_model_fast == "gpt-4o-mini"
    assert settings.openai_embedding_model == "text-embedding-3-small"
    assert settings.qdrant_collection_name == "open_india_law_corpus"
    assert settings.verification_strict_mode is True
    assert settings.max_hallucination_retries == 2
    assert settings.max_retrieved_chunks == 10
    assert settings.rerank_top_k == 6
    assert settings.api_port == 8000
    assert settings.ui_port == 8501


def test_path_resolution(tmp_path: Path):
    """Test relative path resolution against the project root."""
    settings = Settings(project_root=tmp_path)
    relative = Path("data/raw")
    resolved = settings.resolve_path(relative)

    assert resolved == (tmp_path / "data/raw").resolve()


def test_environment_variable_override(monkeypatch):
    """Ensure environment variables correctly override configuration defaults."""
    monkeypatch.setenv("OPENAI_MODEL_PRIMARY", "gpt-4o-custom")
    monkeypatch.setenv("VERIFICATION_STRICT_MODE", "false")
    monkeypatch.setenv("RERANK_TOP_K", "6")

    settings = Settings()

    assert settings.openai_model_primary == "gpt-4o-custom"
    assert settings.verification_strict_mode is False
    assert settings.rerank_top_k == 6


def test_get_settings_singleton():
    """Verify get_settings returns a cached singleton instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
