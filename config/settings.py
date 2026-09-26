"""Application settings and configuration management for solvemycase.

This module encapsulates all environment variables, paths, model choices, and operational
thresholds using Pydantic Settings. Keeping configuration centralized ensures maintainability,
prevents hardcoded constants, and allows seamless switching between local and production modes.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration settings for the solvemycase application.

    Attributes:
        project_root: Root directory of the solvemycase repository.
        openai_api_key: Optional OpenAI API key wrapped in SecretStr to prevent leakages in logs.
        openai_model_primary: Primary reasoning LLM for planning and multi-agent verification.
        openai_model_fast: Fast LLM for input classification, routing, and guardrails.
        openai_embedding_model: Embedding model for dense vector indexing.
        qdrant_path: Local disk directory for embedded Qdrant storage.
        qdrant_url: Optional remote Qdrant instance URL (overrides embedded mode if provided).
        qdrant_api_key: Optional API key for remote Qdrant Cloud.
        qdrant_collection_name: Name of the vector collection.
        data_cache_dir: Local cache directory for raw parquet and dataset files.
        legislation_source_url: Canonical download URL for Vaquill-AI central legislation parquet.
        verification_strict_mode: When True, rejects any citation not strictly present in context.
        max_hallucination_retries: Maximum agent retry attempts if unverified citations are detected.
        max_retrieved_chunks: Number of initial candidates fetched from hybrid retrieval.
        rerank_top_k: Number of highest-scoring chunks retained after cross-encoder reranking.
        cross_encoder_model: Pretrained cross-encoder model identifier for relevance reranking.
        api_host: Host IP to bind the FastAPI service.
        api_port: Port number for the FastAPI service.
        ui_port: Port number for the Streamlit web dashboard.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Project Paths
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )

    # LLM Configuration (OpenAI as approved by user)
    openai_api_key: Optional[SecretStr] = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
        description="OpenAI API key. If absent, fallback mock mode will be enabled.",
    )
    openai_model_primary: str = Field(
        default="gpt-4o",
        validation_alias="OPENAI_MODEL_PRIMARY",
        description="Flagship LLM for agentic planning and synthesis.",
    )
    openai_model_fast: str = Field(
        default="gpt-4o-mini",
        validation_alias="OPENAI_MODEL_FAST",
        description="Cost-effective, low-latency LLM for guardrails and routing.",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias="OPENAI_EMBEDDING_MODEL",
        description="Embedding model for generating semantic vector representations.",
    )

    openai_timeout_seconds: float = Field(
        default=45.0,
        ge=1.0,
        le=600.0,
        validation_alias="OPENAI_TIMEOUT_SECONDS",
        description="Per-request timeout for OpenAI calls (the SDK default is 600 s).",
    )
    openai_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        validation_alias="OPENAI_MAX_RETRIES",
        description="Automatic retries for failed OpenAI requests.",
    )

    @property
    def openai_model_mini(self) -> str:
        """Alias for openai_model_fast ('gpt-4o-mini')."""
        return self.openai_model_fast

    # Vector Database Configuration (Qdrant Embedded / Remote)
    qdrant_path: Path = Field(
        default=Path("data/qdrant_storage"),
        validation_alias="QDRANT_PATH",
        description="Local directory for embedded Qdrant persistence.",
    )
    qdrant_url: Optional[str] = Field(
        default=None,
        validation_alias="QDRANT_URL",
        description="Remote Qdrant service URL. Overrides local embedded mode if provided.",
    )
    qdrant_api_key: Optional[SecretStr] = Field(
        default=None,
        validation_alias="QDRANT_API_KEY",
        description="Optional API key for authenticated remote Qdrant instances.",
    )
    qdrant_collection_name: str = Field(
        default="open_india_law_corpus",
        validation_alias="QDRANT_COLLECTION_NAME",
        description="Target Qdrant collection name for statutes and judgments.",
    )

    # Data Ingestion & Cache
    data_cache_dir: Path = Field(
        default=Path("data/raw"),
        validation_alias="DATA_CACHE_DIR",
        description="Local cache directory for raw parquet and dataset files.",
    )
    legislation_source_url: str = Field(
        default="https://oss-data-in.vaquill.ai/v2026.08.1/in_central_legislation.parquet",
        validation_alias="LEGISLATION_SOURCE_URL",
        description="Official Vaquill-AI open-india-law central legislation Parquet mirror.",
    )

    # Verification & Hallucination Guardrails
    verification_strict_mode: bool = Field(
        default=True,
        validation_alias="VERIFICATION_STRICT_MODE",
        description="Strictly eliminate citations that cannot be mapped directly to retrieved text.",
    )
    max_hallucination_retries: int = Field(
        default=2,
        validation_alias="MAX_HALLUCINATION_RETRIES",
        description="Retry limit for procedural agent if unverified citations are caught.",
    )

    # Retrieval & Reranker Settings
    max_retrieved_chunks: int = Field(
        default=10,
        validation_alias="MAX_RETRIEVED_CHUNKS",
        description="Number of candidate documents fetched from hybrid BM25 + dense search.",
    )
    rerank_top_k: int = Field(
        default=6,
        validation_alias="RERANK_TOP_K",
        description="Final number of top-ranked context documents fed to the planner.",
    )
    cross_encoder_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        validation_alias="CROSS_ENCODER_MODEL",
        description="HuggingFace model ID for cross-encoder reranking.",
    )
    retrieval_min_confidence: float = Field(
        default=0.25,
        validation_alias="RETRIEVAL_MIN_CONFIDENCE",
        description=(
            "Minimum top-1 reranker score (max over the scenario and its sub-queries) before an extra unfiltered "
            "search widens the candidate pool. This is a recall aid only: whether any law applies is decided by "
            "the applicability check. Cross-encoder scores are sigmoid probabilities; the offline lexical scorer "
            "uses a lower scale, so the widening fires more often offline."
        ),
    )
    applicability_check_enabled: bool = Field(
        default=True,
        validation_alias="APPLICABILITY_CHECK_ENABLED",
        description=(
            "Run one fast-LLM applicability check per query that keeps only provisions and judgments that apply "
            "to the facts, and answers with a coverage gap when none do. Without an LLM (offline) or when the check "
            "fails, general_dispute questions get a coverage-gap answer and other domains keep their top reranked, "
            "domain-filtered sources. Set to false to turn the check off entirely."
        ),
    )
    applicability_candidate_k: int = Field(
        default=10,
        ge=1,
        le=30,
        validation_alias="APPLICABILITY_CANDIDATE_K",
        description="Number of top reranked candidates sent to the applicability check.",
    )
    applicability_timeout_seconds: float = Field(
        default=15.0,
        ge=1.0,
        le=120.0,
        validation_alias="APPLICABILITY_TIMEOUT_SECONDS",
        description="Timeout for the applicability-check LLM call (retried at most once).",
    )
    eval_judge_samples: int = Field(
        default=1,
        validation_alias="EVAL_JUDGE_SAMPLES",
        description="Number of LLM-judge samples per criterion (1 for standard runs, 3 for median variance calibration).",
    )
    telemetry_log_path: Path = Field(
        default=Path("data/telemetry/events.jsonl"),
        validation_alias="TELEMETRY_LOG_PATH",
        description="JSONL file path for runtime inference telemetry and drift tracking.",
    )

    # Network Service Settings
    api_host: str = Field(
        default="0.0.0.0",
        validation_alias="API_HOST",
        description="FastAPI bind address.",
    )
    api_port: int = Field(
        default=8000,
        validation_alias="API_PORT",
        description="FastAPI service port.",
    )
    ui_port: int = Field(
        default=8501,
        validation_alias="UI_PORT",
        description="Streamlit application port.",
    )

    def resolve_path(self, relative_path: Path) -> Path:
        """Resolve a path relative to the project root directory.

        Args:
            relative_path: The path to resolve.

        Returns:
            An absolute Path object.
        """
        if relative_path.is_absolute():
            return relative_path
        return (self.project_root / relative_path).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retrieve the singleton application settings instance.

    Uses lru_cache to avoid repeatedly parsing environment variables during runtime.

    Returns:
        The validated Settings singleton.
    """
    return Settings()
