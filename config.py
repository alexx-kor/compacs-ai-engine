"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()
load_dotenv(".env.rag", override=False)
load_dotenv(".env.clickhouse", override=False)

StorageBackend = Literal["clickhouse", "json", "auto"]
LLMProvider = Literal["openai", "ollama", "auto"]
EmbeddingProvider = Literal["openai", "ollama", "auto"]


@dataclass(frozen=True)
class Config:
    """Immutable runtime configuration."""

    project_root: Path
    instructions_dir: Path
    local_vector_store_dir: Path
    few_shot_folder: Path
    results_folder: Path

    storage_backend: StorageBackend
    clickhouse_host: str
    clickhouse_port: int
    clickhouse_user: str
    clickhouse_password: str | None
    clickhouse_secure: bool

    llm_provider: LLMProvider
    llm_fallback_enabled: bool
    embedding_provider: EmbeddingProvider
    embedding_fallback_enabled: bool

    openai_api_key: str | None
    openai_model: str
    openai_embedding_model: str
    openai_max_tokens: int
    openai_max_requests_per_min: int
    openai_daily_budget_usd: float

    ollama_host: str
    ollama_model: str
    ollama_keep_alive: str
    ollama_client_timeout: float
    ollama_context_chunks: int
    ollama_chunk_chars: int
    embed_model: str

    chunk_size: int
    chunk_overlap: int
    top_k: int
    rerank_top_k: int
    similarity_threshold: float
    batch_size: int
    max_text_length: int
    min_chunk_size: int
    max_chunks_per_doc: int

    num_ctx: int
    num_predict: int
    temperature: float
    top_p: float
    repeat_penalty: float

    cache_enabled: bool
    cache_ttl: int

    @classmethod
    def from_env(cls) -> Config:
        """Build configuration from environment variables."""
        root = Path(__file__).resolve().parent
        instructions = Path(os.getenv("INSTRUCTIONS_DIR", str(root / "instructions")))
        return cls(
            project_root=root,
            instructions_dir=instructions,
            local_vector_store_dir=Path(
                os.getenv("LOCAL_VECTOR_STORE_DIR", str(root / "data" / "vectors"))
            ),
            few_shot_folder=Path(
                os.getenv("FEW_SHOT_FOLDER", str(instructions / "few_shot"))
            ),
            results_folder=Path(os.getenv("RESULTS_FOLDER", str(root / "data" / "results"))),
            storage_backend=_parse_storage_backend(os.getenv("STORAGE_BACKEND", "auto")),
            clickhouse_host=os.getenv("CLICKHOUSE_HOST", "localhost"),
            clickhouse_port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
            clickhouse_user=os.getenv("CLICKHOUSE_USER", "default"),
            clickhouse_password=os.getenv("CLICKHOUSE_PASSWORD"),
            clickhouse_secure=os.getenv("CLICKHOUSE_SECURE", "false").lower() == "true",
            llm_provider=_parse_llm_provider(os.getenv("LLM_PROVIDER", "auto")),
            llm_fallback_enabled=os.getenv("LLM_FALLBACK_ENABLED", "true").lower() == "true",
            embedding_provider=_parse_embedding_provider(
                os.getenv("EMBEDDING_PROVIDER", "auto")
            ),
            embedding_fallback_enabled=os.getenv(
                "EMBEDDING_FALLBACK_ENABLED", "true"
            ).lower()
            == "true",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            openai_embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            ),
            openai_max_tokens=int(os.getenv("MAX_TOKENS", "800")),
            openai_max_requests_per_min=int(os.getenv("OPENAI_MAX_REQUESTS_PER_MIN", "60")),
            openai_daily_budget_usd=float(os.getenv("OPENAI_DAILY_BUDGET_USD", "10.0")),
            ollama_host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", os.getenv("LLM_MODEL", "llama3.2:3b")),
            ollama_keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "30m"),
            ollama_client_timeout=float(
                os.getenv(
                    "OLLAMA_CLIENT_TIMEOUT",
                    os.getenv("GATEWAY_TIMEOUT", "300"),
                )
            ),
            ollama_context_chunks=int(os.getenv("OLLAMA_CONTEXT_CHUNKS", "3")),
            ollama_chunk_chars=int(os.getenv("OLLAMA_CHUNK_CHARS", "350")),
            embed_model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
            top_k=int(os.getenv("TOP_K", "8")),
            rerank_top_k=int(os.getenv("RERANK_TOP_K", "3")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.35")),
            batch_size=int(os.getenv("BATCH_SIZE", "32")),
            max_text_length=int(os.getenv("MAX_TEXT_LENGTH", "3072")),
            min_chunk_size=int(os.getenv("MIN_CHUNK_SIZE", "100")),
            max_chunks_per_doc=int(os.getenv("MAX_CHUNKS_PER_DOC", "2000")),
            num_ctx=int(os.getenv("NUM_CTX", "4096")),
            num_predict=int(os.getenv("NUM_PREDICT", "400")),
            temperature=float(os.getenv("TEMPERATURE", "0.1")),
            top_p=float(os.getenv("TOP_P", "0.9")),
            repeat_penalty=float(os.getenv("REPEAT_PENALTY", "1.1")),
            cache_enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
            cache_ttl=int(os.getenv("CACHE_TTL", "3600")),
        )

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")

    @property
    def ch_host(self) -> str:
        return self.clickhouse_host

    @property
    def ch_user(self) -> str:
        return self.clickhouse_user

    @property
    def ch_password(self) -> str | None:
        return self.clickhouse_password

    @property
    def ch_secure(self) -> bool:
        return self.clickhouse_secure

    @property
    def docs_folder(self) -> str:
        return str(self.instructions_dir / "raw")


def _openai_key_configured(key: str | None) -> bool:
    if not key or not str(key).strip():
        return False
    normalized = str(key).strip().lower()
    return normalized not in {"user_provided", "changeme", "none", "null"}


def _parse_storage_backend(value: str) -> StorageBackend:
    normalized = value.lower().strip()
    if normalized in ("clickhouse", "json", "auto"):
        return normalized  # type: ignore[return-value]
    raise ValueError(f"invalid STORAGE_BACKEND: {value}")


def _parse_llm_provider(value: str) -> LLMProvider:
    normalized = value.lower().strip()
    if normalized in ("openai", "ollama", "auto"):
        return normalized  # type: ignore[return-value]
    raise ValueError(f"invalid LLM_PROVIDER: {value}")


def _parse_embedding_provider(value: str) -> EmbeddingProvider:
    normalized = value.lower().strip()
    if normalized in ("openai", "ollama", "auto"):
        return normalized  # type: ignore[return-value]
    raise ValueError(f"invalid EMBEDDING_PROVIDER: {value}")


config = Config.from_env()
