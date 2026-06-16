from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    # API keys
    groq_api_key: str
    google_api_key: str

    # Qdrant
    qdrant_url: str = "http://localhost:6333"

    # Langfuse
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: str = "http://localhost:3000"

    # Models
    groq_model: str = "llama-3.3-70b-versatile"
    embed_model: str = "models/gemini-embedding-001"
    embed_dim: int = 3072
    reranker_model: str = "BAAI/bge-reranker-base"

    # Retrieval
    top_k_retrieval: int = 20
    top_k_rerank: int = 5

    # Paths
    data_raw: Path = Path("data/raw")
    data_processed: Path = Path("data/processed")

    # 2. Modern Pydantic V2 configuration style
    model_config = SettingsConfigDict(
        env_file=".env", 
        case_sensitive=False,
        extra="ignore" # Good practice: ignores extra env vars you aren't using
    )


@lru_cache()          # instantiate once, reuse everywhere
def get_settings() -> Settings:
    return Settings()


# Usage in any module:
# from src.config import get_settings
# cfg = get_settings()