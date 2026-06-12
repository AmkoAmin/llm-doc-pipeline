from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables / .env file."""

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-4-8"
    voyage_api_key: str = ""
    embedding_model: str = "voyage-3.5"
    llm_max_tokens: int = 8192

    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Maximum number of characters of document text passed to a single LLM call.
    max_llm_input_chars: int = 24_000

    # Where the FAISS index is persisted. Empty string disables persistence.
    vector_store_path: str = "data/faiss_index"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
