from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables / .env file."""

    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"

    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Maximum number of characters of document text passed to a single LLM call.
    max_llm_input_chars: int = 24_000

    # Where the FAISS index is persisted. Empty string disables persistence.
    vector_store_path: str = "data/faiss_index"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
