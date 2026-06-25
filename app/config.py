from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables / .env file."""

    # Chat LLM provider: "anthropic" (default) or "openai". The embedding model
    # is independent of this choice (see build_llm in main.py).
    llm_provider: str = "anthropic"

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    voyage_api_key: str = ""
    embedding_model: str = "voyage-3.5"
    llm_max_tokens: int = 8192

    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Maximum number of characters of document text passed to a single LLM call.
    max_llm_input_chars: int = 24_000

    # Where the FAISS index is persisted. Empty string disables persistence.
    vector_store_path: str = "data/faiss_index"

    # Read-only demo index, pre-seeded on startup from the documents in seed_path.
    demo_vector_store_path: str = "data/demo_index"
    seed_path: str = "seed"

    # Comma-separated list of allowed browser origins (CORS).
    cors_origins: str = "https://aminskenderi.me,https://www.aminskenderi.me"

    # POST requests per client IP per minute. 0 disables rate limiting.
    rate_limit_per_minute: int = 10

    # Maximum upload size in bytes.
    max_upload_bytes: int = 10 * 1024 * 1024

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
