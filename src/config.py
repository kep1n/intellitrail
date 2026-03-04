from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Type-validated config loaded from env vars. Validates on startup."""

    # AEMET — required in Phase 2+, optional now so Phase 1 runs without it
    aemet_api_key: str = ""

    # SerpAPI — required in Phase 4+
    serp_api_key: str = ""

    # OpenAI — required in Phase 3+
    openai_api_key: str = ""

    # Pinecone API key
    pinecone_api_key: str = ""

    # Set Langsmith api_key and other envs
    langsmith_api_key: str = ""
    langsmith_endpoint: str = ""
    langsmith_tracing: str = ""
    langsmith_project: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Singleton instance imported by other modules
settings = Settings()
