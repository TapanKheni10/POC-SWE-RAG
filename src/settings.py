from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file = "../env",
        extra = "ignore"
    )

    VOYAGE_API_KEY: str
    PINECONE_API_KEY: str
    GEMINI_API_KEY: str
    GROQ_API_KEY: str
    ANTHROPIC_VERSION: str
    ANTHROPIC_BASE_URL: str
    ANTHROPIC_API_KEY: str
    COHERE_API_KEY: str

Config = Settings()